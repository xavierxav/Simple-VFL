"""
This script extracts class tokens from the PS images of the satellite dataset using the DINOv2 model.
It preprocesses the images by normalizing them and resizing them to 1050x1050 pixels because DINOv2 requires images to be multiples of 14.
Then it extracts the class tokens from the model and saves them in the output directory. 
The batch size was set to 8 to avoid going out of memory on a 80 GB GPU. It took around 2 hours to process all the images.
mean and std values per channel were calculated using the script Satellite_PS_mean_std.py.
"""

import argparse
import logging
import os
from logging.handlers import RotatingFileHandler

import numpy as np
import tifffile as tiff
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel

# Setup logging
log_file = 'feature_extraction.log'

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)
console_handler.setFormatter(console_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class Preprocessor:
    def __init__(self):
        self.Channels_mean = [1128.9812, 1011.69165, 845.47516, 1919.2701]
        self.Channels_std = [1441.1455, 1388.0193, 1411.4835, 1362.1277]
        self.image_paths = []
        self.current_image_path = None  # for debugging purposes

    def preprocess(self, image, image_path, channels):
        self.image_paths.append(image_path)
        self.current_image_path = image_path  # for debugging purposes
        logger.debug(f"Preprocessing image: {image_path}")
        
        # Convert image to channels-first format
        image = np.transpose(image, (2, 0, 1))
        
        transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x).float()),
            transforms.Resize((1050, 1050)),
            transforms.Normalize(
                mean=[self.Channels_mean[c] for c in channels],
                std=[self.Channels_std[c] for c in channels]
            )
        ])
        return transform(image)

# Function to save class tokens
def save_class_tokens(batch_file_paths, class_tokens, output_dir, input_dir):
    for file_path, class_token in zip(batch_file_paths, class_tokens):
        relative_path = os.path.relpath(os.path.dirname(file_path), input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(output_path, exist_ok=True)
        class_token_file_path = os.path.join(output_path, os.path.basename(file_path).replace('.tiff', '_class_token.npy'))
        np.save(class_token_file_path, class_token)

def extract_class_token_batch(image_paths, model, channels_list, preprocessor):
    nb_channels_groups = len(channels_list)
    nb_images = len(image_paths)


    images_list = []
    for image_path in image_paths:
        logger.debug(f"Loading image: {image_path}")
        images = tiff.imread(image_path)
        for channels in channels_list:
            image = images[:,:,channels]
            image = preprocessor.preprocess(image, image_path, channels)
            images_list.append(image)

    images_list = torch.stack(images_list).to(device)
    images_list = images_list.to(torch.float32)
    
    with torch.no_grad():
        try:
            features = model(images_list)
            logger.debug(f"Model output type: {type(features)}")
            class_tokens = features.pooler_output.cpu().numpy()
        except Exception as e:
            logger.error(f"Error during model inference: {str(e)}")
            raise
    
    class_tokens = np.array(class_tokens).reshape(nb_images, nb_channels_groups, -1)
    return class_tokens

def extract_class_tokens(data_root, output_dir, batch_size, channels_list, input_dir='hr_dataset', replace=True):
    input_dir = os.path.normpath(os.path.join(data_root, input_dir))
    output_dir = os.path.normpath(os.path.join(data_root, output_dir))
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading model: facebook/dinov2-giant")
    model = AutoModel.from_pretrained('facebook/dinov2-giant').to(device)
    model.eval()

    preprocessor = Preprocessor()

    logger.info("Scanning for files to process...")

    files_to_process = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files
        if file.endswith('ps.tiff') and (replace or not os.path.exists(os.path.join(output_dir, os.path.relpath(root, input_dir), file.replace('.tiff', '_class_token.npy'))))
    ]

    logger.info(f"Found {len(files_to_process)} files to process")

    for i in tqdm(range(0, len(files_to_process), batch_size), desc="Processing images"):
        batch_file_paths = files_to_process[i:i+batch_size]
        try:
            class_tokens = extract_class_token_batch(batch_file_paths, model, channels_list, preprocessor)
            save_class_tokens(batch_file_paths, class_tokens, output_dir, input_dir)
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")

    logger.info("Class token extraction completed.")
    logger.info(f"Total images processed: {len(preprocessor.image_paths)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 class tokens from the SPOT pansharpened images."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT"),
        help="Path to the 'satellite_dataset' folder (which contains "
             "'hr_dataset') (default: $DATA_ROOT).",
    )
    parser.add_argument(
        "--output-dir",
        default='extracted_features_giant_ps_normalized',
        help="Output folder for the extracted features, relative to --data-root "
             "unless absolute (default: extracted_features_giant_ps_normalized).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images per batch (default: 8).",
    )
    args = parser.parse_args()

    if not args.data_root:
        parser.error(
            "--data-root is required (or set the DATA_ROOT environment variable)."
        )

    channels_list = [[0, 1, 2]]  # this corresponds to the RGB channels for PS images
    data_root = os.path.normpath(os.path.expanduser(args.data_root))
    output_dir = args.output_dir
    # even with an 80 GB GPU the batch size had to be lowered to 8 to avoid OOM
    batch_size = args.batch_size
    
    try:
        extract_class_tokens(data_root=data_root, output_dir=output_dir, channels_list=channels_list, replace=False, batch_size=batch_size)
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")