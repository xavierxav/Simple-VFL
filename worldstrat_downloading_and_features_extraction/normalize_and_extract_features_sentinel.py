"""
This script extracts class tokens from the Sentinel L2A images of the satellite dataset using the DINOv2 model.
It preprocesses the images by normalizing them because DINOv2 requires images to be multiples of 14.
Then it extracts the class tokens from the model and saves them in the output directory.
mean and std values per channel were calculated using the script Satellite_mean_std.py.

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
        self.Channels_mean = [0.12543254, 0.13614272, 0.16135749, 0.17593418, 0.21157593, 0.26788434, 0.2906735, 0.2978914, 0.30594572, 0.33509994, 0.26702592, 0.20812556]
        self.Channels_std = [0.19563292, 0.19202784, 0.18203962, 0.18508251, 0.18341799, 0.16249457, 0.15715967, 0.15798034, 0.15200324, 0.22788922, 0.1411692, 0.13475654]
        self.image_paths = []
        self.current_image_path = None  # Add this line to store the current image path

    def preprocess(self, image, image_path, channels):
        self.image_paths.append(image_path)
        self.current_image_path = image_path  # Set the current image path
        logger.debug(f"Preprocessing image: {image_path}")
        transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x).float()),
            transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # Ensure [C, H, W] format
            transforms.Resize((168, 168)),
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

def extract_class_tokens(data_root, output_dir, batch_size, channels_list, input_dir, replace=True):
    input_dir = os.path.normpath(os.path.join(data_root, input_dir))
    output_dir = os.path.normpath(os.path.join(data_root, output_dir))
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading model: facebook/dinov2-giant")
    model = AutoModel.from_pretrained('facebook/dinov2-giant').to(device)
    model.eval()

    preprocessor = Preprocessor()

    logger.info("Scanning for files to process...")

    files_to_process = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files
        if file.endswith('L2A_data.tiff') and (replace or not os.path.exists(os.path.join(output_dir, os.path.relpath(root, input_dir), file.replace('.tiff', '_class_token.npy'))))
    ]

    logger.info(f"Found {len(files_to_process)} files to process")

    for i in tqdm(range(0, len(files_to_process), batch_size), desc="Processing images"):
        batch_file_paths = files_to_process[i:i+batch_size]
        
        class_tokens = extract_class_token_batch(batch_file_paths, model, channels_list, preprocessor)
        save_class_tokens(batch_file_paths, class_tokens, output_dir, input_dir)

    logger.info("Class token extraction completed.")
    logger.info(f"Total images processed: {len(preprocessor.image_paths)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 class tokens from the Sentinel L2A images."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT"),
        help="Path to the 'satellite_dataset' folder (which contains "
             "'lr_dataset') (default: $DATA_ROOT).",
    )
    parser.add_argument(
        "--output-dir",
        default='extracted_features_giant_rgb_light_preprocess',
        help="Output folder for the extracted features, relative to --data-root "
             "unless absolute (default: extracted_features_giant_rgb_light_preprocess).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images per batch (default: 16).",
    )
    args = parser.parse_args()

    if not args.data_root:
        parser.error(
            "--data-root is required (or set the DATA_ROOT environment variable)."
        )

    # this script extracts class tokens for each channel list specified, so if you want
    # to extract from channels other than RGB, you can add them to the list. For example
    # [[3,2,1], [4,4,4]] for extracting the 4th channel as a grayscale image.
    channels_list = [[3, 2, 1]]  # RGB channels
    data_root = os.path.normpath(os.path.expanduser(args.data_root))
    output_dir = args.output_dir
    batch_size = args.batch_size
    input_dir = 'lr_dataset'

    extract_class_tokens(data_root=data_root, output_dir=output_dir, channels_list=channels_list, input_dir=input_dir, replace=True, batch_size=batch_size)
