"""
This script extracts class tokens from Vertibench Satellite dataset using the DINOv2 model.
This can be useful if you want to compare to their baseline. Their dataset is coming from Worldstrat
but with L1C images not L2A, and a weird preprocessing (pixels are transformed to be int between 0 and 255).
It preprocesses the images by normalizing them because DINOv2 requires images to be multiples of 14.
Then it extracts the class tokens from the model and saves them in the output directory.
mean and std values per channel were calculated using the script Satellite_mean_std.py. 

"""


import argparse
import logging
import os
import traceback
from logging.handlers import RotatingFileHandler

import matplotlib.pyplot as plt
import numpy as np
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
        self.dataset_mean = [ 81.22718, 71.60518, 68.65638, 71.84505, 80.67904, 107.35618, 121.81858, 117.40219, 131.74495, 39.24643, 1.0040737, 111.348335, 78.19384]
        self.dataset_std = [37.995075, 38.487247, 37.658237, 46.64616, 44.534966, 42.94063, 46.846764, 46.01008, 50.308132, 24.336931, 4.5140524, 60.380325, 51.48326]
        self.current_image_path = None #for debugging purposes
        self.n_image_processed = 0

    def preprocess(self, image, image_path, channel_list):
        self.n_image_processed += 1
        self.current_image_path = image_path  # for debugging purposes
        logger.debug(f"Preprocessing image: {image_path}")
        # Select the mean and std for the specified channels
        selected_mean = [self.dataset_mean[i] for i in channel_list]
        selected_std = [self.dataset_std[i] for i in channel_list]

        transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x).float()),
            transforms.Normalize(selected_mean, selected_std),
            transforms.Resize((168, 168)) # Resize to match requirements of DINOv2 model (multiple of 14)
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
    def split_into_channels(image, channels_list, image_path):
        return [preprocessor.preprocess(image[channels,:,:], image_path, channels) for channels in channels_list]

    images_list = []
    for image_path in image_paths:
        logger.debug(f"Loading image: {image_path}")
        image = np.load(image_path)
        images_list.extend(split_into_channels(image, channels_list, image_path))
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
    
    class_tokens = np.array(class_tokens)
    # print(class_tokens.shape)
    return class_tokens

def extract_class_tokens(data_root, input_dir='Vertibench', output_dir='extracted_features', batch_size=128, channels_list=[[2,3,4]], replace=True):
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
        if 'satellite' in file and (replace or not os.path.exists(os.path.join(output_dir, os.path.relpath(root, input_dir), file.replace('.npy', '_class_token.npy'))))
    ]

    logger.info(f"Found {len(files_to_process)} files to process")

    for i in tqdm(range(0, len(files_to_process), batch_size), desc="Processing images"):
        batch_file_paths = files_to_process[i:i+batch_size]
        try:
            class_tokens = extract_class_token_batch(batch_file_paths, model, channels_list, preprocessor)
            save_class_tokens(batch_file_paths, class_tokens, output_dir, input_dir)
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            logger.error("Traceback:", exc_info=True)

    logger.info("Class token extraction completed.")
    logger.info(f"Total images processed: {preprocessor.n_image_processed}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 class tokens from the Vertibench satellite images."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT"),
        help="Path to the 'satellite_dataset' folder (which contains the "
             "'Vertibench' folder) (default: $DATA_ROOT).",
    )
    parser.add_argument(
        "--output-dir",
        default='extracted_features_giant_rgb_Vertibench',
        help="Output folder for the extracted features, relative to --data-root "
             "unless absolute (default: extracted_features_giant_rgb_Vertibench).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Number of images per batch (default: 4096).",
    )
    args = parser.parse_args()

    if not args.data_root:
        parser.error(
            "--data-root is required (or set the DATA_ROOT environment variable)."
        )

    channels_list = [[3, 2, 1]]
    data_root = os.path.normpath(os.path.expanduser(args.data_root))
    output_dir = args.output_dir
    batch_size = args.batch_size
    
    try:
        extract_class_tokens(data_root=data_root, output_dir=output_dir, channels_list=channels_list, replace=True, batch_size=batch_size)
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")