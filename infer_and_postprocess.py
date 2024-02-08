"""
Example:
python infer_and_postprocess.py /home/shravan/documents/deeplearning/github/alpha_matte_segmentation/notebooks/v2/Review_Dir/FootRest/Original/2a5e854530b04c8f86d1c622ab64f608.jpg 
"""

import os
import sys
sys.path.append("/home/shravan/documents/deeplearning/github/production_code_base/d4seg")

import torch
import cv2
import numpy as np
from PIL import Image
from einops import rearrange
from models.carseg import CarSegmentationModel
from models.model_v2 import SegFormerLightning
from torchvision import transforms
import argparse

import glob
from tqdm import tqdm
import shutil

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.carseg import CarSegmentationModel
from models.model_v2 import SegFormerLightning

import time

# Stage - I

def preprocess(im_path, _H=1024, _W=1024):
    image = Image.open(im_path).convert('RGB')
    # resize images
    image = np.array(image.resize((_H,_W), Image.Resampling.LANCZOS))

    # convert to other format HWC -> CHW
    image = rearrange(image, 'h w c -> c h w')
    image = torch.from_numpy(image)
    return image

def get_predicted_mask(model, image):
    with torch.no_grad():
        model.eval()
        logits = model(image.to(model.device))
    pr_mask = logits.sigmoid()
    return pr_mask

def get_stage1_predicted_mask(model, img_path):
    _H, _W= model.config['resize_height'], model.config['resize_width']
    image_tensor = preprocess(img_path, _H, _W)
    mask_tensor = get_predicted_mask(model, image_tensor)
    
    image_array = image_tensor.numpy().transpose(1, 2, 0)
    mask_array = mask_tensor.cpu().numpy().squeeze()
    
    return mask_array

def get_checkpoint(mit_b5_latest_path, best_or_last='best', base_ckpt_dir="/home/shravan/documents/deeplearning/github/segmentation_models/checkpoints"):
    # Model checkpoint selection
    ckpt_dir = f"{base_ckpt_dir}/{mit_b5_latest_path}"
    best_ckpt_path = os.path.join(ckpt_dir, [p for p in os.listdir(ckpt_dir) if p.startswith('best')][0])
    last_ckpt_path = os.path.join(ckpt_dir, [p for p in os.listdir(ckpt_dir) if p.startswith('last')][0])
    if best_or_last=='best':
        return best_ckpt_path
    else:
        return last_ckpt_path
    
def get_model1(ckpt_path):
    pretrained_model = torch.load(ckpt_path)
    model_config = pretrained_model['hyper_parameters']['config']
    model = CarSegmentationModel(model_config)
    model.load_state_dict(pretrained_model['state_dict'])
    return model    

def get_model2(ckpt_path):
    pretrained_model2 = torch.load(ckpt_path)
    model_config2 = pretrained_model2['hyper_parameters']['config']
    model2 = SegFormerLightning(model_config2)
    model2.load_state_dict(pretrained_model2['state_dict'])
    return model2

# Concatenate along the channel axis (axis=1)
def concat_images(rgb_image, trimap_image):
    concatenated_image = torch.cat((rgb_image, trimap_image), dim=0)
    return concatenated_image

class Resize(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        image, trimap = sample["image"], sample["trimap"]
        
        # Resize image and trimap using cv2
        image = cv2.resize(image, self.size[::-1], interpolation=self.interpolation)
        trimap = cv2.resize(trimap, self.size[::-1], interpolation=cv2.INTER_NEAREST)

        sample["image"], sample["trimap"] = image, trimap
        if "mask" in sample:
            mask = sample["mask"]
            mask = cv2.resize(mask, self.size[::-1], interpolation=self.interpolation)
            sample["mask"] = mask

        return sample

class ToTensor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, sample):
        image, trimap = sample['image'], sample['trimap']
        
        image = image.transpose((2, 0, 1)).astype(np.float32)  # HWC -> CHW
        trimap = np.expand_dims(trimap.astype(np.float32), axis=0)  # HW -> 1HW
        
        # numpy array -> torch tensor
        sample['image'], sample['trimap'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long)
        
        if "mask" in sample:
            mask = sample['mask']
            mask = np.expand_dims(mask.astype(np.float32), axis=0)  # HW -> 1HW
            sample['mask'] = torch.from_numpy(mask)

        # Normalization
        sample['image'] = sample['image'].float() / 255.
        sample['trimap'] = sample['trimap'].float() / 255.

        if "mask" in sample:
            sample['mask'] = sample['mask'].float() / 255.

        return sample

def get_validation_augmentation(config):
    val_transform = [
        Resize(size=(config["resize_height"], config["resize_width"])),
        ToTensor(),
    ]
    return transforms.Compose(val_transform)

def add_background(rgba_image_pil, bg_img_path=None):
    if bg_img_path is None:
        bg_img_path = '/home/shravan/documents/deeplearning/datasets/segmentations_samples/backgrounds/Subrata/white_bg_1600x1200.jpg'

    bg_img_pil = Image.open(bg_img_path)
    bg_img_pil = bg_img_pil.convert(rgba_image_pil.mode).resize(rgba_image_pil.size)

    composite_image = Image.alpha_composite(bg_img_pil, rgba_image_pil)
    return composite_image

def get_image_files(directory_path):
    # Create a list to store the image file paths
    image_files = []

    # Define the allowed image file extensions (you can extend this list based on your requirements)
    allowed_extensions = ['jpg', 'jpeg']

    # Use the glob module to get all files matching the specified pattern
    pattern = os.path.join(directory_path, '*.*')  # Match all files in the given directory
    files = glob.glob(pattern)

    # Iterate through the files and filter out non-image files
    for file_path in files:
        _, extension = os.path.splitext(file_path)
        if extension.lower()[1:] in allowed_extensions:
            image_files.append(file_path)

    return image_files

def compute_largest_segmentation(image_path, dilation_kernel_size=9, blur_kernel_size=1):
    image = cv2.imread(image_path, -1)
    mask = image[:, :, 3]

    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    mask = cv2.dilate(mask, dilation_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    result_mask = np.zeros_like(mask)
    cv2.drawContours(result_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    result = cv2.bitwise_and(image, image, mask=result_mask)
    result = cv2.GaussianBlur(result, (blur_kernel_size, blur_kernel_size), 0)

    return result

def generate_glass_image(bgra, rgba_color=(162, 194, 194, 255), opacity=200):
    mask = bgra[:, :, 3].copy()
    imga = bgra.copy()

    if mask.shape[1] != 1000:
        maskResize = cv2.resize(mask, (1280, 960), interpolation=cv2.INTER_CUBIC)
    else:
        maskResize = mask.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    maskResize = cv2.erode(maskResize.copy(), kernel, iterations=3)

    if mask.shape != maskResize.shape:
        maskResize = cv2.resize(maskResize, mask.shape[::-1], interpolation=cv2.INTER_CUBIC)

    contours, hierarchy = cv2.findContours(maskResize.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    contourMask = np.zeros(mask.shape[:2], dtype="uint8")

    for i in range(len(contours)):
        if hierarchy[0, i, 3] > -1:
            cv2.drawContours(contourMask, contours, i, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    contourMask = cv2.erode(contourMask.copy(), kernel, iterations=2)

    r, g, b, a = rgba_color

    glassImg = np.full_like(bgra[:, :, :4], (b, g, r, a))

    glassMask = np.zeros_like(contourMask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    alpha = cv2.dilate(contourMask.copy(), kernel, iterations=2)

    contours, hierarchy = cv2.findContours(alpha.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    for contour in contours:
        cv2.drawContours(glassMask, [contour], -1, opacity, -1)

    glassImg[:, :, 3] = glassMask

    imgaMask = imga[:, :, 3]
    imgaMask = cv2.medianBlur(imgaMask, 1)
    imga[:, :, 3] = imgaMask
    imga[np.where((imgaMask == 0))] = 0

    image_alpha = imga
    glass_image = glassImg

    return image_alpha, glass_image

def infer_and_post_process(input_image_path, model1, model2, device, save_dir="results"):

    # Stage-I
    pred_mask1 = get_stage1_predicted_mask(model1, input_image_path)

    # Stage-II
    image = cv2.imread(input_image_path)
    input_height, input_width = image.shape[:2]  # Get input image dimensions

    # Ground truth
    filename, _ = os.path.splitext(os.path.basename(input_image_path))

    pred_mask1 = np.uint8(pred_mask1 * 255)
    # Trimap generation using stage-I prediction
    trimap = cv2.resize(pred_mask1, (input_width, input_height), interpolation=cv2.INTER_AREA)

    test_augmentation = get_validation_augmentation(model2.config)
    pre_sample = {'image': image, 'trimap': trimap}

    pre_sample = test_augmentation(pre_sample)
    sample = dict()
    sample.update(pre_sample)

    sample["image"] = concat_images(sample["image"], sample["trimap"])

    x = sample['image']
    x = rearrange(x, 'c h w -> 1 c h w')
    x = x.to(device)

    with torch.no_grad():
        logits = model2(x)

    pred_mask2 = logits.cpu().numpy().squeeze() * 255

    x = rearrange(sample['image'].numpy(), "c h w -> h w c")
    image, trimap = x[:, :, :3], x[:, :, 3:]

    pred_mask1 = cv2.resize(pred_mask1, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    pred_mask1 = rearrange(pred_mask1, "h w -> h w 1") / 255.
    out1 = np.concatenate((image, pred_mask1), axis=-1)

    pred_mask2 = rearrange(pred_mask2, "h w -> h w 1") / 255.
    out2 = np.concatenate((image, pred_mask2), axis=-1)
    cv2.imwrite("temp_out.png", (out2 * 255).astype(np.uint8))

    # Post-processing begins
    result = compute_largest_segmentation("temp_out.png")
    image_alpha, glass_image = generate_glass_image(result)

    # Resize image_alpha and glass_image to input image dimensions
    image_alpha = cv2.resize(image_alpha, (input_width, input_height))
    glass_image = cv2.resize(glass_image, (input_width, input_height))

    # Save the processed images
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cv2.imwrite(os.path.join(save_dir, f"{filename}_transparent.png"), image_alpha)
    cv2.imwrite(os.path.join(save_dir, f"{filename}_glass_image.png"), glass_image)

    # Remove the temporary file
    os.remove("temp_out.png")

    return image_alpha, glass_image

def main(args):
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    model1 = get_model1(args.model1_ckpt).to(device)
    model2 = get_model2(args.model2_ckpt).to(device)

    # Run inference and post-processing
    image_alpha, glass_image = infer_and_post_process(args.input_image_path, model1, model2, device, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and post-processing on an input image.")
    parser.add_argument("input_image_path", type=str, help="Path to the input image.")
    
    parser.add_argument('--model1_ckpt', type=str, default='/home/shravan/documents/deeplearning/github/segmentation_models/checkpoints/20230823/model_20230823_133015/last.ckpt', help='Path to the checkpoint of Model 1')
    parser.add_argument('--model2_ckpt', type=str, default='/home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints//20240102/model_20240102_155705/last.ckpt', help='Path to the checkpoint of Model 2')
    
    parser.add_argument("--output_path", type=str, default="results", help="Path to save the output image directory.")

    args = parser.parse_args()
    main(args)
