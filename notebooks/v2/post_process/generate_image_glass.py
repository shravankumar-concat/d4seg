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
import wget

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

def compute_largest_segmentation(image_path, dilation_kernel_size=0, blur_kernel_size=1):
    image = cv2.imread(image_path, -1)
    mask = image[:, :, 3]
    
    if dilation_kernel_size>0:
        dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        mask = cv2.dilate(mask, dilation_kernel, iterations=2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    result_mask = np.zeros_like(mask)
    cv2.drawContours(result_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    result = cv2.bitwise_and(image, image, mask=result_mask)
    result = cv2.GaussianBlur(result, (blur_kernel_size, blur_kernel_size), 0)

    return result

def generate_glass_image(bgra, opacity=120):
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


    glassImg = imga
    glassImg = cv2.cvtColor(glassImg, cv2.COLOR_BGR2BGRA)
    
    glassMask = np.zeros_like(contourMask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    alpha = cv2.dilate(contourMask.copy(), kernel, iterations=2)
    contours, hierarchy = cv2.findContours(alpha.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    for contour in contours:
        cv2.drawContours(glassMask, [contour], -1, opacity, -1)

    glassImg[:, :, 3] = glassMask
    glass_image = glassImg

    return glass_image

def compute_glass_rgba_color(glass_image):
    # Calculate the mean color of the masked image
    return cv2.mean(glass_image, mask=glass_image[:,:,3])

def glass_image_color_norm(bgra, glass_image, bgra_color):
    b, g, r, a = bgra_color
    glassImg = np.full_like(bgra, (b, g, r, a))
    glassImg[:,:,3] = glass_image[:,:,3]
    return glassImg
    

def glass_transform(bgra, opacity):
    glass_image = generate_glass_image(bgra, opacity)
    bgra_color = compute_glass_rgba_color(glass_image)
    transformed_glass_image = glass_image_color_norm(bgra, glass_image, bgra_color)
    return transformed_glass_image

def image_transform(bgra):
    imga = bgra.copy()
    imgaMask = imga[:, :, 3]
    imgaMask = cv2.medianBlur(imgaMask, 1)
    imga[:, :, 3] = imgaMask
    imga[np.where((imgaMask == 0))] = 0
    image_alpha = imga
    return image_alpha

def get_image_and_glass(bgra, opacity=120):
    
    image_alpha = image_transform(bgra)
    glass_image = glass_transform(bgra, opacity)

    return image_alpha, glass_image



def generate_overlay_images(input_image_path, model1, model2, save_dir='./infer_and_postprocess', device='cuda', dilation_kernel_size=9, blur_kernel_size=1, opacity=120):
    """Generates overlay images using two segmentation models."""
    filename, _ = os.path.splitext(os.path.basename(input_image_path))
    # save_dir = os.path.join(save_dir, filename)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    original_image_path = os.path.join(save_dir, 'image.jpg')
    pred_mask1_path = os.path.join(save_dir, 'pred_mask1.png')
    pred_mask2_path = os.path.join(save_dir, 'pred_mask2.png')
    temp_segmentation_path = os.path.join(save_dir, 'temp_segmentation.png')
    largest_segmentation_path = os.path.join(save_dir, 'largest_segmentation.png')
    transparent_image_path = os.path.join(save_dir, 'transparent.png')
    glass_image_path = os.path.join(save_dir, 'glass_image.png')

    image = cv2.imread(input_image_path)
    cv2.imwrite(original_image_path, image)
    image = cv2.imread(original_image_path)

    # Stage-I
    pred_mask1 = get_stage1_predicted_mask(model1, original_image_path)
    pred_mask1 = np.uint8(pred_mask1 * 255)
    cv2.imwrite(pred_mask1_path, pred_mask1)

    pred_mask1 = cv2.imread(pred_mask1_path, cv2.IMREAD_GRAYSCALE)
    pred_mask1 = cv2.resize(pred_mask1, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

    # Stage - II
    trimap = cv2.resize(pred_mask1, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    trimap = np.expand_dims(trimap, axis=-1)

    pre_sample = {'image': image, 'trimap': trimap}

    test_augmentation = get_validation_augmentation(model2.config)
    pre_sample = test_augmentation(pre_sample)

    sample = dict()
    sample.update(pre_sample)
    sample["image"] = concat_images(sample["image"], sample["trimap"])

    x = sample['image']
    x = rearrange(x, 'c h w -> 1 c h w')
    x = x.to(device)

    model2.eval()
    with torch.no_grad():
        logits = model2(x)

    cv2.imwrite(pred_mask2_path, logits.cpu().numpy().squeeze()*255)

    pred_mask2 = cv2.imread(pred_mask2_path, cv2.IMREAD_GRAYSCALE)

    image_height, image_width = image.shape[:2]

    if pred_mask2.shape != image.shape:
        pred_mask2_resized = cv2.resize(pred_mask2, (image_width, image_height), interpolation=cv2.INTER_AREA)
        pred_mask2 = pred_mask2_resized

    pred_mask2 = rearrange(pred_mask2, "h w -> h w 1") / 255.
    out2 = np.concatenate((image / 255., pred_mask2), axis=-1)
    cv2.imwrite(temp_segmentation_path, (out2 * 255).astype(np.uint8))

    result = compute_largest_segmentation(temp_segmentation_path, dilation_kernel_size, blur_kernel_size)
    cv2.imwrite(largest_segmentation_path, result)

    result = cv2.imread(largest_segmentation_path, -1)

    image_alpha, glass_image = get_image_and_glass(result, opacity)

    image_alpha = cv2.resize(image_alpha, (image.shape[1], image.shape[0]))
    glass_image = cv2.resize(glass_image, (image.shape[1], image.shape[0]))

    cv2.imwrite(transparent_image_path, image_alpha)
    cv2.imwrite(glass_image_path, glass_image)

    return {
        'original_image_path': original_image_path,
        'pred_mask1_path': pred_mask1_path,
        'pred_mask2_path': pred_mask2_path,
        'temp_segmentation_path': temp_segmentation_path,
        'largest_segmentation_path': largest_segmentation_path,
        'transparent_image_path': transparent_image_path,
        'glass_image_path': glass_image_path
    }

import os
import wget
def download_from_url(url):
    basename = os.path.basename(url)
    filename, _ = os.path.splitext(basename)
    print(filename)

    if not os.path.isfile(basename):
        print(f"Downloading {basename}")
        wget.download(url)
    else:
        print(f"File {basename} already exists.")
    return basename


def main():
    """Main function for running the script."""
    parser = argparse.ArgumentParser(description="Generate overlay images using two segmentation models")
    parser.add_argument("input_image_path", type=str, help="Path to the input image")
    parser.add_argument("--model1_ckpt", type=str, default="/home/shravan/documents/deeplearning/github/segmentation_models/checkpoints/20230823/model_20230823_133015/last.ckpt", help="Path to the checkpoint of model 1")
    parser.add_argument("--model2_ckpt", type=str, default="/home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints//20240102/model_20240102_155705/last.ckpt", help="Path to the checkpoint of model 2")
    parser.add_argument("--save_dir", type=str, default="./infer_and_postprocess", help="Directory to save the overlay images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--dilation_kernel_size", type=int, default=9, help="Size of the dilation kernel for largest segmentation computation")
    parser.add_argument("--blur_kernel_size", type=int, default=1, help="Size of the blur kernel for largest segmentation computation")
    parser.add_argument("--opacity", type=int, default=180, help="Opacity for the glass effect")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load models
    model1 = get_model1(args.model1_ckpt).to(args.device)
    model2 = get_model2(args.model2_ckpt).to(args.device)
    
    # Generate overlay images
    overlay_images = generate_overlay_images(
        args.input_image_path, model1, model2, args.save_dir, args.device,
        args.dilation_kernel_size, args.blur_kernel_size, args.opacity
    )

    print("Overlay images saved in:", args.save_dir)
    for key, value in overlay_images.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()