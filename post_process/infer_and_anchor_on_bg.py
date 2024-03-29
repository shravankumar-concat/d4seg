"""
Usage:
python infer_and_postprocess.py input_image.jpg --model1_ckpt path/to/custom_model1_checkpoint.pth --model2_ckpt path/to/custom_model2_checkpoint.pth --output_path path/to/output_image.png

or

python infer_and_postprocess.py input_image.jpg 

Defaults for reference: 

input_image_path = '/home/shravan/documents/deeplearning/github/alpha_matte_segmentation/notebooks/v2/Review_Dir/FootRest/Original/2a5e854530b04c8f86d1c622ab64f608.jpg'

Model1 Checkpoint Path: /home/shravan/documents/deeplearning/github/segmentation_models/checkpoints/20230823/model_20230823_133015/last.ckpt

==> checkpoints/model_1_20230823_133015_last.ckpt

Model2 Checkpoint Path: /home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints//20240102/model_20240102_155705/last.ckpt

==> checkpoints/model_2_20240102_155705_last.ckpt

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
from torchvision import transforms
import argparse

import glob
from tqdm import tqdm
import shutil

import torch
from torchvision import transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms


import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import Dataset,DataLoader

from models.carseg import CarSegmentationModel
from models.model_v2 import SegFormerLightning

import time

# Constants
DEFAULT_MODEL1_CKPT = "checkpoints/model_1_20230823_133015_last.ckpt"
DEFAULT_MODEL2_CKPT = "checkpoints/model_2_20240102_155705_last.ckpt"
DEFAULT_BG_IMAGE_PATH = "/home/shravan/documents/deeplearning/datasets/segmentations_samples/backgrounds/Subrata/white_bg_1600x1200.jpg"
DEFAULT_MAX_WIDTH = 1450
DEFAULT_MAX_HEIGHT = 1025
DEFAULT_CENTER_X = 800
DEFAULT_CENTER_Y = 640

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

            
def get_checkpoint(mit_b5_latest_path, best_or_last='best', base_ckpt_dir = "/home/shravan/documents/deeplearning/github/segmentation_models/checkpoints"):
    # Model checkpoint selection
    ckpt_dir = f"{base_ckpt_dir}/{mit_b5_latest_path}"
    best_ckpt_path = os.path.join(ckpt_dir, [p for p in os.listdir(ckpt_dir) if p.startswith('best')][0])
    last_ckpt_path = os.path.join(ckpt_dir, [p for p in os.listdir(ckpt_dir) if p.startswith('last')][0])
    if best_or_last=='best':
        return best_ckpt_path
    else:
        return last_ckpt_path
    
# def get_model1(ckpt_path):
#     pretrained_model = torch.load(ckpt_path)
#     model_config = pretrained_model['hyper_parameters']['config']
#     model = CarSegmentationModel(model_config)
#     model.load_state_dict(pretrained_model['state_dict'])
#     return model    

def load_model1(ckpt_path):
    """Loads the first model."""
    pretrained_model = torch.load(ckpt_path)
    model_config = pretrained_model['hyper_parameters']['config']
    model = CarSegmentationModel(model_config)
    model.load_state_dict(pretrained_model['state_dict'])
    return model

def load_model2(ckpt_path):
    """Loads the second model."""
    pretrained_model = torch.load(ckpt_path)
    model_config = pretrained_model['hyper_parameters']['config']
    model = SegFormerLightning(model_config)
    model.load_state_dict(pretrained_model['state_dict'])
    return model


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


def generate_glass_image(bgra):
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

    # r, g, b = 162, 194, 194
    r, g, b = 111, 133, 133
    a_value = 200
    
    glassImg = np.full_like(bgra[:, :, :3], (b, g, r))
    glassImg = cv2.cvtColor(glassImg, cv2.COLOR_BGR2BGRA)

    glassMask = np.zeros_like(contourMask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    alpha = cv2.dilate(contourMask.copy(), kernel, iterations=2)

    contours, hierarchy = cv2.findContours(alpha.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    for contour in contours:
        cv2.drawContours(glassMask, [contour], -1, a_value, -1)

    glassImg[:, :, 3] = glassMask

    imgaMask = imga[:, :, 3]
    imgaMask = cv2.medianBlur(imgaMask, 1)
    imga[:, :, 3] = imgaMask
    imga[np.where((imgaMask == 0))] = 0
    
    image_alpha = (imga * 255)#S.astype(np.uint8)
    glass_image = glassImg
    
    return image_alpha, glass_image


def infer_and_post_process(input_image_path, model1, model2, device, bg_image_path=None, save_dir = "results"):
        
    # stage-I
    pred_mask1 = get_stage1_predicted_mask(model1, input_image_path)

    #stage-II
    image = cv2.imread(input_image_path)

    # ground truth 
    filename, extension = os.path.splitext(os.path.basename(input_image_path))
    pred_mask1 = np.uint8(pred_mask1*255)
    trimap = cv2.resize(pred_mask1, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    test_augmentation = get_validation_augmentation(model2.config)

    pre_sample = {
        'image': image,
        'trimap': trimap,
    }
    
    if mask is not None:
        pre_sample.update({'mask': mask,})
        
    pre_sample = test_augmentation(pre_sample)

    sample = dict()
    sample.update(pre_sample)

    sample["image"] = concat_images(sample["image"], sample["trimap"])
    x = sample['image']
    x = rearrange(x,'c h w -> 1 c h w')
    x = x.to(device)
    with torch.no_grad():
        logits = model2(x)
    pred_mask2 = logits.cpu().numpy().squeeze()*255
    x = rearrange(sample['image'].numpy(), "c h w -> h w c")
    image, trimap = x[:,:,:3], x[:,:,3:]
    pred_mask1 = cv2.resize(pred_mask1, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    pred_mask1 = rearrange(pred_mask1, "h w -> h w 1")/255.
    out1 = np.concatenate((image, pred_mask1), axis=-1)
    
    pred_mask2 = rearrange(pred_mask2, "h w -> h w 1")/255.
    out2 = np.concatenate((image, pred_mask2), axis=-1)
    cv2.imwrite("temp_out.png",(out2*255).astype(np.uint8))
    
    #post-process begins 
    result = compute_largest_segmentation("temp_out.png")
    image_alpha, glass_image = generate_glass_image(result)
    
    # Remove the temporary file
    os.remove("temp_out.png")
    
    return image_alpha, glass_image


def resize_img(img_trimmed, max_width, max_height):
    height, width, _ = img_trimmed.shape

    if max_width > 0:
        scale_x = max_width / width
        if scale_x > 1:
            img_trimmed = cv2.resize(img_trimmed, None, fx=scale_x, fy=scale_x, interpolation=cv2.INTER_CUBIC)
        else:
            img_trimmed = Image.fromarray(np.uint8(img_trimmed))
            re_height = int(height * scale_x)
            img_trimmed_lan = img_trimmed.resize((max_width, re_height), Image.LANCZOS)
            img_trimmed = np.array(img_trimmed_lan)

    height, width, _ = img_trimmed.shape

    if (height > max_height and max_height != 0) or (max_height != 0 and max_width == 0):
        scale_y = max_height / height
        img_trimmed = Image.fromarray(np.uint8(img_trimmed))
        re_width = int(width * scale_y)
        img_trimmed_lan = img_trimmed.resize((re_width, max_height), Image.LANCZOS)
        img_trimmed = np.array(img_trimmed_lan)

    return img_trimmed

def trim_space(transparent_file_path, max_width, max_height):
    img = cv2.imread(transparent_file_path, -1)
    mask = img[:, :, 3]

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    height, width = mask.shape

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)

        x_left, y_up, w, h = cv2.boundingRect(contour)

        x_right = x_left + w
        y_down = y_up + h

        x_buff = int(w * 0.05)
        y_buff = int(h * 0.05)

        x_left = max(0, x_left - int(x_buff / 2))
        x_right = min(width, x_right + int(x_buff / 2))

        y_up = max(0, y_up - int(y_buff / 2))
        y_down = min(height, y_down + int(y_buff / 2))

    else:
        y_up, y_down, x_left, x_right = 0, height, 0, width

    img_trimmed = img[y_up:y_down, x_left:x_right, :]

    if max_width != 0 or max_height != 0:
        img_trimmed = resize_img(img_trimmed, max_width, max_height)

    return img_trimmed



def compose_images_with_overlay(bg_img_src, overlay_img_src, output_file, max_width, max_height, center_x=None, center_y=None):
    """
    Compose a background image with a transparent overlay.

    Parameters:
    - bg_img_src (str): Path to the background image.
    - overlay_img_src (str): Path to the transparent overlay image (PNG).
    - output_file (str): Path to the output file (JPEG).
    - max_width (int): Maximum width for the output image.
    - max_height (int): Maximum height for the output image.
    - center_x (int): X-coordinate for the center of the overlay. If None, the overlay will be centered.
    - center_y (int): Y-coordinate for the center of the overlay. If None, the overlay will be centered.

    Returns:
    None
    """
    try:
        start1 = int(time.time() * 1000)

        bg_img = Image.open(bg_img_src)
        overlay_img = Image.open(overlay_img_src).convert("RGBA")

        # Resize overlay while maintaining aspect ratio
        overlay_width, overlay_height = overlay_img.size
        aspect_ratio = overlay_width / overlay_height
        print(aspect_ratio)
        new_overlay_width = min(overlay_width, max_width)
        new_overlay_height = int(new_overlay_width / aspect_ratio)
        print(new_overlay_width, new_overlay_height)
        if new_overlay_height > max_height:
            new_overlay_height = max_height
            new_overlay_width = int(new_overlay_height * aspect_ratio)
            print(new_overlay_width, new_overlay_height)

        overlay_img = overlay_img.resize((new_overlay_width, new_overlay_height), Image.Resampling.LANCZOS)

        bg_width, bg_height = bg_img.size
        
        print(overlay_img.size)

        # Set default center coordinates if not provided
        if center_x is None:
            center_x = bg_width // 2
        if center_y is None:
            center_y = bg_height // 2

        overlay_x = center_x - (new_overlay_width // 2)
        overlay_y = center_y - (new_overlay_height // 2)

        start = int(time.time() * 1000)

        # Create a new image with a white background
        canvas = Image.new('RGBA', (bg_width, bg_height), (255, 255, 255, 0))
        canvas.paste(bg_img, (0, 0, bg_width, bg_height))
        canvas.paste(overlay_img, (overlay_x, overlay_y), mask=overlay_img)

        # Convert the image back to RGB before saving
        canvas = canvas.convert("RGB")

        canvas.save(output_file, format='JPEG')

        end = int(time.time() * 1000)
        print(f"Execution time: {end - start} ms")

    except Exception as e:
        raise e
        

def main(args):
    device = torch.device("cuda" 
                          if torch.cuda.is_available() and not args.no_cuda 
                          else "cpu")
    model1 = load_model1(args.model1_ckpt)
    model1 = model1.to(device)
    
    model2 = load_model2(args.model2_ckpt)
    model2 = model2.to(device)

    # Record start time
    start_time = time.time()

    image_alpha, glass_image = infer_and_post_process(
        args.input_image_path, model1, model2, device, save_dir="results", inhouse_dir=None
    )
    


    image1 = Image.fromarray(image_alpha*255)
    image2 = Image.fromarray(glass_image*255)
    
    
    # Extract filename without extension
    filename, _ = os.path.splitext(os.path.basename(args.input_image_path))

    # Save image_alpha and glass_image separately with the input image filename
    image_alpha_path = f"{filename}_image_alpha.png"
    glass_image_path = f"{filename}_glass_image.png"    

    # Save image_alpha and glass_image separately
    cv2.imwrite(image_alpha_path, np.array(image_alpha * 255))
    cv2.imwrite(glass_image_path, np.array(glass_image * 255))
    
    # image_alpha, glass_image are merged together
    merged_image = Image.alpha_composite(image2, image1)
    
    ### Anchoring ForeGround on Background Image
    background_image_path = args.bg_image_path    
    foreground_image_path = './foreground.png'
    output_image_path = args.output_image_path
    
    max_width =    args.max_width
    max_height = args.max_height
    center_x = args.center_x
    center_y = args.center_y

    
    cv2.imwrite(foreground_image_path, np.array(merged_image))  
    if args.crop_tight:
        trimmed_image = trim_space(foreground_image_path, max_width, max_height)
        cv2.imwrite(foreground_image_path, np.array(trimmed_image)) 
        
    compose_images_with_overlay(background_image_path, foreground_image_path, output_image_path, max_width, max_height, center_x, center_y)
    
    # Record end time
    end_time = time.time()
    # Calculate inference time
    inference_time = end_time - start_time
    print(f"Inference and Post processing time: {inference_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Segmentation and Post Processing")
    parser.add_argument("input_image_path", type=str, help="Path to the input image")
    parser.add_argument("--bg_image_path", type=str, default=DEFAULT_BG_IMAGE_PATH, help="Path to the background image")
    parser.add_argument("--output_image_path", type=str, help="Path to save the output image")
    parser.add_argument("--model1_ckpt", type=str, default=DEFAULT_MODEL1_CKPT, help="Path to the model1 checkpoint")
    parser.add_argument("--model2_ckpt", type=str, default=DEFAULT_MODEL2_CKPT, help="Path to the model2 checkpoint")
    parser.add_argument("--image_alpha_path", type=str, default="image_alpha.png", help="Path to save the image_alpha output")
    parser.add_argument("--glass_image_path", type=str, default="glass_image.png", help="Path to save the glass_image output")
    parser.add_argument("--no_cuda", action="store_true", help="Flag to disable CUDA (use CPU)")
        parser.add_argument("--max_width", type=int, default=DEFAULT_MAX_WIDTH, help="Maximum width for the output image")
    parser.add_argument("--max_height", type=int, default=DEFAULT_MAX_HEIGHT, help="Maximum height for the output image")
    parser.add_argument("--center_x", type=int, default=DEFAULT_CENTER_X, help="X-coordinate for the center of the overlay")
    parser.add_argument("--center_y", type=int, default=DEFAULT_CENTER_Y, help="Y-coordinate for the center of the overlay")
    parser.add_argument("--crop_tight", action="store_true", help="Crop the overlay image tightly to its contents")
    

    args = parser.parse_args()
    
    main(args)

    
    
    
    