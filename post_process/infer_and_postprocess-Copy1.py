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
from models.carseg import CarSegmentationModel
from models.model_v2 import SegFormerLightning
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

from PIL import Image
import cv2
import numpy as np
import plotly.express as px
from einops import rearrange
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

            
def get_checkpoint(mit_b5_latest_path, best_or_last='best', base_ckpt_dir = "/home/shravan/documents/deeplearning/github/segmentation_models/checkpoints"):
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


def infer_and_post_process(input_image_path, model1, model2, device, bg_image_path=None, save_dir = "results", inhouse_dir=None):
        
    # stage-I
    pred_mask1 = get_stage1_predicted_mask(model1, input_image_path)

    #stage-II
    image = cv2.imread(input_image_path)

    # ground truth 
    filename, extension = os.path.splitext(os.path.basename(input_image_path))
    
    
    gt_alpha_path = f"{inhouse_dir}/{filename}.jpg"
    
    if os.path.isfile(gt_alpha_path):
        mask = cv2.imread(gt_alpha_path)
        
    else:
        mask=None

    pred_mask1 = np.uint8(pred_mask1*255)
    # trimap generation using stage-I prediction
    # trimap = generate_trimap(pred_mask1)

    # trimap = cv2.resize(trimap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
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
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # Stage - I
    model1 = get_model1(args.model1_ckpt)
    model1 = model1.to(device)

    # Stage - II
    pretrained_model2 = torch.load(args.model2_ckpt)
    model_config2 = pretrained_model2['hyper_parameters']['config']
    model2 = SegFormerLightning(model_config2)
    model2.load_state_dict(pretrained_model2['state_dict'])
    model2 = model2.to(device)

    # image_alpha, glass_image = infer_and_post_process(
    #     args.input_image_path, model1, model2, device, save_dir="results", inhouse_dir=None
    # )
    
    input_image = Image.open(args.input_image_path)
    
    # Get the width and height
    width, height = input_image.size

    
    # Record start time
    start_time = time.time()

    image_alpha, glass_image = infer_and_post_process(
        args.input_image_path, model1, model2, device, save_dir="results", inhouse_dir=None
    )
    
    image_alpha = Image.fromarray(image_alpha*255)
    glass_image = Image.fromarray(glass_image*255)
    
    # Resize the image back to its original dimensions
    image_alpha = image_alpha.resize((width, height))
    glass_image = glass_image.resize((width, height))

    # Record end time
    end_time = time.time()

    # Calculate inference time
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time} seconds")
    
    # Extract filename without extension
    filename, _ = os.path.splitext(os.path.basename(args.input_image_path))

    # Save image_alpha and glass_image separately with the input image filename
    image_alpha_path = f"{filename}_image_alpha.png"
    glass_image_path = f"{filename}_glass_image.png"    

    # Save image_alpha and glass_image separately
    cv2.imwrite(image_alpha_path, np.array(image_alpha * 255))
    cv2.imwrite(glass_image_path, np.array(glass_image * 255))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing and Segmentation")
    parser.add_argument("input_image_path", type=str, help="Path to the input image")
    parser.add_argument("--model1_ckpt", type=str, default="/home/shravan/documents/deeplearning/github/production_code_base/d4seg/checkpoints/model_1_20230823_133015_last.ckpt", help="Path to the model1 checkpoint")
    parser.add_argument("--model2_ckpt", type=str, default="/home/shravan/documents/deeplearning/github/production_code_base/d4seg/checkpoints/model_2_20240102_155705_last.ckpt", help="Path to the model2 checkpoint")
    parser.add_argument("--image_alpha_path", type=str, default="image_alpha.png", help="Path to save the image_alpha output")
    parser.add_argument("--glass_image_path", type=str, default="glass_image.png", help="Path to save the glass_image output")
    parser.add_argument("--no_cuda", action="store_true", help="Flag to disable CUDA (use CPU)")

    args = parser.parse_args()
    
    main(args)
