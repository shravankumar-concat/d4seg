import os
import sys
import glob
from tqdm import tqdm
sys.path.append('/home/shravan/documents/deeplearning/github/segmentation_models/')
import torch
from torchvision.transforms import PILToTensor

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

from models.carseg import CarSegmentationModel

# from utils.inference_model import CarSegmenter

# from dataset import D4SegmentationDataset, SimpleD4SegDataset
# from dataset import get_training_augmentation, get_validation_augmentation

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


def segment_PIL(img_path, model):
    _H, _W= model.config['resize_height'], model.config['resize_width']
    image_tensor = preprocess(img_path, _H, _W)
    mask_tensor = get_predicted_mask(model, image_tensor)
    
    image_array = image_tensor.numpy().transpose(1, 2, 0)
    mask_array = mask_tensor.cpu().numpy().squeeze()
    mask_array = np.expand_dims(mask_array, axis=-1)  
    
    seg_img = image_array*mask_array
    
    image_pil = Image.fromarray(np.uint8(image_array))  # Assuming image range is [0, 1] and converting to [0, 255]
    mask_pil = Image.fromarray(np.uint8(mask_array.squeeze() * 255))    # Assuming mask range is [0, 1] and converting to [0, 255]
    seg_img_pil = Image.fromarray(np.uint8(seg_img))
    
    return image_pil, mask_pil, seg_img_pil
         
def make_background_transparent(rgb_image, segmentation_mask, output_path=None):
    rgba_image = rgb_image.convert("RGBA")
    segmentation_mask = segmentation_mask.resize(rgba_image.size)
    rgba_image.putalpha(segmentation_mask)

    if output_path:
        rgba_image.save(output_path, format="PNG")

    return rgba_image

def add_bg(rgba_image_pil, bg_img_path=None):
    if bg_img_path is None:
        bg_img_path = '/home/shravan/documents/deeplearning/datasets/segmentations_samples/backgrounds/Subrata/white_bg_1600x1200.jpg'

    bg_img_pil = Image.open(bg_img_path)
    bg_img_pil = bg_img_pil.convert(rgba_image_pil.mode).resize(rgba_image_pil.size)

    composite_image = Image.alpha_composite(bg_img_pil,rgba_image_pil)
    return composite_image
            
def get_checkpoint(mit_b5_latest_path, best_or_last='best', base_ckpt_dir = "/home/shravan/documents/deeplearning/github/segmentation_models/checkpoints"):
    # Model checkpoint selection
    # mit_b5_latest_path = "20230807/model_20230807_161902/" # baseline1
    # mit_b5_latest_path = "20230809/model_20230809_174900" # baseline2
    # mit_b5_latest_path= "20230811/model_20230811_163027/"  # baseline 3
    # mit_b5_latest_path = "20230817/model_20230817_120203"
    # mit_b5_latest_path = "20230817/model_20230817_180942/"
    # mit_b5_latest_path_v6 = "20230823/model_20230823_133015/"
    
    ckpt_dir = f"{base_ckpt_dir}/{mit_b5_latest_path}"
    best_ckpt_path = os.path.join(ckpt_dir, [p for p in os.listdir(ckpt_dir) if p.startswith('best')][0])
    last_ckpt_path = os.path.join(ckpt_dir, [p for p in os.listdir(ckpt_dir) if p.startswith('last')][0])
    if best_or_last=='best':
        return best_ckpt_path
    else:
        return last_ckpt_path
    
def get_model(ckpt_path):
    pretrained_model = torch.load(ckpt_path)
    model_config = pretrained_model['hyper_parameters']['config']
    model = CarSegmentationModel(model_config)
    model.load_state_dict(pretrained_model['state_dict'])
    return model    

# Generate Trimap
def erode_and_dilate(mask, k_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)

    eroded = cv2.erode(mask, kernel, iterations=iterations)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)

    trimap = np.full(mask.shape, 128)
    trimap[eroded >= 254] = 255
    trimap[dilated <= 1] = 0
    
    trimap = np.uint8(trimap)

    return trimap

def generate_trimap(mask, threshold=0.05, iterations=3, kernel_size=3):
    threshold = threshold * 255

    trimap = mask.copy()
    
    if isinstance(trimap, Image.Image):
        trimap = np.uint8(trimap)
    
    # Erode and dilate the mask with the specified kernel_size
    trimap = erode_and_dilate(trimap, k_size=(kernel_size, kernel_size), iterations=iterations)
    return trimap

def get_trimap_array(image_pil, mask_pil, kernel_size=3):
    trimap_array = generate_trimap(mask_pil, kernel_size=kernel_size)
    trimap_tensor = torch.tensor(trimap_array, dtype=torch.float32)
    trimap_tensor = rearrange(trimap_tensor, 'h w -> 1 h w')
    
    image_array = np.array(image_pil)
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    image_tensor = rearrange(image_tensor, 'h w c -> c h w')
    
    image_trimap_tensor = concat_images(image_tensor, trimap_tensor)  # concatenate along the channel axis 
    image_trimap_array = rearrange(image_trimap_tensor, 'c h w -> h w c').numpy().astype(np.uint8)
    
    return image_array, trimap_array, image_trimap_array    


def get_trimap_array_v21(image_pil, mask_pil, kernel_size=3):
    trimap_array = generate_trimap(mask_pil, kernel_size=kernel_size)
    trimap_tensor = torch.tensor(trimap_array, dtype=torch.float32)
    trimap_tensor = rearrange(trimap_tensor, 'h w -> 1 h w')
    
    image_array = np.array(image_pil)
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    image_tensor = rearrange(image_tensor, 'h w c -> c h w')
    
    image_trimap_tensor = concat_images(image_tensor, trimap_tensor)  # concatenate along the channel axis 
    image_trimap_array = rearrange(image_trimap_tensor, 'c h w -> h w c').numpy().astype(np.uint8)
    
    return image_array, trimap_array, image_trimap_array   

# Concatenate along the channel axis (axis=1)
def concat_images(rgb_image, trimap_image):
    concatenated_image = torch.cat((rgb_image, trimap_image), dim=0)
    return concatenated_image

def concatenate_images_horizontally(image_pil, mask_pil, segmented_image_pil):
    widths = [img.size[0] for img in (image_pil, mask_pil, segmented_image_pil)]
    heights = [img.size[1] for img in (image_pil, mask_pil, segmented_image_pil)]

    max_height = max(heights)
    total_width = sum(widths)
    concatenated_image = Image.new('RGBA', (total_width, max_height))
    positions = [(0, 0), (widths[0], 0), (widths[0] + widths[1], 0)]

    for img, pos in zip((image_pil, mask_pil, segmented_image_pil), positions):
        concatenated_image.paste(img, pos)

    return concatenated_image

def concatenate_images_horizontally(image_list):
    widths = [img.size[0] for img in image_list]
    heights = [img.size[1] for img in image_list]

    max_height = max(heights)
    total_width = sum(widths)

    concatenated_image = Image.new('RGBA', (total_width, max_height))
    positions = [(sum(widths[:i]), 0) for i in range(len(image_list))]

    for img, pos in zip(image_list, positions):
        concatenated_image.paste(img, pos)

    return concatenated_image

def save_image(image_pil, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    image_pil.save(save_path)


def segment_and_save_trimaps(model, root_dir, bg_img_path=None, save=True, save_dir_path=None, show=False, n_samples=None):
    
    # Save the segmented image to the "results" directory
    if save_dir_path is None:
        save_dir_path = f"{root_dir}/trimaps/{model.config['project_name']}"
    else:
        save_dir_path = f"{root_dir}/{save_dir_path}"
        
    print(save_dir_path)
    save_dir = {}
    # save_dir['mask'] = f"{save_dir_path}/mask"
    # save_dir['segmented'] = f"{save_dir_path}/segmented"
    # save_dir['composited'] = f"{save_dir_path}/composited"
    save_dir['concatinated'] = f"{save_dir_path}/concatinated"
    save_dir['trimap'] = f"{save_dir_path}/trimap"
    
    seg_test_files = sorted([os.path.join(root_dir,p) for p in os.listdir(root_dir) if p.endswith('.jpg')])
    
    # Control number of example for testing
    if n_samples is None:
        seg_test_files = seg_test_files
    else:
        seg_test_files = seg_test_files[:n_samples]
        
    
    # Run the segmentation over the test_sample images
    for img_path in tqdm(seg_test_files):
        image_pil, mask_pil, seg_pil = segment_PIL(img_path, model)
        image_array, trimap_array, image_trimap_array = get_trimap_array(image_pil, mask_pil)
        trimaps_pil_list = list(map(Image.fromarray, [image_array, trimap_array, image_trimap_array]))
        concatinated_trimaps = concatenate_images_horizontally(*trimaps_pil_list)
        
        if show:
            concatinated_trimaps.show()
        
        # Save the segmented image to the "results" directory
        if save:
            file_name = img_path.split('/')[-1].replace('.jpg', '.png')
            # save_image(mask_pil, save_dir['mask'], file_name)
            # save_image(seg_img_pil, save_dir['segmented'], file_name)
            # save_image(composite_image, save_dir['composited'], file_name)
            save_image(trimaps_pil_list[1], save_dir['trimap'], file_name)
            save_image(concatinated_trimaps, save_dir['concatinated'], file_name)
            
            

def segment_and_save_pred_masks(model, root_dir, bg_img_path=None, save=True, save_dir_path=None, show=False, n_samples=None):
    
    # Save the segmented image to the "results" directory
    if save_dir_path is None:
        save_dir_path = f"{root_dir}/pred_masks/{model.config['project_name']}"
    else:
        save_dir_path = f"{save_dir_path}"
        
    print(save_dir_path)
    save_dir = {}
    save_dir['pred_mask'] = f"{save_dir_path}"
    
    seg_test_files = sorted([os.path.join(root_dir,p) for p in os.listdir(root_dir) if p.endswith('.jpg')])
    # Control number of example for testing
    if n_samples is None:
        seg_test_files = seg_test_files
    else:
        seg_test_files = seg_test_files[:n_samples]
        
    
    # Run the segmentation over the test_sample images
    for img_path in tqdm(seg_test_files):
        try:
            image_pil, pred_mask_pil, seg_pil = segment_PIL(img_path, model)
        except Exception as e: 
            print(e)
            print(f"failed: {img_path}")

        concatinated_trimaps = concatenate_images_horizontally(image_pil, pred_mask_pil, seg_pil)
        
        if show:
            concatinated_trimaps.show()
        
        # Save the segmented image to the "results" directory
        if save:
            file_name = img_path.split('/')[-1].replace('.jpg', '.png')
            save_image(pred_mask_pil, save_dir['pred_mask'], file_name)            