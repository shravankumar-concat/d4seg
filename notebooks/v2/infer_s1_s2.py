import sys
sys.path.append("/home/shravan/documents/deeplearning/github/alpha_matte_segmentation/alpha_matte_segmentation")

import os
from PIL import Image
from datetime import datetime
import gc
from pprint import pprint
import random
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from dataset.trimap_dataset import D4SegmentationTrimapDataset#TrimapD4SegDataset
from dataset.data_augmentations import get_training_augmentation, get_validation_augmentation
from models.callbacks import VisualizationCallback
from models.model_v2 import SegFormerLightning
from losses.trimap_losses import AlphaPredictionLoss
from utils.segmentation_utils import get_predicted_mask

from utils.utils import concatenate_images, get_checkpoint_dir_name, SaveConcatImageCallback

from utils.segmentation_utils import get_checkpoint, get_model, generate_trimap, get_trimap_array, get_predicted_mask, save_image
from utils.segmentation_utils import preprocess, segment_PIL, make_background_transparent, add_bg

from configs import get_config
import wandb

from plotly import express as px

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


def get_stage2_pred(input_image, model1, model2, kernel_size=3):
    # Extracting the filename and extension

    # stage-1
    image_pil, mask_pil, seg_pil = segment_PIL(input_image, model1)
    segmented_image_s1_pil = make_background_transparent(image_pil, mask_pil)
    
    # ground truth 
    filename, extension = os.path.splitext(os.path.basename(input_image))
    gt_alpha_path = f"{model1.config['root_dir']}/matte_annotations/validation/{filename}.png"
    # print(gt_alpha_path, os.path.isfile(gt_alpha_path))
    if os.path.isfile(gt_alpha_path):
        gt_alpha_pil = Image.open(gt_alpha_path).convert('L')
        # print((image_pil.size, gt_alpha_pil.size))
        gt_segmented_image_pil = make_background_transparent(image_pil,gt_alpha_pil)
    
    #stage-2
    orig_image, trimap, image_trimap = get_trimap_array(image_pil, mask_pil, kernel_size=kernel_size)
    x = torch.tensor(image_trimap, dtype=torch.float32)
    x = rearrange(x,'w h c -> c w h')
    x = x.to(device)

    with torch.no_grad():
        logits = model(x.unsqueeze(0))
    # prob_mask = logits.sigmoid()
    
    prob_mask = logits.cpu().numpy().squeeze()*255
    
    orig_image_pil = Image.fromarray(orig_image)
    trimap_pil = Image.fromarray(trimap)

    aplha_pil = Image.fromarray(logits.cpu().numpy().squeeze()*255).convert('L')
    segmented_image_s2_pil = make_background_transparent(image_pil, aplha_pil)
    
    if os.path.isfile(gt_alpha_path):
        return image_pil, trimap_pil, segmented_image_s1_pil, segmented_image_s2_pil, gt_segmented_image_pil
    else:
        return image_pil, trimap_pil, segmented_image_s1_pil, segmented_image_s2_pil

def save_results(input_image_path, model1, model2, kernel_size=3, save_dir='s1_s2_results'):
    out = get_stage2_pred(input_image=input_image_path, model1=model1, model2=model, kernel_size=kernel_size)
    concatinated_out = concatenate_images_horizontally(out)
    
    # Extracting the filename and extension
    filename, extension = os.path.splitext(os.path.basename(input_image_path))
    # Replacing the extension with 'png'
    save_filename = f'{filename}.png'

    save_image(image_pil=concatinated_out, save_dir=save_dir, file_name=save_filename)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# stage-I
# /home/shravan/documents/deeplearning/github/segmentation_models/checkpoints/20230823/model_20230823_133015/
mit_b5_latest_path_v6 = "20230823/model_20230823_133015/"

ckpt1 = get_checkpoint(mit_b5_latest_path_v6, 'last')
# ckpt2 = get_checkpoint(mit_b5_latest_path_v6, 'best')
model1 = get_model(ckpt1)
_H, _W= model1.config['resize_height'], model1.config['resize_width']
model1 = model1.to(device)
# model.device

# stage-II
# base_ckpt_dir = "/home/shravan/documents/deeplearning/github/segmentation_models/checkpoints/"
# https://wandb.ai/shravanp-ai/trimap_d4seg_v3/runs/awd8x9ak/logs?workspace=user-shravanp-ai
# mit_b5_latest_path_s2 =  "20231014/model_20231014_053908"  #75epochs

# /home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints/20231107/model_20231107_062452/
base_ckpt_dir = "/home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints/"
mit_b5_latest_path_s2 = "20231120/model_20231120_075738/"

print(os.listdir(f"{base_ckpt_dir}/{mit_b5_latest_path_s2}"))

ckpt_path = get_checkpoint(mit_b5_latest_path_s2, 'last', base_ckpt_dir)

pretrained_model = torch.load(ckpt_path)
model_config = pretrained_model['hyper_parameters']['config']

model = SegFormerLightning(model_config)
model.load_state_dict(pretrained_model['state_dict'])
model = model.to(device)

root_dir = f"{model1.config['root_dir']}/images/validation"
seg_test_files = sorted([os.path.join(root_dir,p) for p in os.listdir(root_dir) if p.endswith('.jpg')])

seg_test_files = seg_test_files[:20]

for test_img in tqdm(seg_test_files):
    save_results(test_img, model1=model1, model2=model, kernel_size=1, save_dir='s1_s2_results/model_20231120_075738/s1_s2_k1')
    
for test_img in tqdm(seg_test_files):
    save_results(test_img, model1=model1, model2=model, kernel_size=3, save_dir='s1_s2_results/model_20231120_075738/s1_s2_k3')