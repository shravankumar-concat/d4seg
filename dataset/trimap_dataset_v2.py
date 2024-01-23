import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import cv2
from PIL import Image
import albumentations as albu
import albumentations.pytorch as albu_pt
import matplotlib.pyplot as plt
from utils.segmentation_utils import preprocess, concat_images
from plotly import express as px

from scipy.ndimage import distance_transform_edt

def generate_trimap(alpha):
    # alpha \in [0, 1] should be taken into account
    # be careful when dealing with regions of alpha=0 and alpha=1
    fg = np.array(np.greater_equal(alpha, 254).astype(np.float32))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))  # unknown = alpha > 0
    unknown = unknown - fg
    # image dilation implemented by Euclidean distance transform
    unknown = distance_transform_edt(unknown == 0) <= 3  # np.random.randint(1, 5)  # reduce it to 2
    trimap = fg * 255
    trimap[unknown] = 128
    return trimap.astype(np.uint8)

class D4SegmentationTrimapDataset(Dataset):
    
    def __init__(self, config, mode, augmentation=None):
        assert mode in {"train", "valid", "test"}
        self.config = config
        self.root_dir = config['root_dir']
        self.resize_height = config['resize_height']
        self.resize_width = config['resize_width']
        self.mode = mode
        self.augmentation = augmentation
        self.get_filenames()
    
    def get_filenames(self):
        images_dir = os.path.join(self.root_dir, 'images')
        annotations_dir = os.path.join(self.root_dir, 'matte_annotations')
        pred_masks_dir = os.path.join(self.root_dir, 'pred_masks')

        self.images_training_dir = os.path.join(images_dir, 'training')
        self.annotations_training_dir = os.path.join(annotations_dir, 'training')
        self.pred_masks_dir_training_dir = os.path.join(pred_masks_dir, 'training')
        
        self.images_validation_dir = os.path.join(images_dir, 'validation')
        self.annotations_validation_dir = os.path.join(annotations_dir, 'validation')
        self.pred_masks_dir_validation_dir = os.path.join(pred_masks_dir, 'validation')
        
        if self.mode == 'train':
            self.images, self.masks, self.pred_masks = self.get_file_lists(self.images_training_dir, self.annotations_training_dir, self.pred_masks_dir_training_dir)
        else:
            self.images, self.masks, self.pred_masks = self.get_file_lists(self.images_validation_dir, self.annotations_validation_dir, self.pred_masks_dir_validation_dir)
            
    def get_file_lists(self, images_dir, annotations_dir, pred_masks_dir):
        images = sorted([i for i in os.listdir(images_dir) if i.endswith('.jpg')])
        masks = sorted([i for i in os.listdir(annotations_dir) if i.endswith('.png')])
        pred_masks = sorted([i for i in os.listdir(pred_masks_dir) if i.endswith('.png')])
        return images, masks, pred_masks
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx].split('.')[0]
        
        self.original_image = os.path.join(self.images_training_dir if self.mode == 'train' else self.images_validation_dir, f"{image_name}.jpg")
        self.mask_image = os.path.join(self.annotations_training_dir if self.mode == 'train' else self.annotations_validation_dir, f"{image_name}.png")
        self.pred_mask_image = os.path.join(self.pred_masks_dir_training_dir if self.mode == 'train' else self.pred_masks_dir_validation_dir, f"{image_name}.png")

        self.image = cv2.imread(self.original_image)
        self.mask = cv2.imread(self.mask_image, cv2.IMREAD_GRAYSCALE)
        self.pred_mask = cv2.imread(self.pred_mask_image, cv2.IMREAD_GRAYSCALE)
                
        
        #trimap generation using image+pred_mask        
        self.pred_mask = cv2.resize(self.pred_mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_AREA)
        self.trimap = generate_trimap(self.pred_mask)  
        
        self.mask = np.expand_dims(self.mask, axis=-1)
        self.trimap = np.expand_dims(self.trimap, axis=-1)
        
        pre_sample = {
            'image': self.image, 
            'trimap': self.trimap,
            'mask': self.mask,
        }

        # Apply augmentations
        if self.augmentation:
            pre_sample = self.augmentation(pre_sample)
                        
        sample = dict() 
        sample.update(pre_sample)

        sample["image_trimap"] = concat_images(sample["image"], sample["trimap"])
        
        sample["image"] = sample["image_trimap"]
        
        # sample["image_mask"] = concat_images(sample["image"], sample["mask"])

        sample["image_filepath"] = self.original_image
        sample["mask_filepath"]= self.mask_image
        
        selected_keys = ['image', 'mask', 'image_filepath', 'mask_filepath']
        selected_dict = {key: sample[key] for key in selected_keys}

        return selected_dict
