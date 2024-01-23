# train.py
import os
from PIL import Image
from datetime import datetime
import gc
from pprint import pprint
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from dataset.trimap_dataset import D4SegmentationTrimapDataset #, TrimapD4SegDataset
from dataset.data_augmentations import get_training_augmentation, get_validation_augmentation
from models.callbacks import VisualizationCallback
from models.model_v2 import SegFormerLightning
from losses.trimap_losses import AlphaPredictionLoss
from utils.segmentation_utils import get_predicted_mask

from utils.utils import concatenate_images, get_checkpoint_dir_name, SaveConcatImageCallback
from configs import get_config
import wandb



# # Main configuration
# config = {
#     'root_dir': "/home/shravan/documents/deeplearning/datasets/D4SegDataset",
#     'resize_height': 512,
#     'resize_width': 512,
#     'in_channels': 4,
#     'out_classes': 1,
#     'batch_size': 1,
#     'num_workers': 8,
#     'encoder_name': "mit_b5",
#     'loss_fns': ['AlphaLoss', 'FocalLoss'],
#     'loss_weights': [0.7, 0.3],
# }

def main(config):
    print(f"Model training ...")
    ## Data Loaders
    root_dir = config['root_dir']
    resize_height = config['resize_height']
    resize_width = config['resize_width']

    # Create dataset instances
    train_transforms = get_training_augmentation(config)
    valid_transforms = get_validation_augmentation(config)

    train_dataset = D4SegmentationTrimapDataset(config, mode='train', augmentation=train_transforms)
    valid_dataset = D4SegmentationTrimapDataset(config, mode='valid', augmentation=valid_transforms)

    # It is a good practice to check datasets don`t intersects with each other
    assert set(valid_dataset.images).isdisjoint(set(train_dataset.images))
    # assert set(test_dataset.images).isdisjoint(set(valid_dataset.images))
    assert set(train_dataset.images).isdisjoint(set(valid_dataset.images))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    # print(f"Test size: {len(test_dataset)}")    
    
    batch_size = config['batch_size']
    n_cpu = config['num_workers']  # os.cpu_count()

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  num_workers=n_cpu)
    
    # model = SegFormerLightning(config)
    model = SegFormerLightning(config)


    # Callback for visualization
    visualization_callback = VisualizationCallback(model, get_predicted_mask, valid_dataloader)
    
    #########
    # Logger
    #########

    wandb.init(project=config['project_name'], group="DDP")
    wandb_logger = pl.loggers.WandbLogger()
    
    base_checkpoints_dir = config['checkpoints_dir']
    
    print(f"Checkpoint directory: {base_checkpoints_dir}")

    if config['resume_checkpoint']:
        ckpt_dir_path = os.path.dirname(config['resume_checkpoint'])
        experiment_name = get_checkpoint_dir_name()
        checkpoints_dir = os.path.join(f"{base_checkpoints_dir}/{experiment_name}")
        print(f"Resuming from checkpoint. Using checkpoints_dir: {ckpt_dir_path}")
    else:
        experiment_name = get_checkpoint_dir_name()
        checkpoints_dir = os.path.join(f"{base_checkpoints_dir}/{experiment_name}")
        print(f"Creating a new training session. Using checkpoints_dir: {checkpoints_dir}")

    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    ## Callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_loss",  # Monitor the validation loss (same as the EarlyStopping)
        mode='min',
        verbose=True,
        # filename='best_model',
        filename="best_model_{epoch:03d}_{val_loss:.2f}",
        dirpath=checkpoints_dir,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # Monitor the validation loss (same as the LR scheduler)
        min_delta=1e-4,
        patience=50,
        verbose=True,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')  
    
    project_name = config['project_name']
    save_dir = f"results/"
    
    # Create an instance of the SaveConcatImageCallback
    save_concat_image_callback = SaveConcatImageCallback(save_dir, project_name)

    
    ## Trainer setup
    trainer = pl.Trainer(
        gpus=config['num_gpus'],
        strategy=config['strategy'],
        check_val_every_n_epoch=1,
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            save_concat_image_callback,
        ],
        fast_dev_run=config['fast_dev_run'] if 'fast_dev_run' in config else False,
        resume_from_checkpoint=config['resume_checkpoint'] if config['resume_checkpoint'] else None,
    )
    
    ## Training
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    
    ## Validation
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

    ckpts = os.listdir(checkpoints_dir)
    print(f"Model saved at {checkpoints_dir}: {ckpts}")
    print(f"Model training finished")

if __name__ == "__main__":
    config = get_config()
    print(config)
    main(config)
    
    
# python train.py --max_epochs 10 --project_name trimap_d4seg_v1 --loss_fns ['AlphaLoss','FocalLoss'] --loss_weights "[0.7, 0.3]" --batch_size 1 --resize_height 512 --resize_width 512

#python train.py --max_epochs 100 --project_name trimap_d4seg_v1 --loss_fns ['AlphaLoss','DiceLoss','FocalLoss'] --loss_weights "[0.3, 0.4, 0.3]" --batch_size 1 --resize_height 512 --resize_width 512

# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 10 --project_name trimap_d4seg_v1 --loss_fns "['AlphaLoss','DiceLoss','FocalLoss']" --loss_weights "[0.3, 0.4, 0.3]" --batch_size 1 --resize_height 512 --resize_width 512 > logs/run_log_20231001.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 10 --project_name trimap_d4seg_v1 --loss_fns "['AlphaLoss','DiceLoss','FocalLoss']" --loss_weights "[0.3, 0.4, 0.3]" --batch_size 1 --resize_height 512 --resize_width 512 --fast_dev_run > logs/run_log_20231004.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 50 --project_name trimap_d4seg_v2 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 224 --resize_width 224 --fast_dev_run > logs/run_log_20231009.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 50 --project_name trimap_d4seg_v2 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 224 --resize_width 224 > logs/run_log_20231009_50epochs.out 2>&1 &

#nohup torchrun --nnodes 1 --nproc_per_node 2 --node_rank 0 train.py --num_gpus 2 --max_epochs 200 --project_name trimap_d4seg_v3 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 2 --resize_height 1024 --resize_width 1024 > logs/run_log_20231014_v3.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 70 --project_name trimap_d4seg_v4 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 224 --resize_width 224 --fast_dev_run > logs/run_log_20231026.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 70 --project_name trimap_d4seg_v4 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 224 --resize_width 224 > logs/run_log_20231026.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 60 --project_name trimap_d4seg_v5 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 224 --resize_width 224 --fast_dev_run > logs/run_log_20231030.out 2>&1 &

# loss function changed to loss = loss.sum() from loss.mean()
# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 60 --project_name trimap_d4seg_v5 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 224 --resize_width 224 > logs/run_log_20231030.out 2>&1 &




# continue to train for another 140 epoch,
# with scheduler updates: 
#     lr - current scheduler setup,
#     then for every 20epochs reduce by 0.5 * lr 
#     

#max_apochs 140
# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 61 --project_name trimap_d4seg_v6 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 224 --resize_width 224 --resume_checkpoint /home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints/20231031/model_20231031_163543/last.ckpt  > logs/run_log_20231106.out 2>&1 &


# /home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints/20231106/model_20231106_182106/

# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 100 --project_name trimap_d4seg_v6 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 224 --resize_width 224 --resume_checkpoint /home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints/20231106/model_20231106_182106/last.ckpt  > logs/run_log_20231107.out 2>&1 &


#nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 50 --project_name trimap_d4seg_v7 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 224 --resize_width 224 --fast_dev_run > logs/run_log_20231116.out 2>&1 &


# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 50 --project_name trimap_d4seg_v7 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 224 --resize_width 224 --resume_checkpoint /home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints/20231118/model_20231118_155542/last.ckpt > logs/run_log_20231120.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 train.py --arch Unet --max_epochs 50 --project_name trimap_d4seg_v8 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 256 --resize_width 256 --fast_dev_run > logs/run_log_20231123.out 2>&1 &


# /home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints/20231123/model_20231123_150736/


# nohup torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 train.py --arch Unet --max_epochs 100 --project_name trimap_d4seg_v8 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 256 --resize_width 256 --resume_checkpoint /home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints/20231123/model_20231123_150736/last.ckpt > logs/run_log_20231125.out 2>&1 &


# nohup torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 train.py --arch Unet --max_epochs 100 --project_name trimap_d4seg_v9 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 512 --resize_width 512 --fast_dev_run > logs/run_log_20231130.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 train.py --arch Unet --max_epochs 100 --project_name trimap_d4seg_v9 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 512 --resize_width 512 > logs/run_log_20231130.out 2>&1 &


# nohup torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 train.py --arch Unet --max_epochs 200 --project_name trimap_d4seg_v9 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 512 --resize_width 512 --resume_checkpoint /home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints/20231130/model_20231130_131351/last.ckpt> logs/run_log_20231203.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 train.py --arch Unet --max_epochs 100 --project_name premask_d4seg_v11 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 512 --resize_width 512 --fast_dev_run > logs/run_log_20231213.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 train.py --arch Unet --max_epochs 100 --project_name premask_d4seg_v11 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 512 --resize_width 512 > logs/run_log_20231213.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 train.py --arch Unet_Plain --max_epochs 100 --project_name premask_d4seg_v11 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 1024 --resize_width 1024 --fast_dev_run > logs/run_log_20231218.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 train.py --arch Unet_Plain --max_epochs 100 --project_name premask_d4seg_v11 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 1024 --resize_width 1024 > logs/run_log_20231218.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 train.py --arch Unet_Plain --max_epochs 200 --project_name predmask_d4seg_v13 --loss_fns "['AlphaLoss']" --loss_weights "[1.0]" --batch_size 8 --resize_height 1024 --resize_width 1024 --resume_checkpoint /home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints/20231221/model_20231221_041709/last.ckpt> logs/run_log_20240102.out 2>&1 &