import argparse
import ast
import json

def get_config():
    parser = argparse.ArgumentParser(description="D4 Segmentation Configurations")

    # Trainer args
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--strategy', type=str, default='ddp', help='Training strategy')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--fast_dev_run', action='store_true', help='Whether to use fast_dev_run')

    # Model specific arguments
    #arch : Unet | Unet_Plain 
    parser.add_argument('--arch', type=str, default='Unet', help='Model architecture')
    parser.add_argument('--encoder_name', type=str, default='mit_b5', help='Encoder name for the model')
    parser.add_argument('--in_channels', type=int, default=4, help='Number of input channels')
    parser.add_argument('--out_classes', type=int, default=1, help='Number of output classes')
    
    # Data module related arguments    
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
        
    parser.add_argument('--root_dir', type=str, default='/home/shravan/documents/deeplearning/datasets/D4SegDataset', help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    
    # Add arguments for image resizing
    parser.add_argument('--resize_height', type=int, default=640, help='Height for image resizing')
    parser.add_argument('--resize_width', type=int, default=640, help='Width for image resizing')
        
    # Program arguments
    parser.add_argument('--checkpoints_dir', type=str, default='/home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume_checkpoint', type=str, default='', help='Path to the checkpoint to resume from')
    parser.add_argument('--project_name', type=str, default='trimap_d4_segmentation_test', help='Project name')
    
    parser.add_argument('--loss_fns', type=ast.literal_eval, default=[])
    # parser.add_argument('--loss_fns', nargs='+', default=[])
    parser.add_argument('--loss_weights', type=json.loads, default=[])
    parser.add_argument('--lr', type=float, default=1e-5)

    args = parser.parse_args()
    return vars(args)



# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 10 --project_name trimap_d4seg_v1 --loss_fns "['AlphaLoss','FocalLoss']" --loss_weights "[0.7, 0.3]" --batch_size 1 --resize_height 512 --resize_width 512 > logs/run_log_20230929.out 2>&1 &

# nohup torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 train.py --max_epochs 50 --project_name trimap_d4seg_v1 --loss_fns "['AlphaLoss','DiceLoss','FocalLoss']" --loss_weights "[0.3,0.4,0.3]" --batch_size 1 --resize_height 512 --resize_width 512 > logs/run_log_20231003.out 2>&1 &