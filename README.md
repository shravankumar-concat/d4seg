
# D4 Segmentation - Inference

This script performs image processing and segmentation using two pre-trained deep learning models. It loads the models, processes an input image, and produces segmented outputs.

## Usage

### Prerequisites

- Python 3.x
- Install required Python packages: spec-file.txt / environment.yml

Ref: 
1. [Create Identical Conda Environment using spec-file.txt](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments)

2. [Creating Environment using environment.yml](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

### Run the Script
```python
python infer_and_postprocess.py input_image_path [--model1_ckpt MODEL1_CHECKPOINT] [--model2_ckpt MODEL2_CHECKPOINT] [--image_alpha_path IMAGE_ALPHA_OUTPUT] [--glass_image_path GLASS_IMAGE_OUTPUT] [--no_cuda]
```

- input_image_path: Path to the input image.
- model1_ckpt: Path to the checkpoint file for Model 1 (default: "model1_checkpoint.pth").
- model2_ckpt: Path to the checkpoint file for Model 2 (default: "model2_checkpoint.pth").
- image_alpha_path: Path to save the image alpha output (default: "image_alpha.png").
- glass_image_path: Path to save the glass image output (default: "glass_image.png").
- no_cuda: Flag to disable CUDA (use CPU).

### Example: 

```python
python infer_and_postprocess.py path/to/input_image.jpg --model1_ckpt path/to/model1_checkpoint.pth --model2_ckpt path/to/model2_checkpoint.pth --image_alpha_path output/image_alpha.png --glass_image_path output/glass_image.png --no_cuda
```

## Models
Model Naming Cinvention:
```
arch
resolution
epoch
metrics
part_number
MMYY
```
Model 1: 

Model Run Details: 
https://wandb.ai/shravanp-ai/d4seg_mitb5_adam_loss_DF_1024_v4/runs/lc3wri2c/overview?workspace=user-shravanp-ai

```
arch : Unet - MiT-B5
resolution: 1024
epoch: 200
metrics: 0.9928 [Validation Dice Coeff]
part_number: 01 [Model1]
MMYY: 0823
```

- Recommended Name: unet_mitb5_1024_200_0.9928_01_0823.ckpt

[OneDrive - Model1](https://nuncsystems-my.sharepoint.com/:f:/p/shravan_p/EquCz6-QASpJkuLKLyEurGYB0_zlSBogy91uEGC4DS6prA?e=u5ldeo)

Place it in here: 

    Ex: checkpoints/unet_mitb5_1024_200_0.9928_01_0823.ckpt

Model 2:

Model Run Details: 
https://wandb.ai/shravanp-ai/predmask_d4seg_v13/runs/rnu2r6tr/overview?workspace=user-shravanp-ai

```
arch : Unet - Plain
resolution: 1024
epoch: 195
metrics: 0.0002 [MSE]
part_number: 02 [Model2]
MMYY: 0124
```

- Recommended Name: unet_plain_1024_195_0.0002_02_0124.ckpt


[OneDrive - Model2](https://nuncsystems-my.sharepoint.com/:f:/p/shravan_p/EqXHbL4_UABOs3V7JdwqRksBfsUUpISL0jgACLKxFOyzyg?e=MT9vUE)

Place it in here: 

    Ex: checkpoints/unet_plain_1024_195_0.0002_02_0124.ckpt
    
Or download from GDRIVE: 

[Google Drive](https://drive.google.com/file/d/14P6nz2qHNxt3LnnomAlCsSgDXF1XSatl/view?usp=drive_link)

### Output
The script generates two output images:

    image_alpha.png: Image alpha segmentation result.
    glass_image.png: Glass image result.

## License
This project is licensed under the [CONCAT License Name] - see the LICENSE.md file for details.

Acknowledgments
