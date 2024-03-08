

# D4 Segmentation - Inference

This script performs image processing and segmentation using two pre-trained deep learning models. It loads the models, processes an input image, and produces segmented outputs.

## Usage

### Prerequisites

- Python 3.x
- Install required Python packages: spec-file.txt / environment.yml

Ref: 
1. [Create Identical Conda Environment using spec-file.txt](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments)

2. [Creating Environment using environment.yml](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

# Generate Overlay Images:
[Image Alpha and Glass Image generation](docs/generate_image_glass.md)

# Anchoring : 
[anchoring documentaiton](docs/anchoring.md)

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




## License
This project is licensed under the [CONCAT License Name] - see the LICENSE.md file for details.

Acknowledgments
