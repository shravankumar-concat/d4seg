
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
Model 1: [model1.ckpt](checkpoints/model_1_20230823_133015_last.ckpt)
Model 2: [model2.ckpt](checkpoints/model_2_20240102_155705_last.ckpt)
### Output
The script generates two output images:

    image_alpha.png: Image alpha segmentation result.
    glass_image.png: Glass image result.

## License
This project is licensed under the [CONCAT License Name] - see the LICENSE.md file for details.

Acknowledgments
