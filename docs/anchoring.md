
# Image Overlay Processor

This Python script processes overlay images, merging a transparent image (e.g., glasses) onto a background image

```
python anchoring.py transparent_image.png glass_image.png --save_dir ./infer_and_postprocess --max_width 1500 --max_height 1000 --center_x 700 --center_y 500 --crop_tight
```

Arguments:
```
transparent_image_path: Path to the transparent image (e.g., glasses).
glass_image_path: Path to the glass image.
--save_dir SAVE_DIR (optional): Directory to save the processed images (default: './infer_and_postprocess').
--max_width MAX_WIDTH (optional): Maximum width for the processed images (default: 1450).
--max_height MAX_HEIGHT (optional): Maximum height for the processed images (default: 1025).
--center_x CENTER_X (optional): X coordinate for centering the images (default: 800).
--center_y CENTER_Y (optional): Y coordinate for centering the images (default: 640).
--crop_tight (optional): Whether to crop the images tightly.
--background_image_path BACKGROUND_IMAGE_PATH (optional): Path to the background image (default: '/home/shravan/documents/deeplearning/datasets/segmentations_samples/backgrounds/Subrata/Background_38.png').```