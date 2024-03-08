"""
Example:

python anchoring.py transparent_image.png glass_image.png --save_dir ./infer_and_postprocess --max_width 1500 --max_height 1000 --center_x 700 --center_y 500 --crop_tight

Example2:
python anchoring.py ./infer_and_postprocess/transparent.png ./infer_and_postprocess/glass_image.png --crop_tight

"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image

import cv2
import glob
import time

def resizeImg(imgTrimmed, maxWidth, maxHeight):
    height, width, _ = imgTrimmed.shape

    if maxWidth > 0:
        scaleX = maxWidth / width
        if scaleX > 1:
            imgTrimmed = cv2.resize(imgTrimmed, None, fx=scaleX, fy=scaleX, interpolation=cv2.INTER_CUBIC)
            # shadowTrimmed = cv2.resize(shadowTrimmed, None, fx=scaleX, fy=scaleX, interpolation=cv2.INTER_CUBIC)
        else:
            imgTrimmed = Image.fromarray(np.uint8(imgTrimmed))
            reHeight = int(height * scaleX)
            imgTrimmed_lan = imgTrimmed.resize((maxWidth, reHeight), Image.LANCZOS)
            imgTrimmed = np.array(imgTrimmed_lan)

    height, width, _ = imgTrimmed.shape

    if (height > maxHeight and maxHeight != 0) or (maxHeight != 0 and maxWidth == 0):
        scaleY = maxHeight / height
        imgTrimmed = Image.fromarray(np.uint8(imgTrimmed))
        reWidth = int(width * scaleY)
        imgTrimmed_lan = imgTrimmed.resize((reWidth, maxHeight), Image.LANCZOS)
        imgTrimmed = np.array(imgTrimmed_lan)

    return imgTrimmed

def trim_space(transparentFilePath, maxWidth, maxHeight):
    img = cv2.imread(transparentFilePath, -1)
    mask = img[:, :, 3]

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    height, width = mask.shape

    if len(contours) > 0:
        contour = sorted(contours, reverse=True, key=cv2.contourArea)[0]

        (xLeft, yUp, w, h) = cv2.boundingRect(contour)

        xRight = xLeft + w
        yDown = yUp + h

        xBuff = int(w * 0.05)
        yBuff = int(h * 0.05)

        xLeft = xLeft - int(xBuff / 2)
        xRight = xRight + int(xBuff / 2)

        yUp = yUp - int(yBuff / 2)
        yDown = yDown + int(yBuff / 2)

        if xLeft < 0:
            xLeft = 0
        if xRight > width:
            xRight = width
        if yUp < 0:
            yUp = 0
        if yDown > height:
            yDown = height
    else:
        yUp = 0
        yDown = height
        xLeft = 0
        xRight = width

    imgTrimmed = img[yUp:yDown, xLeft:xRight, :]

    if maxWidth != 0 or maxHeight != 0:
        imgTrimmed = resizeImg(imgTrimmed, maxWidth, maxHeight)

    return imgTrimmed



def compose_images_with_overlay(bg_img_src, overlay_img_src, output_file, max_width, max_height, center_x=None, center_y=None):
    """
    Compose a background image with a transparent overlay.

    Parameters:
    - bg_img_src (str): Path to the background image.
    - overlay_img_src (str): Path to the transparent overlay image (PNG).
    - output_file (str): Path to the output file (JPEG).
    - max_width (int): Maximum width for the output image.
    - max_height (int): Maximum height for the output image.
    - center_x (int): X-coordinate for the center of the overlay. If None, the overlay will be centered.
    - center_y (int): Y-coordinate for the center of the overlay. If None, the overlay will be centered.

    Returns:
    None
    """
    try:
        start1 = int(time.time() * 1000)

        bg_img = Image.open(bg_img_src)
        overlay_img = Image.open(overlay_img_src).convert("RGBA")

        # Resize overlay while maintaining aspect ratio
        overlay_width, overlay_height = overlay_img.size
        aspect_ratio = overlay_width / overlay_height
        print(f"Aspect Ratio: {aspect_ratio}")
        
        new_overlay_width = min(overlay_width, max_width)
        new_overlay_height = int(new_overlay_width / aspect_ratio)
        
        if new_overlay_height > max_height:
            new_overlay_height = max_height
            new_overlay_width = int(new_overlay_height * aspect_ratio)
            print(f"new_overlay_width, new_overlay_height: {new_overlay_width, new_overlay_height}")
        
        overlay_img = overlay_img.resize((new_overlay_width, new_overlay_height), Image.Resampling.LANCZOS)
        
        bg_width, bg_height = bg_img.size
        print(f"overlay_img size: {overlay_img.size}")

        # Set default center coordinates if not provided
        if center_x is None:
            center_x = bg_width // 2
        if center_y is None:
            center_y = bg_height // 2

        overlay_x = center_x - (new_overlay_width // 2)
        overlay_y = center_y - (new_overlay_height // 2)

        start = int(time.time() * 1000)

        # Create a new image with a white background
        canvas = Image.new('RGBA', (bg_width, bg_height), (255, 255, 255, 0))
        canvas.paste(bg_img, (0, 0, bg_width, bg_height))
        canvas.paste(overlay_img, (overlay_x, overlay_y), mask=overlay_img)

        # Convert the image back to RGB before saving
        canvas = canvas.convert("RGB")

        canvas.save(output_file, format='JPEG')

        end = int(time.time() * 1000)
        print(f"Execution time: {end - start} ms")
        return canvas

    except Exception as e:
        raise e

def process_overlay_images(transparent_image_path, glass_image_path, save_dir='./infer_and_postprocess', max_width=1450, max_height=1025, center_x=800, center_y=640, crop_tight=True, background_image_path="/home/shravan/documents/deeplearning/datasets/segmentations_samples/backgrounds/Subrata/Background_38.png"):
    merged_image_path = os.path.join(save_dir, 'merged_image.png')
    cropped_image_path = os.path.join(save_dir, 'tight_cropped_image.png')
    output_image_path = os.path.join(save_dir, 'fg_overlay_on_bg_image.png')

    image_alpha = cv2.imread(transparent_image_path, cv2.IMREAD_UNCHANGED)
    glass_image = cv2.imread(glass_image_path, cv2.IMREAD_UNCHANGED)

    image_alpha = Image.fromarray(image_alpha)
    glass_image = Image.fromarray(glass_image)

    merged_image = Image.alpha_composite(glass_image, image_alpha)
    cv2.imwrite(merged_image_path, np.array(merged_image))

    if crop_tight:
        trimmed_image = trim_space(merged_image_path, max_width, max_height)
        cv2.imwrite(cropped_image_path, np.array(trimmed_image)) 
        fg_on_bg = compose_images_with_overlay(background_image_path, cropped_image_path, output_image_path, max_width, max_height, center_x, center_y)
    else:
        fg_on_bg = compose_images_with_overlay(background_image_path, merged_image_path, output_image_path, max_width, max_height, center_x, center_y)

    return fg_on_bg

def main():
    parser = argparse.ArgumentParser(description='Process overlay images')
    parser.add_argument('transparent_image_path', type=str, help='Path to the transparent image')
    parser.add_argument('glass_image_path', type=str, help='Path to the glass image')
    parser.add_argument('--save_dir', type=str, default='./infer_and_postprocess', help='Directory to save the processed images')
    parser.add_argument('--max_width', type=int, default=1450, help='Maximum width for the processed images')
    parser.add_argument('--max_height', type=int, default=1025, help='Maximum height for the processed images')
    parser.add_argument('--center_x', type=int, default=800, help='X coordinate for centering the images')
    parser.add_argument('--center_y', type=int, default=640, help='Y coordinate for centering the images')
    parser.add_argument('--crop_tight', action='store_true', help='Whether to crop the images tightly')
    parser.add_argument('--background_image_path', type=str, default="/home/shravan/documents/deeplearning/datasets/segmentations_samples/backgrounds/Subrata/Background_38.png", help='Path to the background image')

    args = parser.parse_args()

    process_overlay_images(args.transparent_image_path, args.glass_image_path, args.save_dir, args.max_width, args.max_height, args.center_x, args.center_y, args.crop_tight, args.background_image_path)

if __name__ == "__main__":
    main()
