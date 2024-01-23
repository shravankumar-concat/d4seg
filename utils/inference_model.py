import torch
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import os
import plotly.express as px
from einops import rearrange

#v2
import torch
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import os
import plotly.express as px
from einops import rearrange

class CarSegmenter(pl.LightningModule):
    def __init__(self, model, ckpt_path=None):
        super().__init__()
        
        self.model = model
        
        if ckpt_path:
            self.load_model(ckpt_path)

    def load_model(self, ckpt_path):
        new_model = torch.load(ckpt_path)
        self.model.load_state_dict(new_model['state_dict'])
        self.model.eval()

    def preprocess(self, im_path, _H=1024, _W=1024):
        image = Image.open(im_path)
        orig_H, orig_W = image.size
        image = np.array(image)
        image = np.array(Image.fromarray(image).resize((_H, _W), Image.Resampling.BILINEAR))
        image = rearrange(image, 'h w c -> c h w')
        image = torch.from_numpy(image)
        return image, orig_H, orig_W

    def get_predicted_mask(self, image):
        with torch.no_grad():
            logits = self.model(image)
        prob_mask = logits.sigmoid()
        return prob_mask

    def make_background_transparent(self, rgb_image, segmentation_mask, output_path=None):
        rgba_image = rgb_image.convert("RGBA")
        rgba_image.putalpha(segmentation_mask)
        
        if output_path:
            rgba_image.save(output_path, format="PNG")
        
        return rgba_image

    def segment_PIL(self, img_path, _H=1024, _W=1024):
        image_tensor, orig_H, orig_W = self.preprocess(img_path, _H, _W)
        mask_tensor = self.get_predicted_mask(image_tensor)

        image_array = image_tensor.numpy().transpose(1, 2, 0)
        mask_array = mask_tensor.numpy().squeeze()
        mask_array = np.expand_dims(mask_array, axis=-1)

        seg_img = image_array * mask_array

        rgb_image_pil = Image.fromarray((image_array).astype(np.uint8))
        self.mask_pil = Image.fromarray((mask_array.squeeze() * 255).astype(np.uint8)) 

        alpha_mask = Image.fromarray((mask_array.squeeze() * 255).astype(np.uint8))  
        self.rgba_image_pil = self.make_background_transparent(rgb_image_pil, alpha_mask)

        return rgb_image_pil, self.mask_pil, self.rgba_image_pil 

    def add_bg(self, bg_img_path=None):
        if bg_img_path is None:
            bg_img_path = '/home/shravan/documents/deeplearning/datasets/segmentations_samples/backgrounds/Subrata/white_bg_1600x1200.jpg'
        
        bg_img_pil = Image.open(bg_img_path)
        bg_img_pil = bg_img_pil.convert(self.rgba_image_pil.mode).resize(self.rgba_image_pil.size)
        
        self.composite_image = Image.alpha_composite(bg_img_pil,self.rgba_image_pil)
        return self.composite_image

    def concatenate_images_horizontally(self, image_pil, mask_pil, segmented_image_pil, show=False):
        segmented_image_pil = segmented_image_pil.resize((image_pil.size),Image.Resampling.BILINEAR)
        
        width1, height1 = image_pil.size
        width2, height2 = mask_pil.size
        width3, height3 = segmented_image_pil.size

        max_height = max(height1, height2, height3)
        total_width = width1 + width2 + width3

        concatenated_image = Image.new('RGBA', (total_width, max_height))

        concatenated_image.paste(image_pil, (0, 0))
        concatenated_image.paste(mask_pil, (width1, 0))
        concatenated_image.paste(segmented_image_pil, (width1 + width2, 0))

        if show: 
            concatenated_image.show()
        else:
            self.save_image(concatenated_image, './results', 'predicted.png')
        return concatenated_image

    def save_image(self, image_pil, save_dir, file_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file_name)
        image_pil.save(save_path)



# import torch
# import pytorch_lightning as pl
# from PIL import Image
# import numpy as np
# import os
# import plotly.express as px
# from einops import rearrange

# class CarSegmenter(pl.LightningModule):
#     def __init__(self, model, ckpt_path=None):
#         super().__init__()
        
#         self.model = model
        
#         if ckpt_path:
#             self.load_model(ckpt_path)

#     def load_model(self, ckpt_path):
#         new_model = torch.load(ckpt_path)
#         self.model.load_state_dict(new_model['state_dict'])
#         self.model.eval()

#     @staticmethod
#     def preprocess(im_path, _H=1024, _W=1024):
#         image = Image.open(im_path)
#         orig_H, orig_W = image.size
#         image = np.array(image)
#         # resize images
#         image = np.array(Image.fromarray(image).resize((_H, _W), Image.Resampling.BILINEAR))

#         # convert to other format HWC -> CHW
#         # image = np.moveaxis(image, -1, 0)
#         image = rearrange(image, 'h w c -> c h w')
#         image = torch.from_numpy(image)
#         return image, orig_H, orig_W

#     @staticmethod
#     def get_predicted_mask(model, image):
#         with torch.no_grad():
#             logits = model(image)
#         prob_mask = logits.sigmoid()
#         return prob_mask

    
#     def make_background_transparent(self, rgb_image, segmentation_mask, output_path=None):
#         # Convert the RGB image to RGBA and set alpha channel from the segmentation mask
#         rgba_image = rgb_image.convert("RGBA")
#         rgba_image.putalpha(segmentation_mask)

#         # Save the transparent image as PNG if an output path is provided
#         if output_path:
#             rgba_image.save(output_path, format="PNG")

#         return rgba_image
    
#     def segment_PIL(self, img_path):
#         image_tensor, orig_H, orig_W = self.preprocess(img_path)
#         mask_tensor = self.get_predicted_mask(self.model, image_tensor)

#         image_array = image_tensor.numpy().transpose(1, 2, 0)
#         mask_array = mask_tensor.numpy().squeeze()
#         mask_array = np.expand_dims(mask_array, axis=-1)

#         seg_img = image_array * mask_array

#         # Convert image and mask arrays to PIL images
#         rgb_image_pil = Image.fromarray((image_array).astype(np.uint8))
#         # Assuming mask range is [0, 1] and converting to [0, 255]
#         self.mask_pil = Image.fromarray((mask_array.squeeze() * 255).astype(np.uint8)) 

#         # Create RGBA image with transparent background
#         # Convert mask back to [0, 255]
#         alpha_mask = Image.fromarray((mask_array.squeeze() * 255).astype(np.uint8))  
#         self.rgba_image_pil = self.make_background_transparent(rgb_image_pil, alpha_mask)

#         return rgb_image_pil, self.mask_pil, self.rgba_image_pil 
    
#     def add_bg(self, bg_img_path=None):
        
#         if bg_img_path is None:
#             bg_img_path = '/home/shravan/documents/deeplearning/datasets/segmentations_samples/backgrounds/Subrata/img_acp_background_4323.jpg'
        
#         bg_img_pil = Image.open(bg_img_path)
        
#         bg_img_pil = bg_img_pil.convert(self.rgba_image_pil.mode).resize(self.rgba_image_pil.size)

#         self.composite_image = Image.alpha_composite(bg_img_pil,self.rgba_image_pil)
#         return self.composite_image
    

#     # @staticmethod
#     def concatenate_images_horizontally(self, image_pil, mask_pil, segmented_image_pil, show=False):
        
#         segmented_image_pil = segmented_image_pil.resize((image_pil.size),Image.Resampling.BILINEAR)
#         # Get the dimensions of the input images
        
#         width1, height1 = image_pil.size
#         width2, height2 = mask_pil.size
#         width3, height3 = segmented_image_pil.size

#         # Find the maximum height among the three images
#         max_height = max(height1, height2, height3)

#         # Calculate the total width required to accommodate all images side by side
#         total_width = width1 + width2 + width3

#         # Create a new blank image with the calculated dimensions
#         concatenated_image = Image.new('RGBA', (total_width, max_height))

#         # Paste the three images side by side on the new blank image
#         concatenated_image.paste(image_pil, (0, 0))
#         concatenated_image.paste(mask_pil, (width1, 0))
#         concatenated_image.paste(segmented_image_pil, (width1 + width2, 0))

#         # Display the concatenated image
#         if show: 
#             concatenated_image.show()
#         else:
#             self.save_image(concatenated_image, './results', 'predicted.png')
#         return concatenated_image

#     # @staticmethod
#     def save_image(self,image_pil, save_dir, file_name):
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         save_path = os.path.join(save_dir, file_name)
#         image_pil.save(save_path)