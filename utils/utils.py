import os
from PIL import Image
from datetime import datetime
import pytorch_lightning as pl
import matplotlib.pyplot as plt


def get_checkpoint_dir_name():
    now = datetime.now()
    _today = now.strftime("%Y%m%d")
    formatted_date_time = now.strftime("%Y%m%d_%H%M%S")
    return f"{_today}/model_{formatted_date_time}/"


class SaveConcatImageCallback(pl.Callback):
    def __init__(self, save_dir, project_name):
        self.save_dir = os.path.join(save_dir, project_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # valid_dataloader = trainer.valid_dataloaders[0]  # Assuming there's only one validation dataloader
        
        valid_dataloader = trainer.val_dataloaders[0]

        # Get the concatenated image
        image_pil, mask_pil, segmented_image_pil = pl_module.eval_model_and_log(valid_dataloader)
        concatenated_image = concatenate_images(image_pil, mask_pil, segmented_image_pil)

        # Save the concatenated image to the results directory
        epoch = trainer.current_epoch
        image_path = os.path.join(self.save_dir, f"epoch_{epoch}_concatenated_image.png")
        concatenated_image.save(image_path)

        # print(f"Concatenated image saved for epoch {epoch}.")


        

def concatenate_images(image_pil, mask_pil, segmented_image_pil):
    # Resize the segmented image to the size of the input image
    segmented_image_pil = segmented_image_pil.resize(image_pil.size, Image.Resampling.BILINEAR)
    
    # Get the dimensions of the input images
    width1, height1 = image_pil.size
    width2, height2 = mask_pil.size
    width3, height3 = segmented_image_pil.size

    # Find the maximum height among the three images
    max_height = max(height1, height2, height3)

    # Calculate the total width required to accommodate all images side by side
    total_width = width1 + width2 + width3

    # Create a new blank image with the calculated dimensions
    concatenated_image = Image.new('RGBA', (total_width, max_height))

    # Paste the three images side by side on the new blank image
    concatenated_image.paste(image_pil, (0, 0))
    concatenated_image.paste(mask_pil, (width1, 0))
    concatenated_image.paste(segmented_image_pil, (width1 + width2, 0))
    
    return concatenated_image

def preprocess(im_path, _H=1024, _W=1024):
    image = Image.open(im_path)
    image = np.array(image)
    # resize images
    image = np.array(Image.fromarray(image).resize((_H,_W), Image.Resampling.BILINEAR))

    # convert to other format HWC -> CHW
    image = np.moveaxis(image, -1, 0)
    image = torch.from_numpy(image)
    return image

def get_predicted_mask(model, image):
    with torch.no_grad():
        model.eval()
        logits = model(image)
    prob_mask = logits.sigmoid()
    return prob_mask

def segment_PIL(img_path, model):
    
    image_tensor = preprocess(img_path)
    mask_tensor = get_predicted_mask(model, image_tensor)
    
    image_array = image_tensor.numpy().transpose(1, 2, 0)
    mask_array = mask_tensor.numpy().squeeze()
    mask_array = np.expand_dims(mask_array, axis=-1)  
    
    seg_img = image_array*mask_array
    
    image_pil = Image.fromarray(np.uint8(image_array))  # Assuming image range is [0, 1] and converting to [0, 255]
    mask_pil = Image.fromarray(np.uint8(mask_array.squeeze() * 255))    # Assuming mask range is [0, 1] and converting to [0, 255]

    seg_img_pil = Image.fromarray(np.uint8(seg_img))
    
    return image_pil, mask_pil, seg_img_pil

# helper function for data visualization
def visualize_image_mask(sample):
    plt.subplot(1,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(sample["image"].numpy().transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
    plt.subplot(1,2,2)
    # Turn off tick labels
    plt.xticks([])
    plt.yticks([])
    plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
    plt.show()