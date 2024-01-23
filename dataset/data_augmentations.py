import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import transforms as T

class Resize(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        image, mask, trimap = sample["image"], sample["mask"], sample["trimap"]
        
        # Resize image, mask, and trimap using cv2
        image = cv2.resize(image, self.size[::-1], interpolation=self.interpolation)
        mask = cv2.resize(mask, self.size[::-1], interpolation=self.interpolation)
        trimap = cv2.resize(trimap, self.size[::-1], interpolation=cv2.INTER_NEAREST)

        sample["image"], sample["mask"], sample["trimap"] = image, mask, trimap
        return sample


class RandomHueSaturationValue(object):
    def __init__(self, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255), p=0.5):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.p = p

    def __call__(self, sample):
        image, mask, trimap = sample["image"], sample["mask"], sample["trimap"]

        if np.random.random() < self.p:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            sample["image"] = image

        return sample

class RandomShiftScaleRotate(object):
    def __init__(self, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=(-45, 45),
                 aspect_limit=(0, 0), border_mode=cv2.BORDER_CONSTANT, p=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.aspect_limit = aspect_limit
        self.border_mode = border_mode
        self.p = p

    def __call__(self, sample):
        image, mask, trimap = sample["image"], sample["mask"], sample["trimap"]

        if np.random.random() < self.p:
            height, width, channel = image.shape

            angle = np.random.uniform(self.rotate_limit[0], self.rotate_limit[1])  # degree
            scale = np.random.uniform(1 + self.scale_limit[0], 1 + self.scale_limit[1])
            aspect = np.random.uniform(1 + self.aspect_limit[0], 1 + self.aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * width)
            dy = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=self.border_mode,
                                        borderValue=(0, 0, 0,))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=self.border_mode,
                                       borderValue=(0, 0, 0,))
            trimap = cv2.warpPerspective(trimap, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=self.border_mode,
                                         borderValue=(0, 0, 0,))

            sample["image"], sample["mask"], sample["trimap"] = image, mask, trimap
        return sample

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask, trimap = sample["image"], sample["mask"], sample["trimap"]

        if np.random.random() < self.p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            trimap = cv2.flip(trimap, 1)
            sample["image"], sample["mask"], sample["trimap"] = image, mask, trimap
        return sample

class RandomCrop(object):
    def __init__(self, crop_ratio=0.3, p=0.3):
        self.crop_ratio = crop_ratio
        self.p = p

    def __call__(self, sample):
        image, mask, trimap = sample["image"], sample["mask"], sample["trimap"]

        if np.random.random() < self.p:
            width = int((1 - self.crop_ratio) * image.shape[1])
            height = int((1 - self.crop_ratio) * image.shape[0])
            assert image.shape[0] >= height
            assert image.shape[1] >= width
            assert image.shape[0] == mask.shape[0]
            assert image.shape[1] == mask.shape[1]
            x = np.random.randint(0, image.shape[1] - width)
            y = np.random.randint(0, image.shape[0] - height)
            image = image[y:y + height, x:x + width]
            mask = mask[y:y + height, x:x + width]
            trimap = trimap[y:y + height, x:x + width]
            sample["image"], sample["mask"], sample["trimap"] = image, mask, trimap
        return sample


class ToTensor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, sample):
        image, trimap, mask = sample['image'], sample['trimap'], sample['mask']
        
        image = image.transpose((2, 0, 1)).astype(np.float32) # HWC -> CHW
        trimap = np.expand_dims(trimap.astype(np.float32), axis=0) # HW -> 1HW
        mask = np.expand_dims(mask.astype(np.float32), axis=0) # HW -> 1HW
        
        # numpy array -> torch tensor
        sample['image'], sample['trimap'], sample['mask'] = \
            torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long), torch.from_numpy(mask)
        
        # Normalization
        sample['image'] = sample['image'].float()/255.
        sample['trimap'] = sample['trimap'].float()/255.
        sample['mask'] = sample['mask'].float()/255.

        return sample


# Data augmentations
def get_training_augmentation(config):
    train_transform = [
        Resize(size=(config["resize_height"], config["resize_width"])),
        RandomHueSaturationValue(
            hue_shift_limit=(-50, 50),
            sat_shift_limit=(-50, 50),
            val_shift_limit=(-50, 50)
        ),
        RandomShiftScaleRotate(
            shift_limit=(-0.0625, 0.0625),
            scale_limit=(-0.1, 0.1),
            rotate_limit=(-2, 2)
        ),
        RandomHorizontalFlip(),
        ToTensor(),
    ]
    return transforms.Compose(train_transform)

def get_validation_augmentation(config):
    val_transform = [
        Resize(size=(config["resize_height"], config["resize_width"])),
        ToTensor(),
    ]
    return transforms.Compose(val_transform)
