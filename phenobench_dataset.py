import json
import cv2
from os.path import join
import numpy as np
from glob import glob

from torch.utils.data import Dataset


class Phenobench(Dataset):
    def __init__(self, root_dir):
        self.source_image_dir = join(root_dir, 'plants_panoptic_train')
        self.target_image_dir= join(root_dir, 'train', 'images')
        self.source_images = sorted(glob(self.source_image_dir+'/*.png'))
        self.target_images = sorted(glob(self.target_image_dir+'/*.png'))
        assert len(self.source_images) == len(self.target_images)

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):

        source_filename = self.source_images[idx]
        target_filename = self.target_images[idx]
        if '05-15' in target_filename:
            prompt = "sugarbeet crops and weed plants of different species in early stages with sunny lighting conditions in the morning and dry darker brown soil background"
        elif '05-26' in target_filename:
            prompt = "sugarbeet crops and weed plants of different species in early stages with sunny lighting conditions in the afternoon and dry lighter brown soil background"
        elif '06-05' in target_filename:
            prompt = "sugarbeet crops and weed plants of different species in slightly later growth stages with overcast weather conditions without shadows and dark brown soil background with a bit of moisture"
        else:
            prompt = "None"

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

