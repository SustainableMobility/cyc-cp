import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import zipfile


class VicCycLegacyDataset(data.Dataset):
    """
    Characterizes Victorian On-bike Cycling (Legacy) dataset for PyTorch
    """
    def __init__(self, csv_data_path, rgb_frame_data_path, opt_frame_data_path, fused_frame_data_path,
                 split, forward_frame_len, backward_frame_len, input_type='rgb_only',
                 use_gray_scale_image=False, transform=None):
        """
        Initialize
        :param csv_data_path: path to the .csv file containing the dataset
        :param image_data_path: path to the images (this is to avoid hard-coded image path in the .csv file)
        :param split: string used to identify train or test
        :param forward_frame_len: number of frames before the keyframe added to the image sequence
        :param backward_frame_len: number of frames after the keyframe added to the image sequence
        :param transform: transform image
        """
        "Initialization"
        self.data = pd.read_csv(csv_data_path)
        self.rgb_frame_data_path = rgb_frame_data_path
        self.opt_frame_data_path = opt_frame_data_path
        self.fused_frame_data_path = fused_frame_data_path
        self.input_type = input_type
        self.data = self.data[self.data['Split'] == split]   # Retrieve train, validation and test dataset according to split
        self.data.reset_index(drop=True, inplace=True)
        self.forward_frame_len = int(forward_frame_len)
        self.backward_frame_len = int(backward_frame_len)
        self.use_gray_scale_image = use_gray_scale_image
        self.transform = transform                         # Transform image size and value range, etc.

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def _read_images(self, row, use_transform):
        """
        Read frame sequence about 1 second before and after an event according to data in each row in .csv file.
        :param row:
        :param use_transform:
        :return:
        """
        rgb_X, opt_X, fused_X = [], [], []
        clip_duration = int(row['ClipFramePath'].split('_')[1].split('Duration')[1].split('s')[
                                0])             # Event happens in the middle of a video clip
        frame_id = int(clip_duration / 2 * 25)  # Calculate the frame id where an event is detected and FPS = 25
        for i in range(frame_id-self.forward_frame_len, frame_id+self.backward_frame_len):         # Retrieve frames about 1s before and after the event
            # Video frames are extracted to a zip file
            rgb_clip_zip_path = os.path.join(self.rgb_frame_data_path, '{}.zip'.format(row['ClipFramePath']))
            opt_clip_zip_path = os.path.join(self.opt_frame_data_path, '{}.zip'.format(row['ClipFramePath']))
            fused_clip_zip_path = os.path.join(self.fused_frame_data_path, '{}.zip'.format(row['ClipFramePath']))
            frame_name = 'image_{:06d}.jpg'.format(i)

            # Read rgb frames
            with zipfile.ZipFile(rgb_clip_zip_path) as rgb_clip_zip:
                rgb_img_f = rgb_clip_zip.open(frame_name)
                rgb_img = Image.open(rgb_img_f)
                rgb_img = rgb_img.convert('L')              # convert to grayscale
            # Read opt frames
            with zipfile.ZipFile(opt_clip_zip_path) as opt_clip_zip:
                opt_img_f = opt_clip_zip.open(frame_name)
                opt_img = Image.open(opt_img_f)
                opt_img = opt_img.convert('L')              # convert to grayscale
            # Read fused frames
            with zipfile.ZipFile(fused_clip_zip_path) as fused_clip_zip:
                fused_img_f = fused_clip_zip.open(frame_name)
                fused_img = Image.open(fused_img_f)
                fused_img = fused_img.convert('L')          # convert to grayscale

            if use_transform is not None:
                rgb_img = use_transform(rgb_img)            # Apply defined transform
                opt_img = use_transform(opt_img)            # Apply defined transform
                fused_img = use_transform(fused_img)        # Apply defined transform
            rgb_X.append(rgb_img.squeeze_(0))               # Transform grayscale cause the 1st dimension to 1
            opt_X.append(opt_img.squeeze_(0))
            fused_X.append(fused_img.squeeze_(0))

        rgb_X = torch.stack(rgb_X, dim=0)
        opt_X = torch.stack(opt_X, dim=0)
        fused_X = torch.stack(fused_X, dim=0)
        # return a sample with format (batch=1, sequence, w, h)
        return rgb_X.unsqueeze_(0), opt_X.unsqueeze_(0), fused_X.unsqueeze_(0)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        row = self.data.loc[index]

        # Load data
        rgb_X, opt_X, fused_X = self._read_images(row, self.transform)    # (input) spatial images
        if row['NearMiss'] == 0:
            y = torch.LongTensor([0])
        elif row['NearMiss'] == 1:
            y = torch.LongTensor([1])
        else:
            raise ValueError('{} has a wrong label {}!'.format(index, row['NearMiss']))
        if self.input_type == 'rgb_only':
            x = rgb_X
        elif self.input_type == 'rgb_and_opt':
            x = [rgb_X, opt_X]
        elif self.input_type == 'fused':
            x = fused_X
        return x, y

