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


class VicCycLegacyDataset(data.Dataset):
    """
    Characterizes Victorian On-bike Cycling (Legacy) dataset for PyTorch
    """
    def __init__(self, csv_data_path, image_data_path, optical_flow_data_path, split, forward_frame_len, backward_frame_len,
                 transform=None):
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
        self.image_data_path = image_data_path
        self.optical_flow_data_path = optical_flow_data_path
        self.data = self.data[self.data['Split'] == split]   # Retrieve train, validation and test dataset according to split
        self.data.reset_index(drop=True, inplace=True)
        self.forward_frame_len = int(forward_frame_len)
        self.backward_frame_len = int(backward_frame_len)
        self.transform = transform                         # Transform image size and value range, etc.
        # import pdb;
        # pdb.set_trace()
        # row = self.data.loc[0]
        #
        # os.path.join(self.image_data_path, row['ClipFramePath'])
        # os.path.join(self.optical_flow_data_path, row['ClipFramePath'])


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

        img_X, opt_X = [], []
        # time = row['Time']                               # The time that an event is happening within a video
        # h, m, s = time.split(':')
        # secs = float(h)*60*60+float(m)*60+float(s)       # Convert time to seconds
        # frame_id = round(secs*25)
        clip_duration = int(row['ClipFramePath'].split('_')[1].split('Duration')[1].split('s')[0])  # Event happens in the middle of a video clip
        frame_id = int(clip_duration / 2 * 25)             # Calculate the frame id where an event is detected and FPS = 25
        for i in range(frame_id-self.forward_frame_len, frame_id+self.backward_frame_len):         # Retrieve frames about 1s before and after the event
            image = Image.open(os.path.join(self.image_data_path, row['ClipFramePath'], 'image_{:06d}.jpg'.format(i)))  # Video is pre-extracted as frames saving in FramePath
            optical_flow = Image.open(os.path.join(self.optical_flow_data_path, row['ClipFramePath']+'.mp4', 'optical_flow_image_{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)             # Apply defined transform
                optical_flow = use_transform(optical_flow)             # Apply defined transform
            img_X.append(image)
            opt_X.append(optical_flow)
        img_X = torch.stack(img_X, dim=0)
        opt_X = torch.stack(opt_X, dim=0)

        return img_X, opt_X

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        row = self.data.loc[index]

        # Load data
        img_X, opt_X = self._read_images(row, self.transform)     # (input) spatial images
        if row['NearMiss'] == 0:
            y = torch.LongTensor([0])
        elif row['NearMiss'] == 1:
            y = torch.LongTensor([1])
        else:
            raise ValueError('{} has a wrong label {}!'.format(index, row['NearMiss']))
        return img_X, opt_X, y