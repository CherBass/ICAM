##################################################
# Author: {Cher Bass}
# Copyright: Copyright {2020}, {ICAM}
# License: {MIT license}
##################################################
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import nibabel as nib


class BiobankRegAgeDataset(torch.utils.data.Dataset):
    def __init__(self, image_path='/data/biobank/MNI_aligned_preprocessed_data',
                 mask_path='MNI_aligned_preprocessed_2d_blob_masks',
                 label_path='/data/biobank/labels.pkl',
                 num_classes=2,
                 class_bins=(40,60),
                 class_label=0,
                 get_id=False,
                 transform=None):
        """
        3D MRI dataset loader- biobank
        :param image_path: path to image folder
        :param mask_path: path to mask folder
        :param label_path: path to labels file
        :param num_classes: number of classes
        :param class_bins: age range
        :param class_label: class label
        :param get_id: whether to get subject id for testing
        :param transform: Optional image transforms
        """
        self.transform = transform
        self.num_classes = num_classes
        self.img_dir = image_path
        self.mask_dir = mask_path
        image_paths = sorted(os.listdir(self.img_dir))
        self.labels = pd.read_pickle(label_path)
        self.class_label = class_label
        self.get_id = get_id

        self.labels = self.labels[self.labels['age'] >= class_bins[0]]
        self.labels = self.labels[self.labels['age'] < class_bins[1]]

        # remove image paths in opposite class
        id_list = sorted(image_paths)
        remove_ind = []
        i = 0
        # check which images are present in labels
        for img in id_list:
            f = img.split('_')[0]
            f = f.split('-')
            subject = int(f[1])
            if not (any(self.labels['id'] == subject)):
                remove_ind.append(i)
            i = i + 1
        id_list = [i for j, i in enumerate(id_list) if j not in remove_ind]

        self.image_paths = sorted(id_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx, plot=False):
        img_name = sorted(self.image_paths)[idx]

        image = nib.load(os.path.join(self.img_dir, img_name))
        image = np.float32(image.get_fdata())

        if self.transform:
            image = self.transform(image)

        image = torch.from_numpy(image.copy()).float()
        image = image.unsqueeze(0)

        f = img_name.split('_')[0]
        f = f.split('-')
        subject = int(f[1])
        labels = self.labels.loc[self.labels['id'] == subject]
        label = labels['age'].to_numpy().astype(float)[0]
        label = np.array(label)
        label = torch.from_numpy(label.copy()).float()
        label = label.unsqueeze(0)
        sex = labels['sex'].to_numpy().astype(int)[0]

        mask = torch.zeros(1)

        if self.get_id:
            label = [label, img_name.split('_')[0], sex]

        return image, label, mask
