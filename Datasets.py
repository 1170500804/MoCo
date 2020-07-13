from __future__ import print_function, division

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import skimage
class Rolling_Window_Year_Dataset(Dataset):
    '''
    Generic Dataset to access building type information. Possible values are
    'building_address_full',
       'first_floor_elevation_ft', 'assessment_type', 'year_built',
       'effective_year_built', 'roof_shape', 'roof_cover', 'wall_cladding',
       'number_of_stories', 'building_address_full_cleaned'
    '''

    def __init__(self, attribute_name, csv_path, img_path, transform=None, regression=False, mask_buildings=False, softmask=False):
        if (attribute_name != 'year_built' and attribute_name != 'effective_year_built:') or regression:
            raise ValueError('Wrong attribute or training type for this dataset')

        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.regression = regression
        self.attribute_name = attribute_name
        self.mask_buildings = mask_buildings
        self.softmask=softmask

        #min_year = self.df[self.attribute_name].min()
        #max_year = self.df[self.attribute_name].max()
        #max_year += int((max_year - min_year) % 10)+2 # padd to full 10 year intervals
        min_year = 1913
        max_year = 2023 # Not all datasets have all years so this needs to be hard set

        #classes = sliding_window(np.array(range(int(min_year),int(max_year))), size=10, stepsize=10)
        classes = skimage.util.view_as_windows(np.array(range(int(min_year),int(max_year))),10,step=10)
        self.class_names = [(str(start) + '-' + str(end)) for start, end in zip(classes[:, 0], classes[:, -1])]
        self.label_lookup = {}
        for year in self.df[self.attribute_name].unique():
            for i in range(len(classes)):
                if int(year) in classes[i]:
                    self.label_lookup[int(year)] = i
                    break


        self.img_path = img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx]['filename']
        image = Image.open(os.path.join(self.img_path, img_name))

        if self.mask_buildings:
            image = np.array(image)
            if self.softmask:
                mask_filename = self.df.iloc[idx]['filename'].replace('.jpg', '-softmask.npy')
                mask = np.load(os.path.join(self.img_path,mask_filename))
                mask = np.array(mask)
                image = np.array(np.stack(
                    (image[:, :, 0] * mask, image[:, :, 1] * mask, image[:, :, 2] * mask), 2),
                         dtype=np.uint8)
                #plt.imshow(image)
                #plt.show()
            else:
                mask_filename = self.df.iloc[idx]['filename'].replace('jpg', 'png')
                mask = Image.open(os.path.join(self.img_path, mask_filename))
                mask = np.array(mask)
                # Filter building labels
                mask[np.where((mask != 25) & (mask != 1))] = 0
                image[mask == 0, :] = 0
                #plt.imshow(image)
                #plt.show()
            image = Image.fromarray(np.uint8(image))

        label = self.df.iloc[idx][self.attribute_name]
        label = self.label_lookup[int(label)] # Translate to coarse class




        if (self.transform):
            image = self.transform(np.array(image))

        return (image, label)
