from __future__ import print_function, division

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import skimage
class cluster_year_built_dataset(Dataset):
    def __init__(self, batchsize, attribute_name, csv_path, img_path, transform=None, regression=False, mask_buildings=False, softmask=False,steps=10):
        if (attribute_name != 'year_built' and attribute_name != 'effective_year_built:') or regression:
            raise ValueError('Wrong attribute or training type for this dataset: {}'.format(attribute_name))

        self.df = pd.read_csv(csv_path)
        length = len(self.df)
        length = int(length/batchsize) * batchsize
        self.df = self.df.iloc[:length,:]
        self.transform = transform
        self.regression = regression
        self.attribute_name = attribute_name
        self.mask_buildings = mask_buildings
        self.softmask=softmask
        min_year = 1913
        max_year = 2012
        self.img_path = img_path
        self.classes = []
        for year in self.df[self.attribute_name].unique():
            self.classes.append(year)
        self.classes = sorted(self.classes)

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
        # try:
        #     label = self.label_lookup[int(label)] # Translate to coarse class
        # except KeyError:
        #     # year not in class. That can happen when validation classes are not in training
        #     # We choose the most appropriate class instead
        #     for i, class_range in enumerate(self.classes):
        #         if int(label) > class_range[0] and int(label) < class_range[-1]:
        #             label = i
        #             break


        if (self.transform):
            image = self.transform(image)

        return (image, label, []) # [] for compatibility
class Rolling_Window_Year_Dataset(Dataset):
    '''
    Generic Dataset to access building type information. Possible values are
    'building_address_full',
       'first_floor_elevation_ft', 'assessment_type', 'year_built',
       'effective_year_built', 'roof_shape', 'roof_cover', 'wall_cladding',
       'number_of_stories', 'building_address_full_cleaned'
    '''

    def __init__(self, batchsize, attribute_name, csv_path, img_path, transform=None, regression=False, mask_buildings=False, softmask=False,step=10, style='fixed'):
        if (attribute_name != 'year_built' and attribute_name != 'effective_year_built:') or regression:
            raise ValueError('Wrong attribute or training type for this dataset')

        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.regression = regression
        self.attribute_name = attribute_name
        self.mask_buildings = mask_buildings
        self.softmask=softmask
        self.style = style

        #min_year = self.df[self.attribute_name].min()
        #max_year = self.df[self.attribute_name].max()
        #max_year += int((max_year - min_year) % 10)+2 # padd to full 10 year intervals
        min_year = np.min(self.df[attribute_name])
        max_year = np.max(self.df[attribute_name])+step # Not all datasets have all years so this needs to be hard set

        #classes = sliding_window(np.array(range(int(min_year),int(max_year))), size=10, stepsize=10)
        if(style == 'fixed'):
            classes = skimage.util.view_as_windows(np.array(range(int(min_year),int(max_year))),10,step=step)
            self.class_names = [(str(start) + '-' + str(end)) for start, end in zip(classes[:, 0], classes[:, -1])]
            self.label_lookup = {}
            for year in self.df[self.attribute_name].unique():
                for i in range(len(classes)):
                    if int(year) in classes[i]:
                        self.label_lookup[int(year)] = i
                        break
        else:
            self.time_point = np.array([1975, 1983, 1987, 1992, 1996, 2000, 2009, 2015])



        self.img_path = img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx]['filename']
        image = Image.open(os.path.join(self.img_path, img_name))
        if (self.style == 'fixed'):
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
        else:
            label = int(self.df.iloc[idx][self.attribute_name])
            # class 0 is (-inf, 1975]: the sum over relation_with_time_point is 8
            # class 1 is (1975, 1983]: the sum over relation with time point is 7
            #...
            # the summation of class and relation_with_time_point is 8
            relation_with_time_point = (label <= self.time_point)
            label = 8 - np.sum(relation_with_time_point)
        if (self.transform):
            image = self.transform(image)

        return (image, label)


