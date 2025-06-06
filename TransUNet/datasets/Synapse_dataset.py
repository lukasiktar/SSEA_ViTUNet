import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from natsort import natsorted

class MultiscaleGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, preannotation = sample['image'], sample['label'], sample['preannotation']

        x, y = image.shape
        image = cv2.resize(image, self.output_size[3])

        label = cv2.resize(label, self.output_size[3])
        label0 = cv2.resize(label, self.output_size[0])
        label1 = cv2.resize(label, self.output_size[1])
        label2 = cv2.resize(label, self.output_size[2])

        preannotation = cv2.resize(preannotation, self.output_size[3])
        preannotation0 = cv2.resize(preannotation, self.output_size[0])
        preannotation1 = cv2.resize(preannotation, self.output_size[1])
        preannotation2 = cv2.resize(preannotation, self.output_size[2])

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        label = torch.from_numpy(label.astype(np.float32))
        label0 = torch.from_numpy(label0.astype(np.float32))
        label1 = torch.from_numpy(label1.astype(np.float32))
        label2 = torch.from_numpy(label2.astype(np.float32))

        preannotation = torch.from_numpy(preannotation.astype(np.float32))
        preannotation0 = torch.from_numpy(preannotation0.astype(np.float32))
        preannotation1 = torch.from_numpy(preannotation1.astype(np.float32))
        preannotation2 = torch.from_numpy(preannotation2.astype(np.float32))

        sample = {'image': image, 'label': label.long(), 'label0': label0.long(), 'label1': label1.long(), 'label2': label2.long(), 
                  'preannotation': preannotation.long(), 'preannotation0': preannotation0.long(), 'preannotation1': preannotation1.long(), 'preannotation2': preannotation2.long()}
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, dataset_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.data_dir = dataset_dir
        self.image_list = natsorted(open(os.path.join('TransUNet/'+list_dir, 'image'+'.txt')).readlines())
        self.label_list = natsorted(open(os.path.join('TransUNet/'+list_dir, 'mask'+'.txt')).readlines())
        self.preannotation_list = natsorted(open(os.path.join('TransUNet/'+list_dir, 'preannotation'+'.txt')).readlines())
        self.test_image_list = natsorted(open(os.path.join('TransUNet/'+list_dir, 'test'+'_image'+'.txt')).readlines())
        self.test_label_list = natsorted(open(os.path.join('TransUNet/'+list_dir, 'test'+'_mask'+'.txt')).readlines())
        self.validation_image_list = natsorted(open(os.path.join('TransUNet/'+list_dir, 'validation'+'_image'+'.txt')).readlines())
        self.validation_label_list = natsorted(open(os.path.join('TransUNet/'+list_dir, 'validation'+'_mask'+'.txt')).readlines())
        
    def __len__(self):
        if self.split == 'train':
            data_size = len(self.image_list)
        elif self.split == "validation":
            data_size = len(self.test_image_list)
        else:
            data_size = len(self.test_image_list)
        return data_size
    
    def __getitem__(self, idx):
        if self.split == "train":
            image_name = self.image_list[idx].strip('\n')
            label_name = self.label_list[idx].strip('\n')
            preannotation_name = self.preannotation_list[idx].strip('\n')
            image_path = os.path.join(self.data_dir, image_name+'.png')
            label_path = os.path.join(self.data_dir, label_name+'.png')
            preannotation_path = os.path.join(self.data_dir, preannotation_name+'.png')
            #maybe change the image to gray
            image = cv2.imread(image_path)[:,:,0]/255.0
            label = cv2.imread(label_path)[:,:,0]/255.0
            preannotation = cv2.imread(preannotation_path)[:,:,0]/255.0
                       
            sample = {'image': image, 'label': label, 'preannotation': preannotation}
            sample['case_name'] = self.image_list[idx].strip('\n')

        elif self.split == "validation":
            image_name = self.validation_image_list[idx].strip('\n')
            label_name = self.validation_label_list[idx].strip('\n')
            image_path = os.path.join(self.data_dir, image_name+'.png')
            label_path = os.path.join(self.data_dir, label_name+'.png')
            image = cv2.imread(image_path)[:,:,0]/255.0
            label = cv2.imread(label_path)[:,:,0]/255.0           
            #print(f"{image_name}, {label_name}")
            sample = {'image': image, 'label': label}
            sample['case_name'] = self.validation_image_list[idx].strip('\n')
           

        else:
            image_name = self.test_image_list[idx].strip('\n')
            label_name = self.test_label_list[idx].strip('\n')
            image_path = os.path.join(self.data_dir, image_name+'.png')
            label_path = os.path.join(self.data_dir, label_name+'.png')
            image = cv2.imread(image_path)[:,:,0]/255.0
            label = cv2.imread(label_path)[:,:,0]/255.0
           
            sample = {'image': image, 'label': label}
            sample['case_name'] = self.test_image_list[idx].strip('\n')

        if self.transform:
            sample = self.transform(sample)

        
        return sample