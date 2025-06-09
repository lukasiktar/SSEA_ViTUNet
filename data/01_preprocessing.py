
import os
import glob

import cv2
from tqdm import tqdm
import pydicom
import numpy as np
import pandas as pd

from natsort import natsorted
# from pydicom.dataset import FileDataset, FileMetaDataset
from sklearn.model_selection import train_test_split

#Data preprocessing and 2d images generation
image_path='data/DICOM_dataset/micro_ultrasound_images/'
mask_path='data/DICOM_dataset/micro_ultrasound_masks/'
preannotation_path='data/DICOM_dataset/micro_ultrasound_preannotations/'

#Lists of preprocessed images
list_path='TransUNet/lists/'
#Stored images
out_image_path='data/train_png/'
out_validation_image_path='data/validation_png/'
out_test_image_path='data/test_png/'

#Create train, validation and test dataset
os.makedirs(list_path, exist_ok=True)
os.makedirs(out_image_path, exist_ok=True)
os.makedirs(out_validation_image_path, exist_ok=True)
os.makedirs(out_test_image_path, exist_ok=True)

#Store filenames into a list
list_of_image=glob.glob(image_path + "*.dcm")
list_of_mask=glob.glob(mask_path + "*.dcm")
list_of_preannotation=glob.glob(preannotation_path + "*dcm")

data = list(zip(natsorted(list_of_image), natsorted(list_of_mask), natsorted(list_of_preannotation)))

# split into train and temp 
train_data, temp_data = train_test_split(data, test_size=0.3, shuffle=False)
# split temp into validation and test
validation_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

list_of_image, list_of_mask, list_of_preannotation = zip(*train_data)
list_of_validation_image, list_of_validation_mask, list_of_validation_preannotation = zip(*validation_data)
list_of_test_image, list_of_test_mask, list_of_test_preannotation = zip(*test_data)

list_of_image = natsorted(list_of_image)
list_of_mask = natsorted(list_of_mask)
list_of_preannotation = natsorted(list_of_preannotation)
list_of_test_image = natsorted(list_of_test_image)
list_of_validation_mask = natsorted(list_of_validation_mask)
list_of_test_mask = natsorted(list_of_test_mask)


assert len(list_of_image) == len(list_of_mask) == len(list_of_preannotation), 'Each training case must contain image, mask and non-expert annotation.'
assert len(list_of_validation_image) == len(list_of_validation_mask), 'Each validation case must contain image and mask.'
assert len(list_of_test_image) == len(list_of_test_mask), 'Each testing case must contain image and mask.'

# # Downsample images by 2 for storage first. All images will be resized to 224*224 afterward.
# down = 2
# width = int(1372/down)
# height = int(962/down)

print('Preprocessing starts!')
print('There are {} images, {} masks and {} non-expert annotations for training.'.format(len(list_of_image), len(list_of_mask), len(list_of_preannotation)))
print('There are {} images, {} masks for validation.'.format(len(list_of_validation_image),len(list_of_validation_mask)))
print('There are {} images, {} masks for testing.'.format(len(list_of_test_image), len(list_of_test_mask)))

#Load DICOM images
print('Storing train images.')
for i in tqdm(range(len(list_of_image))):
    img_name = list_of_image[i]
    mask_name = list_of_mask[i]
    prean_name = list_of_preannotation[i]

    dicom_img_data=pydicom.dcmread(img_name)
    img_data=dicom_img_data.pixel_array
    img=cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dicom_mask_data=pydicom.dcmread(mask_name)
    mask_data=dicom_mask_data.pixel_array
    mask=cv2.normalize(mask_data, None, 0, 255, cv2.NORM_MINMAX)
    mask=cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    dicom_prean_data=pydicom.dcmread(prean_name)
    prean_data=dicom_prean_data.pixel_array
    prean=cv2.normalize(prean_data, None, 0, 255, cv2.NORM_MINMAX)
    prean=cv2.cvtColor(prean, cv2.COLOR_BGR2RGB)

    # img_resized=cv2.resize(img, (width, height))
    # mask_resized=cv2.resize(mask, (width, height))
    # prean_resized=cv2.resize(prean, (width, height))

    sub_name = img_name.split("/")[-1].split("_")[0]
    idx = img_name.split("/")[-1].split("_")[-1].split(".")[0]
    output_image_name = out_image_path + sub_name + "_train_img_slice_" + idx + ".png"
    output_mask_name = out_image_path + sub_name + "_train_mask_slice_" + idx + ".png"
    output_prean_name = out_image_path + sub_name + "_train_prean_slice_" + idx + ".png"

    cv2.imwrite(output_image_name, img)
    cv2.imwrite(output_mask_name, mask)
    cv2.imwrite(output_prean_name, prean)

print('Storing validation images.')
for i in tqdm(range(len(list_of_validation_image))):
    val_img_name = list_of_validation_image[i]
    val_mask_name = list_of_validation_mask[i]

    dicom_val_img_data=pydicom.dcmread(val_img_name)
    val_img_data=dicom_val_img_data.pixel_array
    val_img=cv2.normalize(val_img_data, None, 0, 255, cv2.NORM_MINMAX)
    val_img=cv2.cvtColor(val_img, cv2.COLOR_BGR2RGB)

    dicom_val_mask_data=pydicom.dcmread(val_mask_name)
    val_mask_data=dicom_val_mask_data.pixel_array
    val_mask=cv2.normalize(val_mask_data, None, 0, 255, cv2.NORM_MINMAX)
    val_mask=cv2.cvtColor(val_mask, cv2.COLOR_BGR2RGB)

    sub_name = val_img_name.split("/")[-1].split("_")[0]
    idx = val_img_name.split("/")[-1].split("_")[-1].split(".")[0]
    output_val_image_name = out_validation_image_path + sub_name + "_val_img_slice_" + idx + ".png"
    output_val_mask_name = out_validation_image_path + sub_name  + "_val_mask_slice_" + idx + ".png"

    cv2.imwrite(output_val_image_name, val_img)
    cv2.imwrite(output_val_mask_name, val_mask)

print('Storing test images.')
for i in tqdm(range(len(list_of_test_image))):
    test_img_name = list_of_test_image[i]
    test_mask_name = list_of_test_mask[i]

    dicom_test_img_data=pydicom.dcmread(test_img_name)
    test_img_data=dicom_test_img_data.pixel_array
    test_img=cv2.normalize(test_img_data, None, 0, 255, cv2.NORM_MINMAX)
    test_img=cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    dicom_test_mask_data=pydicom.dcmread(test_mask_name)
    test_mask_data=dicom_test_mask_data.pixel_array
    test_mask=cv2.normalize(test_mask_data, None, 0, 255, cv2.NORM_MINMAX)
    test_mask=cv2.cvtColor(test_mask, cv2.COLOR_BGR2RGB)

    sub_name = test_img_name.split("/")[-1].split("_")[0]
    idx = test_img_name.split("/")[-1].split("_")[-1].split(".")[0]
    output_test_image_name = out_test_image_path + sub_name + "_test_img_slice_" + idx + ".png"
    output_test_mask_name = out_test_image_path + sub_name + "_test_mask_slice_" + idx + ".png"

    cv2.imwrite(output_test_image_name, test_img)
    cv2.imwrite(output_test_mask_name, test_mask)

# Generate CSV file for checking data.
print('Start to generate csv file!')
def extract_numbers(filename):
    parts = filename.split("_")
    first_number = int(parts[1].split("/")[-1])  
    second_number = int(parts[-1].split("_")[-1].split('.')[0])  
    return (first_number, second_number)

image_names = sorted(glob.glob(out_image_path + "*img_slice*"),key=extract_numbers)
mask_names = sorted(glob.glob(out_image_path + "*mask_slice*"),key=extract_numbers)
prean_mask_names = sorted(glob.glob(out_image_path + "*prean_slice*"),key=extract_numbers)
validation_image_names = sorted(glob.glob(out_validation_image_path + "*val_img_slice*"), key=extract_numbers)
validation_mask_names = sorted(glob.glob(out_validation_image_path + "*val_mask_slice*"), key=extract_numbers)
test_image_names = sorted(glob.glob(out_test_image_path + "*test_img_slice*"),key=extract_numbers)
test_mask_names = sorted(glob.glob(out_test_image_path + "*test_mask_slice*"),key=extract_numbers)

array = np.empty((len(image_names) + 1,7), dtype='U60')
array[0,0] = "image"
array[0,1] = "mask"
array[0,2] = "preannotation_mask"
array[0,3] = "validation_image"
array[0,4] = "validation_mask"
array[0,5] = "test_image"
array[0,6] = "test_mask"

for i in range(1,len(image_names)+1):
    array[i,0] = image_names[i-1].replace(out_image_path,"").split('.')[0]
    array[i,1] = mask_names[i-1].replace(out_image_path,"").split('.')[0]
    array[i,2] = prean_mask_names[i-1].replace(out_image_path,"").split('.')[0]

for i in range(1, len(validation_image_names)+1):
    array[i,3] = validation_image_names[i-1].split('/')[-1].split('.')[0]
    array[i,4] = validation_mask_names[i-1].split('/')[-1].split('.')[0]

for i in range(1, len(test_image_names)+1):
    array[i,5] = test_image_names[i-1].split('/')[-1].split('.')[0]
    array[i,6] = test_mask_names[i-1].split('/')[-1].split('.')[0]


    
np.savetxt('data.csv', array, delimiter=",", fmt='%s')
print('Finished generating data.csv file!')

# Generate lists for loading data
print('Start to generate list files!')

# image txt
data= pd.read_csv('data.csv')
key='image'
num = data[key].values.size
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)

with open('TransUNet/lists/image.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

# mask txt
key='mask'
num = data[key].values.size
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)

with open('TransUNet/lists/mask.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

# non-expert txt
key='preannotation_mask'
num = data[key].values.size
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)

with open('TransUNet/lists/preannotation.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

# validation image txt
key='validation_image'
num = data[key].count()
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)

with open('TransUNet/lists/validation_image.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

# validation mask txt
key='validation_mask'
num = data[key].count()
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)

with open('TransUNet/lists/validation_mask.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

# test image txt
key='test_image'
num = data[key].count()
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)


with open('TransUNet/lists/test_image.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

# test mask txt
key='test_mask'
num = data[key].count()
name = []
for i in range(num):
    a = data[key].values[i]
    name.append(a)

with open('TransUNet/lists/test_mask.txt', 'w') as f:
    for item in name:
        f.write("%s\n" % item)

print('Finished generating list files!')
print('Preprocessing done!')