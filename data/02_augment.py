import glob
import cv2
import random
import numpy as np
import pandas as pd

def blur(image, blur_val:int=5):
    '''
    Blurs an image with given blur value.
    
    Arguments:
        image_path (str): path to image
        blur_val (int): blur noise value
    '''
    
    blur_val = blur_val if blur_val % 2 == 1 else blur_val + 1
    aug_img = cv2.blur(image,(blur_val, blur_val))
    return aug_img
    
def rotate(image, angle:int):
    '''
    Rotates the image around its axis by a given angle.
    
    Arguments:
        image_path (str): path to image
    '''
    
    if angle < -180 or angle > 180:
        print('Invalid angle...modifying')
        angle = angle % 180 if angle > 0 else -(abs(angle) % 180)
        print(f'using angle {angle}')
        
    
    rows, cols = image.shape[:2]
    cx, cy = rows, cols # center of rotation
    M = cv2.getRotationMatrix2D((cy//2, cx//2), angle, 1)
    
    return cv2.warpAffine(image, M, (cols, rows))

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


list_of_train_images=glob.glob(out_image_path + "*.png")
list_of_validaiton_images=glob.glob(out_validation_image_path + "*.png")
list_of_test_images=glob.glob(out_test_image_path + "*png")

#Sort the image names and append the list with elements consisting of orig image, mask and preannotation
for image_name in list_of_train_images:
    if image_name.split("/")[-1].split("_")[2]=="img":
        mask_name=image_name.replace("img", "mask")
        preannotation_name=image_name.replace("img","prean")
        print(image_name)
        image=cv2.imread(image_name)
        mask=cv2.imread(mask_name)
        preannotation=cv2.imread(preannotation_name)

        blur_val = random.randint(1,11)
        angle = random.randint(-10,10)

        augmented_image= blur(image, blur_val=blur_val)
        augmented_image= rotate(augmented_image, angle=angle)


        augmented_mask= blur(mask, blur_val=blur_val)
        augmented_mask= rotate(augmented_mask, angle=angle)

        augmented_preannotation= blur(preannotation, blur_val=blur_val)
        augmented_preannotation= rotate(augmented_preannotation, angle=angle)

        augmented_image_name = image_name.replace("img", "augmentedimg")
        augmented_mask_name = mask_name.replace("mask", "augmentedmask")
        augmented_preannotation_name = preannotation_name.replace("prean", "augmentedprean")

        cv2.imwrite(augmented_image_name, augmented_image)
        cv2.imwrite(augmented_mask_name, augmented_mask)
        cv2.imwrite(augmented_preannotation_name, augmented_preannotation)



# Generate CSV file for checking data.
print('Start to generate csv file!')
def extract_numbers_dicom(filename):
    parts = filename.split("_")
    first_number = int(parts[1].split("/")[-1])  
    second_number = int(parts[-1].split("_")[-1].split('.')[0])  
    return (first_number, second_number)

def extract_numbers(filename):
    parts = filename.split("_")
    first_number = int(parts[2])  
    second_number = int(parts[-1].split("_")[-1].split('.')[0])  
    return (first_number, second_number)



if image_path.split("/")[1]=="DICOM_dataset":
    image_names = sorted(glob.glob(out_image_path + "*img_slice*"),key=extract_numbers_dicom)
    mask_names = sorted(glob.glob(out_image_path + "*mask_slice*"),key=extract_numbers_dicom)
    prean_mask_names = sorted(glob.glob(out_image_path + "*prean_slice*"),key=extract_numbers_dicom)
    validation_image_names = sorted(glob.glob(out_validation_image_path + "*val_img_slice*"), key=extract_numbers_dicom)
    validation_mask_names = sorted(glob.glob(out_validation_image_path + "*val_mask_slice*"), key=extract_numbers_dicom)
    test_image_names = sorted(glob.glob(out_test_image_path + "*test_img_slice*"),key=extract_numbers_dicom)
    test_mask_names = sorted(glob.glob(out_test_image_path + "*test_mask_slice*"),key=extract_numbers_dicom)

    array = np.empty((len(image_names) + 1,7), dtype='U70')
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
else:
    image_names = sorted(glob.glob(out_image_path + "*img_slice*"),key=extract_numbers)
    seg_names = sorted(glob.glob(out_image_path + "*gt_slice*"),key=extract_numbers)
    student_seg_names = sorted(glob.glob(out_image_path + "*st_slice*"),key=extract_numbers)
    validation_image_names = sorted(glob.glob(out_validation_image_path + "*val_img_slice*"), key=extract_numbers)
    validation_seg_names = sorted(glob.glob(out_validation_image_path + "*val_gt_slice*"), key=extract_numbers)
    test_image_names = sorted(glob.glob(out_test_image_path + "*test_img_slice*"),key=extract_numbers)
    test_seg_names = sorted(glob.glob(out_test_image_path + "*test_gt_slice*"),key=extract_numbers)

    array = np.empty((len(image_names) + 1,7), dtype='U70')
    array[0,0] = "image"
    array[0,1] = "mask"
    array[0,2] = "non_expert_mask"
    array[0,3] = "validation_image"
    array[0,4] = "validation_mask"
    array[0,5] = "test_image"
    array[0,6] = "test_mask"

    for i in range(1,len(student_seg_names)+1):
        array[i,0] = image_names[i-1].replace(out_image_path,"").split('.')[0]
        array[i,1] = seg_names[i-1].replace(out_image_path,"").split('.')[0]
        array[i,2] = student_seg_names[i-1].replace(out_image_path,"").split('.')[0]

    for i in range(1, len(validation_image_names)+1):
        array[i,3] = validation_image_names[i-1].split('/')[-1].split('.')[0]
        array[i,4] = validation_seg_names[i-1].split('/')[-1].split('.')[0]

    for i in range(1, len(test_image_names)+1):
        array[i,5] = test_image_names[i-1].split('/')[-1].split('.')[0]
        array[i,6] = test_seg_names[i-1].split('/')[-1].split('.')[0]

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

if image_path.split("/")[1]=="DICOM_dataset":
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
else:
    key='non_expert_mask'
    num = data[key].values.size
    name = []
    for i in range(num):
        a = data[key].values[i]
        name.append(a)

    with open('TransUNet/lists/non_expert.txt', 'w') as f:
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

        