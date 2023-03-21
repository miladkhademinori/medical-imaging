# In previous assignments, you experimented with downloading clinical data of a substantial size,
# exploring the data, and pre-processing, resampling, and filtering the images. In this assignment
# you will apply deep learning (DL) to analyze the images that you have already explored.

# You will devise a segmentation approach for delineation of clinically significant prostate cancer (csPCa)
#  in the PIC-AI challenge dataset. For this assignment, you are using all 5 folds (1500 samples) of the dataset. 
# You will build either a 2D U-Net, nnU-Net, or other variations for this purpose.

import os # for os.path.exists
import SimpleITK as sitk
import numpy as np
import pickle
import torch # for torch.cuda.is_available
import matplotlib.pyplot as plt
import time
from myModel import UNet
from train import train_model


# Define the function that reads all the images as well as the segmentation masks
# Both the prostate mask (anatomical_delineations/AI) and lesion mask (csPCa_lesion_delineations/AI)
# are binary masks. The prostate mask is used to define the region of interest (ROI) for the
# segmentation task. The lesion mask is used to define the positive class for the segmentation task.


# This function walks inside the directory ./picai_public_images_fold0 and returns a list of all the files
# that have the extension .mha and also has t2w, adc, hbv in the file name. It should store them in a list named image_list.
# And then it walks inside the directory ./picai_labels-main/anatomical_delineations/whole_gland/AI/Bosma22b and
# returns a list of all the files that have the extension .nii.gz. It should store them in a list named mask_list.
# Finally, it walks inside the directory ./picai_labels-main/csPCa_lesion_delineations/AI/Bosma22b and
# returns a list of all the files that have the extension .nii.gz. It should store them in a list named lesion_list.
def searcher(image_dir='./picai_public_images_fold0',\
     label_dir='./picai_labels-main/anatomical_delineations/whole_gland/AI/Bosma22b',\
         MRI_type='t2w'):
    # Define the list to store the file names

    image_list = []
    label_list = []

    # Walk through the directory
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            # Check if the file ends with .mha
            if file.endswith('.mha'):
                # Check if the file has the string t2w, adc, or hbv in the file name
                if MRI_type in file:
                    # Append the file name to the list
                    image_list.append(os.path.join(root, file))

    # Walk through the directory
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            # Check if the file ends with .nii.gz
            if file.endswith('.nii.gz'):
                # Append the file name to the list
                label_list.append(os.path.join(root, file))

    # # Walk through the directory
    # for root, dirs, files in os.walk('./picai_labels-main/csPCa_lesion_delineations/AI/Bosma22a'):
    #     for file in files:
    #         # Check if the file ends with .nii.gz
    #         if file.endswith('.nii.gz'):
    #             # Append the file name to the list
    #             lesion_list.append(os.path.join(root, file))

    
    image_label_list = []
    # Fill the mask_list_t2w list
    for file in image_list:
        for mask in label_list:
            # Get the file name with base name
            file_name = os.path.basename(file)
            temp_first = file_name.split('_')[0]
            # Get the mask name with base name
            mask_name = os.path.basename(mask)
            temp_second = mask_name.split('_')[0]
            # Check if the file name and the mask name are the same
            if temp_first == temp_second:
                # Append the mask name to the list
                image_label_list.append([file, mask])


    # image_lesion_list = []
    # # Fill the lesion_list_adc list
    # for file in file_list_adc:
    #     for lesion in lesion_list:
    #         # Get the file name with base name
    #         file_name = os.path.basename(file)
    #         temp_first = file_name.split('_')[0]
    #         # Get the lesion name with base name
    #         lesion_name = os.path.basename(lesion)
    #         temp_second = lesion_name.split('_')[0]
    #         # Check if the file name and the lesion name are the same
    #         if temp_first == temp_second:
    #             # Append the lesion name to the list
    #             image_lesion_list.append([file, lesion])


    # image_lesion_list_hbv = []
    # # Fill the lesion_list_hbv list
    # for file in file_list_hbv:
    #     for lesion in lesion_list:
    #         # Get the file name with base name
    #         file_name = os.path.basename(file)
    #         temp_first = file_name.split('_')[0]
    #         # Get the lesion name with base name
    #         lesion_name = os.path.basename(lesion)
    #         temp_second = lesion_name.split('_')[0]
    #         # Check if the file name and the lesion name are the same
    #         if temp_first == temp_second:
    #             # Append the lesion name to the list
    #             image_lesion_list_hbv.append([file, lesion])    

    return image_label_list

# This function takes the list of file names as input and returns a list of SimpleITK images
# And the segmentation masks and the lesion masks are also read and returned as SimpleITK images
# In a list of the form [image, mask, lesion]
def reader(file_list):
    # Define the list to store the images, masks, and lesions
    # All the images, masks, and lesions are supposed to be stored in this list
    # Which has three columns: the first column is the image, 
    # the second column is the mask, and the third column is the lesion
    image_list = []

    # Loop through the file_list and read the images, masks, and lesions
    # And place them in the first, second, and third column of the list
    for file in file_list:
        image = sitk.ReadImage(file[0])
        mask = sitk.ReadImage(file[1])
        image_list.append([image, mask])

    # Return the list of images, masks, and lesions
    return image_list

# Re-slicing: The dimension of each pixel should be the same in whole dataset to preserve
# the scale in our deep model. Use the resampling code you implemented in Assignment 1
# part 6 to re-slice the data volumes (T2W, ADC, HBV), prostate mask
# (anatomical_delineations/AI) and lesion mask (csPCa_lesion_delineations/AI) to have
# the spacing of (0.5, 0.5, 3.0), for all cases. To make sure that you do this step
# correctly, please double check your re-slicing code with the function provided in the
# preprocessing repository of the challenge:

# Define the function to re-sample the image but be careful that the mask should be re-sampled not
# with the linear interpolation but with the nearest neighbor interpolation
def resample_image(image_list, new_spacing=[0.5, 0.5, 3.0]):
    # Loop through the image_list which contains the images and masks
    for image in image_list:
        
        # Get the original spacing of the image and the mask
        original_spacing = image[0].GetSpacing()
        original_spacing_mask = image[1].GetSpacing()

        # Get the original size of the image and the mask
        original_size = image[0].GetSize()
        original_size_mask = image[1].GetSize()

        # Get the original direction of the image and the mask
        original_direction = image[0].GetDirection()
        original_direction_mask = image[1].GetDirection()

        # Get the original origin of the image and the mask
        original_origin = image[0].GetOrigin()
        original_origin_mask = image[1].GetOrigin()

        # Calculate the new size of the image
        new_size = [int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                    int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                    int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))]

        # Calculate the new size of the mask
        new_size_mask = [int(np.round(original_size_mask[0] * (original_spacing_mask[0] / new_spacing[0]))),
                            int(np.round(original_size_mask[1] * (original_spacing_mask[1] / new_spacing[1]))),
                            int(np.round(original_size_mask[2] * (original_spacing_mask[2] / new_spacing[2])))]

        # Define the resample filter to re-sample the image and the mask
        # Note that the mask should be re-sampled with the nearest neighbor interpolation
        resample = sitk.ResampleImageFilter()
        # Set the interpolator to be sitk.sitkLinear
        resample.SetInterpolator(sitk.sitkLinear)
        # Set the output direction to be the same as the input direction
        resample.SetOutputDirection(original_direction)
        # Set the output origin to be the same as the input origin
        resample.SetOutputOrigin(original_origin)
        # Set the output spacing to be the new spacing
        resample.SetOutputSpacing(new_spacing)
        # Set the size of the output image
        resample.SetSize(new_size)

        # Define the resample filter to re-sample the mask
        # Note that the mask should be re-sampled with the nearest neighbor interpolation
        resample_mask = sitk.ResampleImageFilter()
        # Set the interpolator to be sitk.sitkNearestNeighbor
        resample_mask.SetInterpolator(sitk.sitkNearestNeighbor)
        # Set the output direction to be the same as the input direction
        resample_mask.SetOutputDirection(original_direction_mask)
        # Set the output origin to be the same as the input origin
        resample_mask.SetOutputOrigin(original_origin_mask)
        # Set the output spacing to be the new spacing
        resample_mask.SetOutputSpacing(new_spacing)
        # Set the size of the output image
        resample_mask.SetSize(new_size_mask)

        # Resample the image
        image[0] = resample.Execute(image[0])
        # Resample the mask
        image[1] = resample_mask.Execute(image[1])
        
        new_spacing = image[0].GetSpacing()
        new_spacing_mask = image[1].GetSpacing()
        # print(new_spacing)
        # Get the original size of the image
        new_size = image[0].GetSize()
        new_size_mask = image[1].GetSize()
        # print(new_size)


    # Return the list of images, masks, and lesions
    return image_list



# Cropping: In addition to spacing, the dimension of images (number of pixels) among all
# patients should be the same for deep model training. Crop a (300, 300, 16) pixel
# volume from the MRI volumes and masks that you re-slice in part 1, symmetrically
# around the center voxel of each volume. Do this for all MRI volumes and masks of all.

def crop_image(image_list, crop_size=[300, 300, 16]):
    # Loop through the image_list which contains the images and masks
    for image in image_list:
        # Get the original size of the image
        original_size = image[0].GetSize()
        
        # Get the original size of the mask
        original_size_mask = image[1].GetSize()

        # Calculate the start and end index of the cropping for the image
        start_index = [int((original_size[0] - crop_size[0]) / 2),
                       int((original_size[1] - crop_size[1]) / 2),
                       int((original_size[2] - crop_size[2]) / 2)]
        end_index = [int((original_size[0] + crop_size[0]) / 2),
                     int((original_size[1] + crop_size[1]) / 2),
                     int((original_size[2] + crop_size[2]) / 2)]

        
        # Calculate the start and end index of the cropping for the mask
        start_index_mask = [int((original_size_mask[0] - crop_size[0]) / 2),
                          int((original_size_mask[1] - crop_size[1]) / 2),
                            int((original_size_mask[2] - crop_size[2]) / 2)]
        end_index_mask = [int((original_size_mask[0] + crop_size[0]) / 2),
                        int((original_size_mask[1] + crop_size[1]) / 2),
                        int((original_size_mask[2] + crop_size[2]) / 2)]

        # Define the extract filter to crop the image
        extract = sitk.ExtractImageFilter()

        # Define the extract filter to crop the mask
        extract_mask = sitk.ExtractImageFilter()
        # Set the start index of the cropping
        extract.SetIndex(start_index)
        # Set the start index of the cropping
        extract_mask.SetIndex(start_index_mask)
        # Set the size of the cropping
        extract.SetSize(crop_size)
        # Set the size of the cropping
        extract_mask.SetSize(crop_size)

        # Crop the image
        image[0] = extract.Execute(image[0])
        # Crop the mask
        image[1] = extract_mask.Execute(image[1])

        # Get the cropped size of the image and mask
        cropped_size = image[0].GetSize()
        cropped_size_mask = image[1].GetSize()

    # Return the list of images and masks
    return image_list


# 2D slices: Since we want to train a slice-based 2D deep model, we’ll slice each MRI
# volume in z axis to convert it to 16 MRI images of 300x300 pixels. Do this for all MRI
# volumes and masks of all patients. You will use the MRI images of different sequences
# (T2W, ADC, HBV) as the input, and the mask images (prostate gland, lesions) as the
# output for building your deep model.

# Define the function to slice the image
def slice_image(image_list):
    # Loop through the image_list which contains the images and masks
    for image in image_list:
        # Get the original size of the image
        original_size = image[0].GetSize()
        
        # Define the extract filter
        extract = sitk.ExtractImageFilter()
        # Set the size of the cropping
        extract.SetSize([original_size[0], original_size[1], 1])

        # Create a new list to store the sliced images and masks
        sliced_image_list = []
        sliced_mask_list = []

        # Loop through the z axis of the image
        for i in range(original_size[2]):
            # Set the start index of the cropping
            extract.SetIndex([0, 0, i])
            # Slice the image
            sliced_image = extract.Execute(image[0])
            # Slice the mask
            sliced_mask = extract.Execute(image[1])

            # Append the sliced image and mask to the list
            sliced_image_list.append(sliced_image)
            sliced_mask_list.append(sliced_mask)

        # Replace the image and mask with the sliced images and masks
        image[0] = sliced_image_list
        image[1] = sliced_mask_list

    # Return the list of images and masks
    return image_list


# Define a function that converts the SimpleITK images into one large pytorch tensor where the first dimension is the number of slices
# And the second and third dimensions are the image dimensions
def sitk_to_tensor(image_list):
    # Loop through the image_list which contains the images and masks
    for image in image_list:
        # Get the original size of the image
        original_size = image[0][0].GetSize()
        
        # Create a new list to store the tensors
        tensor_image_list = []
        tensor_mask_list = []

        # Loop through the z axis of the image
        for i in range(len(image[0])):
            # Convert the image to a numpy array
            numpy_image = sitk.GetArrayFromImage(image[0][i])
            # Convert the mask to a numpy array
            numpy_mask = sitk.GetArrayFromImage(image[1][i])

            # Convert the numpy array to a tensor
            tensor_image = torch.from_numpy(numpy_image.astype(np.float32))

            # Convert the numpy array to a tensor
            tensor_mask = torch.from_numpy(numpy_mask.astype(np.float32))

            # Append the tensor to the list
            tensor_image_list.append(tensor_image)
            tensor_mask_list.append(tensor_mask)

        # Replace the image and mask with the tensors
        image[0] = tensor_image_list
        image[1] = tensor_mask_list

        # Print the size of the image and mask using the pytorch tensor size function
        # print(image[0][0].size())
        # print(image[1][0].size())

    # Return the list of images and masks
    return image_list


# Define a function that converts the pytorch tensors into one large pytorch array where the first dimension is the number of slices
# And the second and third dimensions are the image dimensions. This function should use the concatenate function in pytorch
def concat_to_all(image_list):
    # Assume that `image_list` is a Python list with 310 rows,
    # where each row has two columns and each column has 16 images.
    # Here's an example of how to create a mock `image_list` variable:

    length = len(image_list)

    float_tensor1 = torch.zeros((length * 16, 300, 300))
    float_tensor2 = torch.zeros((length * 16, 300, 300))

    # iterate over the nested list
    for i in range(length):
        for j in range(2):
            for k in range(16):
                # calculate the index in the 1D tensor
                idx = i * 16 + k
                # assign value to the appropriate tensor based on j
                if j == 0:
                    float_tensor1[idx] = image_list[i][j][k]
                else:
                    float_tensor2[idx] = image_list[i][j][k]
    return float_tensor1, float_tensor2


# Stratification: Your dataset is already stratified by patients into 5 folds (double check
# the Data Split portion of the challenge webpage). Use folds (1, 2, 4) as training, fold 3 as
# validation, and fold 0 as test. Optional: you can use different combinations of folds and
# run a cross-validation study if you have enough computational power.


# Ablation: You should consider a small ablation study to optimize some of your model
# and training parameters. Do not go overboard with experiments but attempt it for a few
# parameters such as the learning rate, batch size, number of layers. Select the models
# with best performance and for the case you visualized in Assignment 1
# (10522_1000532), show the true masks (prostate gland and lesions) as well as the
# prediction of your 3 best models for comparison

def preprocess(images_path, labels_path, MRI_type):
    pkl_dir = 'lists_' + MRI_type + '.pkl'
    if not os.path.exists(pkl_dir):
        image_list = searcher(images_path, labels_path, MRI_type)
        with open(pkl_dir, 'wb') as f:
            pickle.dump(image_list, f)
    else:
        with open(pkl_dir, 'rb') as f:
            image_list = pickle.load(f)

    image_list = reader(image_list)
    image_list = resample_image(image_list)
    image_list = crop_image(image_list)
    image_list = slice_image(image_list)
    image_list = sitk_to_tensor(image_list)
    images, labels = concat_to_all(image_list)
    return images, labels

# The main function of the program
def main():

    # Check if the GPU is available
    if torch.cuda.is_available():
        print('GPU is available')
    else:
        print('GPU is not available')

    images, labels = preprocess(images_path='./picai_public_images_fold0',\
     labels_path='./picai_labels-main/anatomical_delineations/whole_gland/AI/Bosma22b',\
         MRI_type='t2w')

    # Look now I want to see if everything is working fine
    # Please show the images and masks for all rows in t2w_images_gland
    # Remember that images and masks have 310*16 rows and each data is 300*300
    # Use imshow from matplotlib to show the images and masks
    
    # for i in range(310*16):
    #     plt.imshow(images[i,:,:])
    #     plt.show()
    #     # pause for a second
    #     time.sleep(1)
    #     plt.imshow(masks[i,:,:])
    #     plt.show()
    #     # pause for a second
    #     time.sleep(1)

    # Now I want to define the model and kick off the training
    # Define the model
    model_t2w = UNet()

    # Load the images and masks
    # images, labels = preprocess(image_dir='./picai_public_images_fold0',\
    #  label_dir='./picai_labels-main/anatomical_delineations/whole_gland/AI/Bosma22b',\
    #      MRI_type='t2w')

    # Call the train function
    train_model(model_t2w, images, labels, learning_rate=0.0001, batch_size=64, num_epochs=20)


# Call the main function
if __name__ == '__main__':
    main()

