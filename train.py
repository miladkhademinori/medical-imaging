# from as3 import Augmentation
# import the optimizer
from torch.optim import Adam
import torch
# import Dataloader
from torch.utils.data import DataLoader, Dataset
# import random_split
from torch.utils.data import random_split
# import TensorDataset
from torch.utils.data import TensorDataset

# Augmentation: Data augmentation is used in DL to increase the sample size and
# improve model training. You can simply double the number of your samples by flipping
# the images horizontally (left-right flip). Note that the masks should also be flipped for
# these images as well. Double your data size using this approach.


# Normalization: As a preprocessing step, use z-score to normalize each MRI image (NOT
# the masks) in your dataset, which means the intensity of pixels in each image will have a
# mean of 0 and variance of 1.

# Define a Pytorch transform to do augmentation (flip the image and mask)
# And also normalize the image (only the image, not the mask)
# The normalization should be done over each image slice not the entire images
# In other words, the mean and std should be calculated for each image slice
# And then the image slice should be normalized using the calculated mean and std
class Augmentation(object):
    def __init__(self, flip=False, normalize=False):
        self.flip = flip
        self.normalize = normalize

    def __call__(self, image, mask):
        # Flip the image and mask
        if self.flip:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [2])

        # Normalize the image
        if self.normalize:
            # Calculate the mean and std for each image slice
            mean = torch.mean(image)
            std = torch.std(image)

            # Normalize the image
            image = (image - mean) / std

        return image, mask


# Define the loss function
# Loss function: You should use Dice Score (DSC) for optimization of weights and
# monitoring of the performance during training. Note that while DSC is used as your
# metric, one-minus-DSC should be used as your loss.

def one_minus_dice_loss(predicted_mask, true_mask):
    """
    Computes the one-minus-Dice Score loss between the predicted mask and true mask.
    """
    # Flatten predicted and true masks
    predicted_mask = predicted_mask.view(-1)
    true_mask = true_mask.view(-1)

    # Compute intersection and sum of predicted and true masks
    intersection = torch.sum(predicted_mask * true_mask)
    sum_masks = torch.sum(predicted_mask) + torch.sum(true_mask)

    # Calculate Dice Score
    dice_score = 2.0 * intersection / (sum_masks + 1e-7)

    # Calculate one-minus-Dice Score loss
    loss = 1.0 - dice_score

    return loss


# Training: Train 3 models with the following input / output pairs:
# a) T2W MRI images / prostate gland
# b) ADC MRI images / cancer lesions
# c) HBV MRI images / cancer lesions
# Train the model with the training data. Monitor the performance of the model at the
# end of each epoch by plotting the losses of training and validation data. Use early
# stopping based on the performance of validation data (validation loss) to avoid
# overfitting. After training, report the average performance of your model on segmenting
# the test data. Do the same for validation and training sets as well.

def train_model(model, images, labels, learning_rate, batch_size, num_epochs):
    # Define the loss function
    
    # Use the Augementation class to augment the data
    # and feed it to the DataLoader class
    # data_aug = Augmentation()
    # train_dataset = Dataset(images, labels, transform=data_aug)
    # Devide the data into training, validation and test sets

    # Use the Augementation class to augment the data of the training set
    # and feed it to the DataLoader class

    # Get the number of images, in other words the dataset size (this vatiable is a tensor)
    dataset_size = len(images)
    train_size = int(0.8 * dataset_size) # 80% of data for training
    val_size = int(0.1 * dataset_size) # 10% of data for validation
    test_size = dataset_size - train_size - val_size # the remaining 10% of data for testing

    # val_dataset = Dataset(image_val, mask_val, transform=data_aug)

    #image = torch.unsqueeze(image, dim=1) 
    # # add an extra channel dimension to both the image and mask

    images = torch.unsqueeze(images, dim=1)
    labels = torch.unsqueeze(labels, dim=1)



    dataset = TensorDataset(images, labels)

    # Split the data into training, validation and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Use flpping and normalization for the training set before feeding it to the DataLoader class
    data_aug = Augmentation(flip=True, normalize=True)
    train_dataset = Dataset(train_dataset, transform=data_aug)

    # Load the data into the DataLoader class for training, validation and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Use the Augementation class to augment the data of the validation set
    # and feed it to the DataLoader class

    # Define the optimizer which is adam in this case
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Load the model and data to the GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        train_loader = train_loader.cuda()
        val_loader = val_loader.cuda()
        test_loader = test_loader.cuda()

    # Train the model
    for epoch in range(num_epochs):
        # Train the model
        for image, mask in train_loader:
            # Predict the mask

            model.train()
            breakpoint()
            predicted_mask = model(image)

            # Calculate the loss
            loss = one_minus_dice_loss(predicted_mask, mask)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate the model
        for image, mask in val_loader:
            model.eval()

            # Predict the mask
            predicted_mask = model(image)

            # Calculate the loss
            loss_val = one_minus_dice_loss(predicted_mask, mask)
        
        # Print the both the training and validation loss
        print('Epoch [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}' \
            .format(epoch+1, num_epochs, loss.item(), loss_val.item()))

        

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

    return model


