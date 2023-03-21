import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os

# Use matplotlib to plot a figure with x axis and y axis
# Where x axis is the number of epochs and y axis is the loss
# The loss values are stored in the loss_list variable
# The number of epochs is stored in the num_epochs variable

# The loss values are between 0 and 1; therefore, the y axis is between 0 and 1
# The x axis is between 0 and the number of epochs

# We shall have three lines on the figure
# One for the training loss, one for the validation loss and one for the test loss
# The training loss is stored in the train_loss_list variable
# The validation loss is stored in the val_loss_list variable
# The test loss is stored in the test_loss_list variable
def plot_figure_t2w():
    num_epochs = 20
    train_loss_list = [0.6, 0.57, 0.55, 0.54, 0.51, 0.49, 0.47, 0.42, 0.38, 0.37,\
        0.35, 0.32, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22]

    val_loss_list = [0.58, 0.56, 0.52, 0.50, 0.49, 0.46, 0.45, 0.43, 0.41, 0.39,\
        0.38, 0.34, 0.31, 0.33, 0.34, 0.34, 0.35, 0.34, 0.35, 0.35]

    test_loss_list = [0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32,\
        0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]

    # Plot the figure for t2w images with whole gland segmentation
    # The figure is saved in the file t2w_loss.png
    # The training loss is in blue
    # The validation loss is in orange
    # The test loss is in green
    # The y axis limits are between 0 and 1
    plt.figure()
    plt.plot(range(num_epochs), train_loss_list, label='Training Loss')
    plt.plot(range(num_epochs), val_loss_list, label='Validation Loss')
    plt.plot(range(num_epochs), test_loss_list, label='Test Loss (13th Epoch)')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score Loss')
    # Add the title
    plt.title('Loss for T2W Images with Whole Gland Segmentation')
    plt.ylim(0, 1)
    plt.legend()
    # plt.show()
    # tight_layout() is used to adjust the figure size
    plt.tight_layout()
    # Save the figure
    plt.savefig('loss_t2w.png')

# if the loss_t2w.png file does not exist, then plot the figure
# if the loss_t2w.png file exists, then do not plot the figure

if not os.path.exists('loss_t2w.png'):
    plot_figure_t2w()


def plot_figure_adc():
    num_epochs = 20
    train_loss_list = [0.65, 0.61, 0.59, 0.57, 0.56, 0.54, 0.55, 0.53, 0.52, 0.49,\
        0.47, 0.48, 0.45, 0.43, 0.42, 0.40, 0.38, 0.37, 0.37, 0.36]

    val_loss_list = [0.66, 0.63, 0.60, 0.58, 0.57, 0.56, 0.57, 0.54, 0.53, 0.51,\
        0.48, 0.49, 0.47, 0.46, 0.44, 0.44, 0.45, 0.45, 0.46, 0.46]

    test_loss_list = [0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43,\
        0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43]

    # Plot the figure for t2w images with whole gland segmentation
    # The figure is saved in the file t2w_loss.png
    # The training loss is in blue
    # The validation loss is in orange
    # The test loss is in green
    # The y axis limits are between 0 and 1
    plt.figure()
    plt.plot(range(num_epochs), train_loss_list, label='Training Loss')
    plt.plot(range(num_epochs), val_loss_list, label='Validation Loss')
    plt.plot(range(num_epochs), test_loss_list, label='Test Loss (14th Epoch)')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score Loss')
    # Add the title
    plt.title('Loss for ADC Images with Lesion Segmentation')
    plt.ylim(0, 1)
    plt.legend()
    # plt.show()
    # tight_layout() is used to adjust the figure size
    plt.tight_layout()

    # Save the figure
    plt.savefig('loss_adc.png')


if not os.path.exists('loss_adc.png'):
    plot_figure_adc()


def plot_figure_hbv():
    num_epochs = 20
    train_loss_list = [0.64, 0.61, 0.60, 0.58, 0.56, 0.55, 0.53, 0.51, 0.49, 0.48,\
        0.47, 0.46, 0.43, 0.42, 0.42, 0.41, 0.41, 0.39, 0.38, 0.35]

    val_loss_list = [0.63, 0.60, 0.58, 0.59, 0.56, 0.54, 0.53, 0.51, 0.49, 0.47,\
        0.46, 0.45, 0.43, 0.41, 0.43, 0.42, 0.44, 0.45, 0.46, 0.46]

    test_loss_list = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41,\
        0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41]

    plt.figure()
    plt.plot(range(num_epochs), train_loss_list, label='Training Loss')
    plt.plot(range(num_epochs), val_loss_list, label='Validation Loss')
    plt.plot(range(num_epochs), test_loss_list, label='Test Loss (14th Epoch)')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score Loss')
    # Add the title
    plt.title('Loss for HBV Images with Lesion Segmentation')
    plt.ylim(0, 1)
    plt.legend()
    # plt.show()
    # tight_layout() is used to adjust the figure size
    plt.tight_layout()

    # Save the figure
    plt.savefig('loss_hbv.png')


if not os.path.exists('loss_hbv.png'):
    plot_figure_hbv()



def plot_figure_lr():
    num_epochs = 20
    val_loss_1e4 = [0.58, 0.56, 0.52, 0.50, 0.49, 0.46, 0.45, 0.43, 0.41, 0.39,\
        0.38, 0.34, 0.31, 0.33, 0.34, 0.34, 0.35, 0.34, 0.35, 0.35]

    val_loss_1e3 = [0.60, 0.58, 0.57, 0.55, 0.56, 0.55, 0.53, 0.54, 0.53, 0.52,\
        0.50, 0.49, 0.47, 0.47, 0.44, 0.43, 0.42, 0.42, 0.42, 0.41]

    val_loss_1e5 = [0.60, 0.59, 0.58, 0.56, 0.55, 0.54, 0.52, 0.51, 0.50, 0.48,\
        0.47, 0.47, 0.46, 0.43, 0.40, 0.39, 0.38, 0.37, 0.37, 0.36]

    # Plot the figure for t2w images with whole gland segmentation
    # The figure is saved in the file t2w_loss.png
    # The training loss is in blue
    # The validation loss is in orange
    # The test loss is in green
    # The y axis limits are between 0 and 1
    plt.figure()
    plt.plot(range(num_epochs), val_loss_1e4, label='Validation Loss (1e-4)')
    plt.plot(range(num_epochs), val_loss_1e3, label='Validation Loss (1e-3)')
    plt.plot(range(num_epochs), val_loss_1e5, label='Validation Loss (1e-5)')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score Loss')
    # Add the title
    plt.title('The Impact of Learning Rate on Validation Loss')
    plt.ylim(0, 1)
    plt.legend()
    # plt.show()
    # tight_layout() is used to adjust the figure size
    plt.tight_layout()
    # Save the figure
    plt.savefig('loss_t2w_lr.png')

# if the loss_t2w.png file does not exist, then plot the figure
# if the loss_t2w.png file exists, then do not plot the figure

if not os.path.exists('loss_t2w_lr.png'):
    plot_figure_lr()


def plot_figure_bs():
    num_epochs = 20
    val_loss_64 = [0.58, 0.56, 0.52, 0.50, 0.49, 0.46, 0.45, 0.43, 0.41, 0.39,\
        0.38, 0.34, 0.31, 0.33, 0.34, 0.34, 0.35, 0.34, 0.35, 0.35]

    val_loss_32 = [0.60, 0.59, 0.57, 0.56, 0.54, 0.55, 0.54, 0.53, 0.51, 0.52,\
        0.49, 0.48, 0.47, 0.46, 0.45, 0.45, 0.44, 0.43, 0.44, 0.44]

    val_loss_128 = [0.60, 0.56, 0.51, 0.48, 0.46, 0.44, 0.43, 0.41, 0.39, 0.36,\
        0.35, 0.33, 0.32, 0.29, 0.30, 0.32, 0.31, 0.32, 0.33, 0.35]

    # Plot the figure for t2w images with whole gland segmentation
    # The figure is saved in the file t2w_loss.png
    # The training loss is in blue
    # The validation loss is in orange
    # The test loss is in green
    # The y axis limits are between 0 and 1
    plt.figure()
    plt.plot(range(num_epochs), val_loss_64, label='Validation Loss (bs=64)')
    plt.plot(range(num_epochs), val_loss_32, label='Validation Loss (bs=32)')
    plt.plot(range(num_epochs), val_loss_128, label='Validation Loss (bs=128)')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score Loss')
    # Add the title
    plt.title('The Impact of Batch Size on Validation Loss')
    plt.ylim(0, 1)
    plt.legend()
    # plt.show()
    # tight_layout() is used to adjust the figure size
    plt.tight_layout()
    # Save the figure
    plt.savefig('loss_t2w_bs.png')

# if the loss_t2w.png file does not exist, then plot the figure
# if the loss_t2w.png file exists, then do not plot the figure

if not os.path.exists('loss_t2w_bs.png'):
    plot_figure_bs()



def plot_figure_nl():
    num_epochs = 20
    val_loss_4 = [0.58, 0.56, 0.52, 0.50, 0.49, 0.46, 0.45, 0.43, 0.41, 0.39,\
        0.38, 0.34, 0.31, 0.33, 0.34, 0.34, 0.35, 0.34, 0.35, 0.35]

    val_loss_2 = [0.60, 0.58, 0.57, 0.56, 0.53, 0.52, 0.50, 0.49, 0.48, 0.47,\
        0.46, 0.44, 0.43, 0.44, 0.45, 0.46, 0.46, 0.47, 0.47, 0.48]

    # Plot the figure for t2w images with whole gland segmentation
    # The figure is saved in the file t2w_loss.png
    # The training loss is in blue
    # The validation loss is in orange
    # The test loss is in green
    # The y axis limits are between 0 and 1
    plt.figure()
    plt.plot(range(num_epochs), val_loss_4, label='Validation Loss (4 layers)')
    plt.plot(range(num_epochs), val_loss_2, label='Validation Loss (2 layers)')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score Loss')
    # Add the title
    plt.title('The Impact of Number of Layers on Validation Loss')
    plt.ylim(0, 1)
    plt.legend()
    # plt.show()
    # tight_layout() is used to adjust the figure size
    plt.tight_layout()
    # Save the figure
    plt.savefig('loss_t2w_nl.png')

# if the loss_t2w.png file does not exist, then plot the figure
# if the loss_t2w.png file exists, then do not plot the figure

if not os.path.exists('loss_t2w_nl.png'):
    plot_figure_nl()