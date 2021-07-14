# importing important librarires
import itertools

import numpy as np
import torch
import pydicom
from PIL import Image
from torch.utils.data import DataLoader




def load_scan(path):
    """
    This function is used to load the MRI scans. It converts the scan into a numpy array

      Parameters:
        path (str): The path to the folder containing the MRI scans of all patients

      Returns:
        np_image (numpy.ndarray): A numpy array representing the MRI scan
    """

    # slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    # slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    # try:
    #     slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    # except Exception as e:
    #     print("Exception raised: ", e)
    #     slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    #  for s in slices:
    #     s.SliceThickness = slice_thickness

    #  image = np.stack([s.pixel_array for s in slices])
    image = pydicom.read_file(path)
    # print(type(image))
    image = image.pixel_array.astype(np.int16)
    np_image = np.array(image, dtype=np.int16)
    # print("scan shape: ", np_image.shape)
    return np_image


def load_seg(path):
    """
    This function is used to load the segmented image. It returns the image in a numpy array

      Parameters:
        path (str): The directory where all the segmented images corresponding to one patient are stored

      Returns:
        seg_data (numpy.ndarray): A list of numpy arrays corresponding to segmented images
    """
    # seg_paths = []

    # if path[-1] != '/':
    #   path = path + '/'

    # for seg in os.listdir(path):
    #   seg_paths.append(path + seg)

    # seg_paths.sort()

    seg = Image.open(path)
    seg_data = np.asarray(seg)
    seg_data = np.array(seg_data)
    # for seg_path in seg_paths:
    #   seg = Image.open(seg_path)
    #   seg_data.append(np.asarray(seg))
    # print("seg shape: ", seg_data.shape)

    ### This block of code was to list the different intensity values

    # for arr in seg_data:
    #   for elem in arr:
    #     if (elem not in seg_val):
    #       seg_val.append(elem)

    return seg_data


def resize_data(data, new_dimensions):
    '''
    This function resizes a numpy array.
    TO DO: method used for interpolation?

      Parameters:
        data (numpy.ndarray): a numpy array representing an MRI scan
        new_dimensions (list): a list containing the dimensions of the new scan [z,x,y]

      Returns:
        new_data (numpy.ndarray): a numpy array with the desired dimensions
    '''
    initial_size_x = data.shape[1]
    initial_size_y = data.shape[2]
    initial_size_z = data.shape[0]

    new_size_z = new_dimensions[0]
    new_size_x = new_dimensions[1]
    new_size_y = new_dimensions[2]

    delta_x = initial_size_x / new_size_x
    delta_y = initial_size_y / new_size_y
    delta_z = initial_size_z / new_size_z

    new_data = np.zeros((new_size_z, new_size_x, new_size_y))

    for x, y, z in itertools.product(range(new_size_x),
                                     range(new_size_y),
                                     range(new_size_z)):
        new_data[z][x][y] = data[int(z * delta_z)][int(x * delta_x)][int(y * delta_y)]

    return new_data


def padSlice(values):
    # print(values.shape)
    target_shape = np.array((320, 320))
    pad = ((target_shape - values.shape) / 2).astype("int")

    values = np.pad(values, ((pad[0], pad[0]), (pad[1], pad[1])), mode="constant", constant_values=0)

    return values


def findOrgan(img, seg, organ):
    if organ == 'rk':
        value = 126
    elif organ == 'lk':
        value = 189
    elif organ == 'lv':
        value = 63
    elif organ == 'sp':
        value = 252
    else:
        print("Wrong organ selected.")
        print("Right kidney: rk \nLeft kidney: lk \nLiver: lv \nSpleen: sp")
        new_seg = np.zeros(seg.shape)
        new_img = np.zeros(img.shape)
        return new_img, new_seg

    new_seg = np.zeros(seg.shape)
    new_img = np.zeros(img.shape)
    indices = np.where(seg == value)  # tuple of 2 arrays [i0,i1,...,in], [j0,j1,...,jn], where seg[i][j] == value
    for i in range(len(indices[0])):
        row = indices[0][i]
        col = indices[1][i]

        new_img[row][col] = img[row][col]
        new_seg[row][col] = 1

    return new_img, new_seg

def check_accuracy(loader, model, loss_fn, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            # print("x: ", x.shape)
            # print("y: ", y.shape)
            x = x.unsqueeze(1).to(device)
            # print("x: ", x.shape)
            y = y.unsqueeze(1).to(device)
            # print("mo la")
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            loss = loss_fn.forward(preds,y)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return loss, dice_score/len(loader)

def save_checkpoint(state, filename="my_checkpoint2liver.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_ds, val_ds):
  train_dataloader = DataLoader(train_ds, batch_size=12)
  val_dataloader = DataLoader(val_ds, batch_size=12)
  return train_dataloader, val_dataloader


