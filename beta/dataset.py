import os
from utils import findOrgan
import torch
from torch.utils import data
from torch.autograd import Variable

from utils import load_scan, load_seg,padSlice


class SliceDataset(data.Dataset):

    ##
    # img_paths is list of paths to intensity images
    # seg_paths is list of paths to segmentation images, define as None if no segmentations exist
    # sigma is deformation intensity, points the number of coordinates for grid deformation

    ##
    # image path is path of X
    # seg path is path of Y
    def __init__(self, img_paths, seg_paths, transform=None):

        self.seg_paths = seg_paths
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        This function is used to retrieve one scan and its corresponding segmented
        image from a dataset, if it exists. The scan and the segmented image are
        converted into numpy arrays that have the dimensions specified in the constructor.

          Parameters:
            index (int): the index of the scan we want to retrieve. It is in the range [0, size of dataset)

          Returns:
            X (numpy.ndarray): training sample of size new_dimensions
            Y (numpy.ndarray): segmented image if it exists (will exist if in training set). Else, array of 0s

        """
        img_path = self.img_paths[index]
        img = load_scan(img_path)

        seg_exists = len(self.seg_paths) > 0

        if seg_exists:
            seg_path = self.seg_paths[index]
            seg = load_seg(seg_path)

            img_binary, seg_binary = findOrgan(img, seg, 'lv')
            seg_resized = padSlice(seg_binary)

        img_resized = padSlice(img_binary)

        if self.transform is not None:
            img_resized, seg_resized = self.transform((img_resized, seg_resized))

        # Convert images to pytorch tensors
        ## why .float() and .long()
        X = Variable(torch.from_numpy(img_resized)).float()

        if seg_exists:
            Y = Variable(torch.from_numpy(seg_resized)).long()

        else:
            Y = torch.zeros(1)  # dummy segmentation

        name = os.path.basename(self.img_paths[index])

        return X, Y  # , name

