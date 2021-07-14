from torchvision.transforms import Compose
import numpy as np

def transforms(hflip_prob=None, vflip_prob=None):
    transform_list = []

    if hflip_prob is not None:
        transform_list.append(HorizontalFlip(hflip_prob))

    if vflip_prob is not None:
        transform_list.append(VerticalFlip(vflip_prob))

    return Compose(transform_list)


class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask


class VerticalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()

        return image, mask