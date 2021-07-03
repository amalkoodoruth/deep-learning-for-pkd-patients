import dataset as ds
import os
from UNET import test

if __name__ == '__main__':
    my_dir = os.getcwd()
    my_dir = my_dir + '/MR2D/train'

    img_paths = []
    for dcm in os.listdir(my_dir + '/X'):
        if dcm != ".DS_Store":
            img_paths.append(my_dir + '/X/' + dcm)
    img_paths.sort()

    seg_paths = []
    for seg in os.listdir(my_dir + '/Y'):
        if seg != ".DS_Store":
            seg_paths.append(my_dir + '/Y/' + seg)
    seg_paths.sort()

    Train = ds.SliceDataset(img_paths, seg_paths)

    count = 0
    for idx in range(Train.__len__()):
        X, Y, name = Train.__getitem__(idx)
        count += 1

    print("Number of images: ", Train.__len__())
    if (count == len(img_paths)):
        print("All images loaded successfully")

    test()




