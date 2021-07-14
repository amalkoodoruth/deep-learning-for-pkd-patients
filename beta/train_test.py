import shutil
import os
import random

master_path = os.getcwd() # i am in beta
x_dir = master_path + '/MR2D/train/X'
y_dir = master_path + '/MR2D/train/Y'

x_paths = []
y_paths = []


for img in os.listdir(x_dir):
    if img != '.DS_Store':
        img_path = x_dir + '/' + img
        x_paths.append(img_path)
x_paths.sort()

for seg in os.listdir(y_dir):
    if seg != '.DS_Store':
        seg_path = y_dir + '/' + seg
        y_paths.append(seg_path)
y_paths.sort()

for idx in range(len(x_paths)):
    assert x_paths[idx][-20:-4] == y_paths[idx][-20:-4]

test_idx = random.sample(range(0,623), 123)
assert len(test_idx) == len(set(test_idx))

test_idx.sort()

print(test_idx)
print(x_paths[0][80:])

for idx in test_idx:
    new_x_path = master_path + '/MR2D/test/X/' + x_paths[idx][80:]
    shutil.copyfile(x_paths[idx], new_x_path)
    os.remove(x_paths[idx])

    new_y_path = master_path + '/MR2D/test/Y/' + y_paths[idx][80:]
    shutil.copyfile(y_paths[idx], new_y_path)
    os.remove(y_paths[idx])

