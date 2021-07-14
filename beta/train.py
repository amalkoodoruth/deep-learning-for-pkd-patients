from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.optim as optim
import os
from UNET import UNET
from dataset import SliceDataset
import torch
import math
from DiceLoss import myDiceLoss
from transform import transforms
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False

def train_fn(train_dataloader, model, optimizer, loss_fn, scaler):
    loop = tqdm(train_dataloader, position=0, leave=True)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.unsqueeze(1).to(device=DEVICE)

        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            # print("pred: ", predictions.shape)
            loss = loss_fn.forward(predictions, targets)
            # loss = loss_fn(predictions, targets)
            # print("loss: ", loss.shape)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

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

    # Train = ds.SliceDataset(img_paths, seg_paths)
    train_idx = random.sample(range(0, len(img_paths)), math.ceil(0.75*len(img_paths)))
    train_img_paths = []
    train_seg_paths = []
    val_img_paths = []
    val_seg_paths = []
    for idx in range(len(img_paths)):
        if idx in train_idx:
            train_img_paths.append(img_paths[idx])
            train_seg_paths.append(seg_paths[idx])
        else:
            val_img_paths.append(img_paths[idx])
            val_seg_paths.append(seg_paths[idx])
    Train = SliceDataset(train_img_paths, train_seg_paths, transform=transforms(0.5, 0.5))
    Val = SliceDataset(val_img_paths, val_seg_paths, transform=transforms(0.5, 0.5))


val_losses = []
dice_scores = []
UNet = UNET(in_channels=1, out_channels=1).to(DEVICE)
# loss_fn = nn.BCEWithLogitsLoss()
loss_fn = myDiceLoss()
optimizer = optim.Adam(UNet.parameters(), lr=2 * 1e-3)
train_loader, val_loader = get_loaders(Train, Val)
# check_accuracy(val_loader, UNet, device=DEVICE)
scaler = torch.cuda.amp.GradScaler()

NUM_EPOCHS = 10

if LOAD_MODEL:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), UNet)

for epoch in range(NUM_EPOCHS):
    print("Epoch: {epoch}/{total}".format(epoch=epoch + 1, total=NUM_EPOCHS))
    train_fn(train_loader, UNet, optimizer, loss_fn, scaler)

    # save model
    checkpoint = {
        "state_dict": UNet.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    # check accuracy
    loss, dice = check_accuracy(val_loader, UNet, loss_fn, device=DEVICE)
    val_losses.append(loss)
    dice_scores.append(dice)
    # # print some examples to a folder
    # save_predictions_as_imgs(
    #     val_loader, model, folder="saved_images/", device=DEVICE
    # )
save_checkpoint(checkpoint)
