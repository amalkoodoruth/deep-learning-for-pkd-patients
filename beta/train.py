import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.optim as optim
import os
import numpy as np
import csv
from UNET import UNET
from dataset import SliceDataset
import torch
import math
import time
import torch.nn as nn
#from DiceLoss import myDiceLoss
from transform import transforms
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_results,
    findOrgan,
    load_seg,
    load_scan
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False
ORGAN = 'lv'
FEATURE = [64, 128, 256, 512,1024]
FEATURES = [FEATURE,[int(elem*1.2) for elem in FEATURE],
            [int(elem*1.4) for elem in FEATURE],[int(elem*1.6) for elem in FEATURE],
            [int(elem*1.8) for elem in FEATURE], [int(elem*2) for elem in FEATURE]]
#LEARNING_RATES = [lr for lr in range(1,100,5)]
LEARNING_RATES = [1e-6]#[lr/10000000 for lr in LEARNING_RATES]
# LEARNING_RATE = 5e-6
TRANSFORMS = transforms(0.5, 0.5)
BATCH_SIZE = 12

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

    master_dir = os.getcwd()
    my_dir = master_dir + '/MR2D/Train'

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
    Train = SliceDataset(train_img_paths, train_seg_paths, organ=ORGAN, transform=TRANSFORMS)
    Val = SliceDataset(val_img_paths, val_seg_paths, organ=ORGAN, transform=TRANSFORMS)

    df = pd.read_csv('mresults.csv')
    if df.empty:

        HEADERS = ['Organ','Learning rate','Epochs','Batch size',
                'Num layers', 'Features list',
                'Transforms', 'Runtime', 'Optimizer', 'Loss function',
                'Training Loss list', 'Validation Loss list',
                'Training Dice list',
                'Validation Dice list', 'Test Dice',
                'Test results']
        with open('mresults.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(HEADERS)

    test_img_paths = []
    test_seg_paths = []

    test_dir = master_dir + '/MR2D/Test'
    for dcm in os.listdir(test_dir + '/X'):
        if dcm != ".DS_Store":
            test_img_paths.append(test_dir + '/X/' + dcm)
    test_img_paths.sort()

    for seg in os.listdir(test_dir + '/Y'):
        if seg != ".DS_Store":
            test_seg_paths.append(test_dir + '/Y/' + seg)
    test_seg_paths.sort()

    ## apply find organ on everything. then remove the all 0s
    mylist = []
    for idx in range(len(test_img_paths)):
        img = load_scan(test_img_paths[idx])
        seg = load_seg(test_seg_paths[idx])
        bin_img, bin_seg = findOrgan(img, seg, ORGAN)
        if np.amax(bin_seg) != 0:
            mylist.append(idx)

    cleaned_test_img_paths = []
    cleaned_test_seg_paths = []
    for idx in mylist:
        cleaned_test_img_paths.append(test_img_paths[idx])
        cleaned_test_seg_paths.append(test_seg_paths[idx])

    Total_Test = SliceDataset(cleaned_test_img_paths, cleaned_test_seg_paths, organ=ORGAN, transform=None, test=True)

    ## TOTAL
    test_loader = DataLoader(Total_Test, 1)
    for feature in FEATURES:
        for LEARNING_RATE in LEARNING_RATES:
            train_losses = []
            val_losses = []
            train_dice_scores = []
            val_dice_scores = []
            UNet = UNET(in_channels=1, out_channels=1).to(DEVICE)
            loss_fn = nn.BCEWithLogitsLoss()
            # loss_fn = myDiceLoss()
            optimizer = optim.Adam(UNet.parameters(), lr=LEARNING_RATE)
            train_loader, val_loader = get_loaders(Train, Val, b_size= BATCH_SIZE)
            # check_accuracy(val_loader, UNet, device=DEVICE)
            scaler = torch.cuda.amp.GradScaler()

            NUM_EPOCHS = 50

            if LOAD_MODEL:
                load_checkpoint(torch.load("my_checkpoint.pth.tar"), UNet)

            start_time = time.time()

            for epoch in range(NUM_EPOCHS):
                print("Epoch: {epoch}/{total}".format(epoch=epoch + 1, total=NUM_EPOCHS))
                train_fn(train_loader, UNet, optimizer, loss_fn, scaler)

                # save model
                checkpoint = {
                    "state_dict": UNet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

                # check accuracy
                train_loss, train_dice = check_accuracy(train_loader, UNet, loss_fn, device=DEVICE)
                train_losses.append(train_loss)
                train_dice_scores.append(train_dice)
                val_loss, val_dice = check_accuracy(val_loader, UNet, loss_fn, device=DEVICE)
                val_losses.append(val_loss)
                val_dice_scores.append(val_dice)
                # # print some examples to a folder
                # save_predictions_as_imgs(
                #     val_loader, model, folder="saved_images/", device=DEVICE
                # )
            total_time =  time.time() - start_time
            # save_checkpoint(checkpoint)

            ### ACCURACY ON TEST SET


            loop = tqdm(test_loader, position=0, leave=True)

            # preds = []
            # ground = []
            # images = []

            test_dict = {}

            for batch_idx, (data, targets) in enumerate(loop):
                data = data.unsqueeze(1).to(device=DEVICE)

                pred = torch.sigmoid(UNet(data))
                pred = (pred > 0.5).float()

                # preds.append(pred.detach().cpu().numpy()[0][0])
                # images.append(data.float().unsqueeze(1).to(device=DEVICE).detach().cpu().numpy()[0][0])
                # targets = targets.float().unsqueeze(1).to(device=DEVICE)
                # ground.append(targets.detach().cpu().numpy()[0][0])

                key = data.float().unsqueeze(1).to(device=DEVICE).detach().cpu().numpy()[0][0]

                ground = targets.float().unsqueeze(1).to(device=DEVICE).detach().cpu().numpy()[0][0]
                pred = pred.detach().cpu().numpy()[0][0]

                # test_dict[key.tobytes()] = [ground, pred]

                # ## SAMPLE
            # test_idx = random.sample(range(0, len(cleaned_test_img_paths)), 10)
            #
            # sampled_test_img_paths = []
            # sampled_test_seg_paths = []
            #
            # for idx in test_idx:
            #     sampled_test_img_paths.append(cleaned_test_img_paths[idx])
            #     sampled_test_seg_paths.append(cleaned_test_seg_paths[idx])
            # Sample_Test = SliceDataset(sampled_test_img_paths, sampled_test_seg_paths, organ=ORGAN, transform=None, test=True)
            # sample_test_loader = DataLoader(Sample_Test)
            #
            # sample_loop = tqdm(sample_test_loader, position=0, leave=True)
            #
            # sample_preds = []
            # sample_ground = []
            # sample_images = []
            #
            # for batch_idx, (data, targets) in enumerate(sample_loop):
            #     data = data.unsqueeze(1).to(device=DEVICE)
            #
            #     pred = torch.sigmoid(UNet(data))
            #     pred = (pred > 0.5).float()
            #
            #     sample_preds.append(pred.detach().cpu().numpy()[0][0])
            #
            #     targets = targets.float().unsqueeze(1).to(device=DEVICE)
            #     sample_ground.append(targets.detach().cpu().numpy()[0][0])
            #     sample_images.append(data.float().unsqueeze(1).to(device=DEVICE).detach().cpu().numpy()[0][0])
            #
            # print('-----SAMPLE TEST-----')
            # loss, sample_dice = check_accuracy(sample_test_loader, UNet, loss_fn)
            # print('----------------')

            print('-----TEST-----')
            loss, test_dice = check_accuracy(test_loader, UNet, loss_fn)
            print('----------------')


            dict = {'Organ': ORGAN, 'Learning rate': LEARNING_RATE, 'Epochs': NUM_EPOCHS, 'Batch size': BATCH_SIZE,
                    'Num layers': len(feature), 'Features list': feature,
                    'Transforms': TRANSFORMS, 'Runtime': total_time, 'Optimizer': optimizer, 'Loss function': loss_fn,
                    'Training Loss list': train_losses, 'Validation Loss list': val_losses,
                    'Training Dice list': train_dice_scores,
                    'Validation Dice list': val_dice_scores, 'Test Dice': test_dice
                    }
                    # 'Test results': test_dict}
            save_results('mresults.csv', dict)