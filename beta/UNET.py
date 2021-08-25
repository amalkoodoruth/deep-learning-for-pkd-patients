import time
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.optim as optim
from train import train_fn
from utils import (
    get_loaders,
    load_checkpoint,
    check_accuracy
)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), ##[(W−K+2P)/S]+1 = W, solve for P
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module): ## let's start with binary segmentation
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## down sampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        ## up sampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                ))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self,x):
        skip_connections = []

        for down in self.downs:

            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] ## reversing the list


        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)

        return self.final_conv(x)

    def fit(self, DEVICE, Train, Val, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, LOAD_MODEL):
        train_losses = []
        val_losses = []
        train_dice_scores = []
        val_dice_scores = []
        # UNet = UNET(in_channels=1, out_channels=1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        # loss_fn = myDiceLoss()
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        train_loader, val_loader = get_loaders(Train, Val, b_size=BATCH_SIZE)
        # check_accuracy(val_loader, UNet, device=DEVICE)
        scaler = torch.cuda.amp.GradScaler()

        # NUM_EPOCHS = 50

        if LOAD_MODEL:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), self)

        start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            print("Epoch: {epoch}/{total}".format(epoch=epoch + 1, total=NUM_EPOCHS))
            train_fn(train_loader, self, optimizer, loss_fn, scaler)

            # save model
            checkpoint = {
                "state_dict": self.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            # check accuracy
            train_loss, train_dice = check_accuracy(train_loader, self, loss_fn, device=DEVICE)
            train_losses.append(train_loss)
            train_dice_scores.append(train_dice)
            val_loss, val_dice = check_accuracy(val_loader, self, loss_fn, device=DEVICE)
            val_losses.append(val_loss)
            val_dice_scores.append(val_dice)
            # # print some examples to a folder
            # save_predictions_as_imgs(
            #     val_loader, model, folder="saved_images/", device=DEVICE
            # )
        total_time = time.time() - start_time


def test():
    print("----------------")
    print("Testing UNET with inputs divisible by 16")
    x0 = torch.randn((1, 1, 160, 160))
    model0 = UNET(in_channels=1, out_channels=1)
    preds0 = model0(x0)
    print("Input size: ", x0.shape)
    print("Output size: ", preds0.shape)
    if x0.shape == preds0.shape:
        print("Input and output sizes agree")


    print("----------------")
    print("Testing UNET with inputs not divisible by 16")
    x1 = torch.randn((1, 1, 161, 161))
    model1 = UNET(in_channels=1, out_channels=1)
    preds1 = model1(x1)
    print("Input size: ", x1.shape)
    print("Output size: ", preds1.shape)
    if x1.shape == preds1.shape:
        print("Input and output sizes agree")
