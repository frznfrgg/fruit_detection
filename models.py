import torch
import torch.nn as nn
import numpy as np

class Spine(nn.Module):
    def __init__(self, vgg):
        super(Spine, self).__init__()

        for i, child in enumerate(vgg.children()):
            for param in child.parameters():
                param.requires_grad = False

            if i == 0:
                layers_list = child

        self.conv = nn.Sequential(*layers_list[:-1])

    def forward(self, x):

        x = self.conv(x)

        return x


DETECTION_CLASSES = 6
# !!!! класс с индексом 0 - задний фон !!!!
# Классификатор
class СlassifierHead(nn.Module):
    def __init__(self):
        super(СlassifierHead, self).__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features = 8192, out_features = 2048, bias=True),
            nn.BatchNorm1d(2048, momentum = 0.8),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 2048, out_features = 2048, bias=True),
            nn.BatchNorm1d(2048, momentum = 0.8),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 2048, out_features = 1024, bias=True),
            nn.BatchNorm1d(1024, momentum = 0.8),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 1024, out_features = 512, bias=True),
            nn.BatchNorm1d(512, momentum = 0.8),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 512, out_features = DETECTION_CLASSES + 1, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)

        return x

class ClsTrainer(СlassifierHead):
    def __init__(self,
                lr,
                gamma,
                criterion):
        super(ClsTrainer, self).__init__()

        self.optim = torch.optim.Adam(super().parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=gamma)
        self.criterion = criterion
        self.epo_train_losses = []
        self.epo_val_losses = []

        self.train_losses = []
        self.val_losses = []

    def train_step(self, batch, target):
        self.train()
        self.optim.zero_grad()

        output = self.forward(batch)
        loss = self.criterion(output, target)

        loss.backward()
        self.optim.step()

        self.epo_train_losses.append(loss.item())

    def eval_step(self, batch, target):
        output = self.query(batch)
        loss = self.criterion(output, target)
        self.epo_val_losses.append(loss.item())

    def query(self, batch):
        self.eval()
        with torch.no_grad():
            output = self.forward(batch)

        return output

    def epoch_end(self):
        self.scheduler.step()

        self.train_losses.append(np.mean(self.epo_train_losses))
        self.val_losses.append(np.mean(self.epo_val_losses))

        self.epo_train_losses = []
        self.epo_val_losses = []


# Регрессор
class RegressorHead(nn.Module):
    def __init__(self):
        super(RegressorHead, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(in_features = 8192, out_features = 1024, bias=True),
            nn.BatchNorm1d(1024, momentum = 0.8),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = 1024+4, out_features = 512, bias=True),
            nn.BatchNorm1d(512, momentum = 0.8),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features = 512, out_features = 256, bias=True),
            nn.BatchNorm1d(256, momentum = 0.8),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features = 256, out_features = 4, bias=True)
        )

    def forward(self, sample, P):
        x = self.flatten(sample)
        x = self.fc1(x)
        x = torch.cat((x, P), dim=-1)
        x = self.fc2(x)

        return x


class RgsTrainer(RegressorHead):
    def __init__(self,
                lr,
                gamma,
                criterion):
        super(RgsTrainer, self).__init__()

        self.optim = torch.optim.Adam(super().parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=gamma)
        self.criterion = criterion
        self.epo_train_losses = []
        self.epo_val_losses = []

        self.train_losses = []
        self.val_losses = []

    def train_step(self, batch, target):
        self.train()
        self.optim.zero_grad()

        input1, input2 = batch
        output = self.forward(input1, input2)
        loss = self.criterion(output, target)

        loss.backward()
        self.optim.step()

        self.epo_train_losses.append(loss.item())

    def eval_step(self, batch, target):
        output = self.query(batch)
        loss = self.criterion(output, target)
        self.epo_val_losses.append(loss.item())

    def query(self, batch):
        self.eval()
        with torch.no_grad():
            input1, input2 = batch
            output = self.forward(input1, input2)

        return output

    def epoch_end(self):
        self.scheduler.step()

        self.train_losses.append(np.mean(self.epo_train_losses))
        self.val_losses.append(np.mean(self.epo_val_losses))

        self.epo_train_losses = []
        self.epo_val_losses = []


# Сегментатор
class SegmentHead(nn.Module):
    def __init__(self, in_channels=512):
        super(SegmentHead, self).__init__()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()


        self.conv0 = self.set_conv_block(in_channels, 256)

        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.BN1 = nn.BatchNorm2d(128, momentum = 0.8)

        self.conv2 = self.set_conv_block(128, 64)

        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.BN3 = nn.BatchNorm2d(32, momentum = 0.8)

        self.conv4 = nn.Conv2d(32, DETECTION_CLASSES, kernel_size = 4, padding=0)



    def forward(self, x):
        x = self.conv0(x)

        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu(x)

        x = self.conv2(x)

        x = self.conv3(x)
        x = self.BN3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.sigmoid(x)

        return x

    @staticmethod
    def set_conv_block(input_size, output_size, kernel_size = 3, padding = 1):
        block = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size, padding=padding),
            nn.BatchNorm2d(output_size, momentum=0.8),
            nn.LeakyReLU(),
            nn.Conv2d(output_size, output_size, kernel_size, padding=padding),
            nn.BatchNorm2d(output_size, momentum=0.8),
            nn.LeakyReLU()
        )
        return block


class SgmTrainer(SegmentHead):
    def __init__(self,
                lr,
                gamma,
                criterion,
                in_channels = 512):
        super(SgmTrainer, self).__init__(in_channels)

        self.optim = torch.optim.Adam(super().parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=gamma)
        self.criterion = criterion
        self.epo_train_losses = []
        self.epo_val_losses = []

        self.train_losses = []
        self.val_losses = []

    def train_step(self, batch, target):
        self.train()
        self.optim.zero_grad()

        output = self.forward(batch)
        loss = self.criterion(output, target)

        loss.backward()
        self.optim.step()

        self.epo_train_losses.append(loss.item())

    def eval_step(self, batch, target):
        output = self.query(batch)
        loss = self.criterion(output, target)
        self.epo_val_losses.append(loss.item())

    def query(self, batch):
        self.eval()
        with torch.no_grad():
            output = self.forward(batch)

        return output

    def epoch_end(self):
        self.scheduler.step()

        self.train_losses.append(np.mean(self.epo_train_losses))
        self.val_losses.append(np.mean(self.epo_val_losses))

        self.epo_train_losses = []
        self.epo_val_losses = []
