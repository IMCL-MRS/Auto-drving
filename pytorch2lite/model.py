import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import sys,os
# from PIL import Image as PILImage
import random
import matplotlib.cm as cm
import matplotlib.pyplot as plt

LATENT_CODE_NUM = 128
transform = transforms.Compose([transforms.Resize([120, 200]), transforms.ToTensor()])
MSELoss = torch.nn.MSELoss()

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc11 = nn.Linear(128 * 50 * 30, LATENT_CODE_NUM)
        self.fc12 = nn.Linear(128 * 50 * 30, LATENT_CODE_NUM)
        self.fc2 = nn.Linear(LATENT_CODE_NUM, 128 * 50 * 30)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(1, 128)).cuda()
        # eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        # eps = torch.randn(1, 128, device='cuda')
        #z = mu + eps * torch.exp(logvar / 2)
        z = torch.add(mu, eps * torch.exp(logvar / 2))
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)

        #mu = self.fc11(out1.view(out1.size(0), -1))
        #logvar = self.fc12(out2.view(out2.size(0), -1))
        mu = self.fc11(out1.view(1, -1))
        logvar = self.fc12(out2.view(1, -1))
        z = self.reparameterize(mu, logvar)
        out3 = self.fc2(z).view(1, 128, 30,50)
        return self.decoder(out3), mu, logvar

class Learner(nn.Module):
    def __init__(self):
        super(Learner, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
