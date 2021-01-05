# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 06:50:59 2020

@author: Sharjeel Masood
"""

import torch.nn as nn

class Main_NN(nn.Module):
    def __init__(self):
        super(Main_NN, self).__init__()
        self.feature_extraction = nn.Sequential(
                nn.Conv2d(3, 96, 5),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.Conv2d(96, 256, 5),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 384, 5, stride = 2, padding = 1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.Conv2d(384, 384, 5),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.Conv2d(384, 384, 5, stride = 2, padding = 1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.Conv2d(384, 128, 5, stride = 2, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU()

                )
        self.linear_layers = nn.Sequential(
                nn.Linear(128*7*7, 750),
                nn.BatchNorm1d(750), # batch normalization
                nn.ReLU(),
                nn.Linear(750, 180),
                nn.BatchNorm1d(180), # batch normalization
                nn.ReLU(),
                nn.Linear(180, 60),
                nn.BatchNorm1d(60),
                nn.ReLU(),
                nn.Linear(60, 1),
                )

    def forward(self, image):
        x = self.feature_extraction(image)
        x = x.reshape(-1, 256*10*10)  # flattening
        x = self.linear_layers(x)
        return x