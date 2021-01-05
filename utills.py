# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:54:02 2020

@author: Sharjeel Masood
"""
import torch
import torch.nn as nn  

def stlu(input):

    return ((50 * torch.sigmoid(input))-25)


class StLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        
        return stlu(input) 