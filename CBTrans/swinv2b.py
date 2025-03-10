import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
MAX_LENGTH = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self,encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        swin = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
        modules = list(swin.children())[:-3]
        self.swin = nn.Sequential(*modules)

        # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        #print(self.swin)
        out = self.swin(images)
        #print(out.shape)
        #out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        #print(out.shape)
        return out

    def fine_tune(self, fine_tune=True):
        for child in self.swin.children():
            for param in child.parameters():
                param.requires_grad = fine_tune
        #print("fine_tune has been called with", fine_tune)

