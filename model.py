from jittor.models import Resnext101_32x8d
import jittor.nn as nn 

class Net(nn.Module):
    def __init__(self, classes):
        self.base_net = Resnext101_32x8d(pretrained=True,num_classes=classes)
        # self.base_net = Resnext101_32x8d(pretrained=True)
        # original = self.base_net.fc.in_features
        # if classes != original:
        #    self.base_net.fc = nn.Linear(original,classes) 
    def execute(self, x):
        x = self.base_net(x)
        return x 
