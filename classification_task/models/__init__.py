from .custom import *
from .resnet import *
from .resnetv2 import *

def build_model(model_name : str, num_classes : int = 10):
    if model_name == 'resnet_mini':
        model = ResNet_mini(num_classes=num_classes)
    elif model_name == 'custom':
        model = CustomNet(num_classes=num_classes)
    elif model_name == 'resnet_mini_v2':
        model = ResNet_mini_v2(num_classes=num_classes,
                               layers = [3,4,6],
                               dropout = 0,
                               k = 4)
    elif model_name == 'wide_resnet_mini_v2':
        model = ResNet_mini_v2(num_classes=num_classes,
                               layers = [2,2,2],
                               dropout = 0.3,
                               k = 8)
    return model
    