import torch.nn as nn
from timm.models import create_model

class SwinTransformer(nn.Module):
    def __init__(self, config):
        super(SwinTransformer, self).__init__()
        self.model = create_model(
            config.model_name, 
            pretrained=config.pretrained, 
            num_classes=config.num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x
