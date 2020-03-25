import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from serex50 import ResNext50, CONVERSION, load_pretrain


class Serex50_Net(nn.Module):
    def load_pretrain(self, skip=['logit.'], is_print=True):
        load_pretrain(self, skip, pretrain_file='se_resnext50_32x4d-a260b3a4.pth', conversion=CONVERSION, is_print=is_print)

    def __init__(self, num_class=(168,11,7)):
        super(Serex50_Net, self).__init__()
        e = ResNext50()
        self.rgb = e.rgb
        self.block0 = e.block0
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        e = None  #dropped

        self.logit = nn.ModuleList(
            [ nn.Linear(2048,c) for c in num_class ]
        )
    
    def forward(self, x):
        batch_size,C,H,W = x.shape
        #if (H,W) !=(128,128):
        #    x = F.interpolate(x,size=(128,128), mode='bilinear',align_corners=False)
            
        x = self.rgb(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = [l(x) for l in self.logit]
        return logit


sigmoid = torch.nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)
swish_layer = Swish_module()

def relu_fn(x):
    """ Swish activation function """
    # return x * torch.sigmoid(x)
    return swish_layer(x)


class EfficientNet_3_Encoder(nn.Module):
    
    def __init__(self):
        super(EfficientNet_3_Encoder, self).__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b3')

    def forward(self, inputs):
        x = relu_fn(self.model._bn0(self.model._conv_stem(inputs)))
        
        global_features = []

        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            # print(idx, x.shape)
            if idx in [1,4,7,17]:
                global_features.append(x)
        x = relu_fn(self.model._bn1(self.model._conv_head(x)))
        global_features.append(x)
        global_features.reverse()

        return global_features


class EfficientNet_3(nn.Module):

    def __init__(self, num_class=(168,11,7)):
        super(EfficientNet_3, self).__init__()
        self.model_encoder = EfficientNet_3_Encoder()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.logit0 = nn.Linear(1536,num_class[0])  # 1664 for dense 169
        self.logit1 = nn.Linear(1536,num_class[1])
        self.logit2 = nn.Linear(1536,num_class[2])
                
    def forward(self, x):
        batch_size,C,H,W = x.shape
        if (H,W) !=(224,224):
            x = F.interpolate(x,size=(224,224), mode='bilinear',align_corners=False)

        global_features = self.model_encoder(x)        
        cls_feature = global_features[0]
        cls_feature = self.avgpool(cls_feature)
        cls_feature = cls_feature.view(cls_feature.size(0), -1)
        
        logit0 = self.logit0(cls_feature)
        logit1 = self.logit1(cls_feature)
        logit2 = self.logit2(cls_feature)

        logit = [logit0, logit1, logit2]
        return logit


