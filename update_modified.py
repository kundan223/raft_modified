import torch
import torch.nn as nn
import torch.nn.functional as F
from .quaternion_layers import *
from .ReLu import Relu



class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        
        self.conv1 = QuaternionConv(input_dim, hidden_dim, 3, padding=1 , stride=1)
        self.conv2 = QuaternionConv(hidden_dim, 4, 3, padding=1 , stride=1)
        self.relu = Relu()

    def forward(self, x):
        print(x.shape)
        y = self.conv1(x)
        print("after passing x from conv1 of Flowhead" , y.shape)
        y = self.relu(y)
        print("after passing through the relu of Flwohead" , y.shape)
        y = self.conv2(y)
        print(" after passing through the conv2 of flowhead" , y.shape)
        return y
    
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = QuaternionConv(hidden_dim+input_dim, hidden_dim, 3, padding=1 ,stride =1)
        self.convr = QuaternionConv(hidden_dim+input_dim, hidden_dim, 3, padding=1 ,stride =1)
        self.convq = QuaternionConv(hidden_dim+input_dim, hidden_dim, 3, padding=1 ,stride =1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h
    

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = QuaternionConv(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2),stride =1)
        self.convr1 = QuaternionConv(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2),stride =1)
        self.convq1 = QuaternionConv(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2) ,stride =1)

        self.convz2 = QuaternionConv(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0),stride =1)
        self.convr2 = QuaternionConv(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0) ,stride =1)
        self.convq2 = QuaternionConv(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0) ,stride =1)


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h
    

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = QuaternionConv(cor_planes, 96, 1, padding=0,stride =1)
        self.convf1 = QuaternionConv(4, 64, 7, padding=3,stride =1)
        self.convf2 = QuaternionConv(64, 32, 3, padding=1,stride =1)
        self.conv = QuaternionConv(128, 80, 3, padding=1,stride =1)

    def forward(self, flow, corr):
        cor = Relu(self.convc1(corr))
        flo = Relu(self.convf1(flow))
        flo = Relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = Relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)



class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = QuaternionConv(cor_planes, 256, 1, padding=0 ,stride =1)
        self.convc2 = QuaternionConv(256, 192, 3, padding=1,stride =1)
        self.convf1 = QuaternionConv(4, 128, 7, padding=3,stride =1)
        self.convf2 = QuaternionConv(128, 64, 3, padding=1,stride =1)
        self.conv = QuaternionConv(64+192, 128-4, 3, padding=1,stride =1)
        self.relu = Relu()  

    def forward(self, flow, corr):
        print("Shape of corr before convc1:", corr.shape)
        cor = self.relu(self.convc1(corr))
        print("Shape of corr before convc1:", cor.shape)
        cor = self.relu(self.convc2(cor))
        print("Shape of corr before convc1:", cor.shape)
        print ("shape  of flow befire relu and conv:" , flow.shape)
        flo = self.relu(self.convf1(flow))
        flo = self.relu(self.convf2(flo))
        print("shape of floe after relu and conv",flo.shape)

        cor_flo = torch.cat([cor, flo], dim=1)
        print("cor_flo shape" , cor_flo.shape)
        out = self.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)
    

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow
    

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            Relu(),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        print("net after passing through gru ", net.shape)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

