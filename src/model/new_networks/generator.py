import torch.nn as nn
import torch
class selfAttention(nn.Module):
    def __init__(self,input_len_num,output_len_num):
        super(selfAttention, self).__init__()
        self.Q = nn.Linear(input_len_num,output_len_num)
        self.K = nn.Linear(input_len_num,output_len_num)
        self.V = nn.Linear(input_len_num,output_len_num)
        self.sf = nn.Softmax(dim=-1)
    def forward(self,input):
        Q = self.Q(input)
        K = self.K(input)
        V = self.V(input)
        a = torch.matmul(Q,K.transpose(-2,-1))/(len(K)**1/2)
        a = self.sf(a)
        Z = torch.matmul(a,V)
        return Z

class FF(nn.Module):
    def __init__(self,input_len_num):
        super(FF, self).__init__()
        self.L = nn.Sequential(
                    nn.Linear(input_len_num,input_len_num*2),
                    nn.LeakyReLU(),
                    nn.Linear(input_len_num*2,input_len_num)
        )
    def forward(self,input):
        input = self.L(input)
        return input

class AN(nn.Module):
    def __init__(self,input_len_num):
        super(AN, self).__init__()
        
    def forward(self,input,res_input):
        input = input + res_input
        return input

class multiSA(nn.Module): 
    def __init__(self,input_len_num,output_len_num,sa_num):
        super(multiSA, self).__init__()
        self.ma = nn.ModuleList([selfAttention(input_len_num,output_len_num) for i in range(sa_num)])
        self.n = sa_num
        self.ln = nn.Linear(output_len_num*sa_num,input_len_num)
    def forward(self,input):
        Z=None
        for i in self.ma:
            if Z is None:Z = i(input)
            else:
               Z = torch.cat((Z,i(input)),-1)
        Z = self.ln(Z)
        return Z


class Encode(nn.Module):
    def __init__(self,input_len_num,mutiplySA = 4):
        super(Encode, self).__init__()
        self.ad = AN(input_len_num)
        self.FF = FF(input_len_num)
        self.ms = multiSA(input_len_num,input_len_num,mutiplySA)
        self.ln = nn.LayerNorm(input_len_num)
    def forward(self,input):
        res_input = input
        input = self.ln(input)
        input = self.ms(input)
        input = self.ad(input,res_input)
        res_input = input
        input = self.ln(input)
        input = self.FF(input)
        input = self.ad(input,res_input)
        return input

class _SubpixelBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bias=False, upscale_factor=2):
    super(_SubpixelBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels,
                 kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    self.bn = nn.BatchNorm2d(in_channels)
    self.relu = nn.PReLU()

  def forward(self, tensor):
    output = self.conv(tensor)
    output = self.pixel_shuffle(output)
    output = self.bn(output)
    output = self.relu(output)
    return output


class Generator(nn.Module):
  def __init__(self, tag=34):
    super(Generator, self).__init__()
    in_channels = 128 + tag
    self.dense_1 = nn.Linear(in_channels, 64*16*16)
    self.encoders = self.make_residual_layer(16)
    self.BN1 = nn.BatchNorm2d(64)
    self.subpixel_layer = self.make_subpixel_layer(3)
    self.conv_1 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=True)
    self.relu = nn.PReLU()
    self.tanh_1 = nn.Tanh()
  def forward(self, tensor):
    output = self.dense_1(tensor)
    output = output.view(-1,64,16,16)
    res_net = output
    output = self.nv(output,self.BN1)
    output = output.view(-1,64,16*16)
    output = self.encoders(output)
    output = output.view(-1,64,16,16)
    output = self.nv(output,self.BN1)
    output = res_net+output
    output = self.subpixel_layer(output)
    output = self.conv_1(output)
    output = self.tanh_1(output)
    return output
  def nv(self,input,BN):
    output = BN(input)
    output = self.relu(output)
    return output
  def make_subpixel_layer(self, block_size=3):
    layers = []
    for _ in range(block_size):
      layers.append(_SubpixelBlock(64, 256, 3, 1))
    return nn.Sequential(*layers)
  def make_residual_layer(self, block_size=16):
    layers = []
    for _ in range(block_size):
      layers.append(Encode(16*16,mutiplySA = 16))
    return nn.Sequential(*layers)

if __name__ == '__main__':
  from torch.autograd import Variable
  import torch

  gen = Generator()
  x = Variable(torch.rand((8,128+34)), requires_grad=True)
  print(gen(x).shape)

