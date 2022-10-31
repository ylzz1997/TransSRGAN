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
        self.ln = nn.LayerNorm(input_len_num)
    def forward(self,input,res_input):
        input = input + res_input
        input = self.ln(input)
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
        self.ms = multiSA(input_len_num,input_len_num/2,mutiplySA)
    def forward(self,input):
        res_input = input
        input = self.ms(input)
        input = self.ad(input,res_input)    
        res_input = input
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


def make_subpixel_layer(self, block_size=3):
  layers = []
  for _ in range(block_size):
    layers.append(_SubpixelBlock(3,12,3,1))
  return nn.Sequential(*layers)

class Generator(nn.Module):
  def __init__(self, tag=34):
    super(Generator, self).__init__()
    in_channels = 128 + tag
    self.dense_1 = nn.Linear(in_channels,16*16*192)
    self.encoder_1 = Encode(16*16,mutiplySA = 2)
    self.BN1 = nn.BatchNorm2d(192)
    self.ps_1 = nn.PixelShuffle(2)
    self.encoder_2 = Encode(32*32,mutiplySA = 2)
    self.BN2 = nn.BatchNorm2d(48)
    self.ps_2 = nn.PixelShuffle(2)
    self.BN3 = nn.BatchNorm2d(12)
    self.encoder_3 = Encode(64*64,mutiplySA = 1)
    self.ps_3 = nn.PixelShuffle(2)
    self.BN4 = nn.BatchNorm2d(3)
    self.encoder_4 = Encode(128*128,mutiplySA = 1)
    self.relu = nn.PReLU()
    self.tanh_1 = nn.Tanh()
  def forward(self, tensor):
    output = self.dense_1(tensor)
    output = output.view(-1,192,16,16)
    output = self.nv(output,self.BN1)
    output = output.view(-1,192,16*16)
    output = self.encoder_1(output)
    output = output.view(-1,192,16,16)
    output = self.ps_1(output)
    output = self.nv(output,self.BN2)
    output = output.view(-1,48,32*32)
    output = self.encoder_2(output)
    output = output.view(-1,48,32,32)
    output = self.ps_2(output)
    output = self.nv(output,self.BN3)
    output = output.view(-1,12,64*64)
    output = self.encoder_3(output)
    output = output.view(-1,12,64,64)
    output = self.ps_3(output)
    output = self.nv(output,self.BN4)
    output = output.view(-1,3,128*128)
    output = self.encoder_4(output)
    output = output.view(-1,3,128,128)
    output = self.nv(output,self.BN4)
    output = self.tanh_1(output)
    return output
  def nv(self,input,BN):
    output = BN(input)
    output = self.relu(output)
    return output


if __name__ == '__main__':
  from torch.autograd import Variable
  import torch

  gen = Generator()
  x = Variable(torch.rand((8,128+34)), requires_grad=True)
  print(gen(x).shape)

