import argparse
import torch
import numpy as np
import cv2
from  matplotlib import pyplot as plt
torch.set_printoptions(profile="full")
from torch.autograd import Variable
import torchvision.utils as vutils
import os
from new_networks.generator import Generator
import utils
from data_loader import AnimeFaceDataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tag = ['blonde hair','brown hair','black hair','blue hair','pink hair',
               'purple hair','green hair','red hair','silver hair','white hair','orange hair',
               'aqua hair','gray hair','long hair','short hair','twintails','drill hair','ponytail','blush',
               'smile','open mouth','hat','ribbon','glasses','blue eyes','red eyes','brown eyes',
               'green eyes','purple eyes','yellow eyes','pink eyes','aqua eyes','black eyes','orange eyes',]

parser = argparse.ArgumentParser(description="")
parser.add_argument('--tmp_path', type=str, default='../../resource/test5/', help='path of the intermediate files during training')
parser.add_argument('--model_dump_path', type=str, default='../../resource/gan_models2', help='model\'s save path')
parser.add_argument('--img_num', type=int, default=1, help='img number')
parser.add_argument('--all_num', type=int, default=1, help='all number')
parser.add_argument('--epoch', type=int, default=100, help='epoch')
opt = parser.parse_args()
tmp_path= opt.tmp_path
model_dump_path = opt.model_dump_path
img_num = opt.img_num
all_num = opt.all_num
epoch = opt.epoch

def load_checkpoint(model_dir,epoch = -1  ):
  models_path = utils.read_newest_model(model_dir)
  if len(models_path) == 0:
    return None, None
  models_path.sort()
  new_model_path = os.path.join(model_dump_path, models_path[epoch])
  checkpoint = torch.load(new_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
  return checkpoint, new_model_path


def generate(G, file_name, noise,num,tmp_path):
  img = G(noise)
  img_save = img.clone()
  img = img.sum(1)
  img = img.squeeze()
  grads = torch.zeros(*img.shape,*noise.squeeze().shape)
  print(grads.shape)
  for i in range(128):
    for j in range(128):
      img[i][j].backward(retain_graph=True)
      grads[i][j] = torch.abs(noise.grad.squeeze()*noise.squeeze())
      noise.grad.zero_()
  grads = torch.nn.functional.relu(grads).detach()
  img_np = np.uint8(img_save.squeeze().permute(1,2,0).cpu().detach() * 255)
  for i in range(grads.shape[-1]):
    grad = np.array(grads[:,:,i])
    grad = np.maximum(grad, 0)
    gmax = np.max(grad)
    if gmax == 0 : continue
    grad /= gmax
    grad = np.uint8(255 * grad)
    grad = cv2.GaussianBlur(grad,(5,5),2,2)
    cv2.normalize(grad,grad,0,255,cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    cam_img = 0.8 * heatmap + 0.2 * img_np
    if i >=128: i = tag[i-128]
    path_cam_img = os.path.join(tmp_path,'hotmap_{}.png'.format(i))
    cv2.imwrite(path_cam_img, cam_img)
  vutils.save_image(img_save.data.view(num, 3, 128, 128),
                    os.path.join(tmp_path,'{}.png'.format(file_name)))
  return img_save.data.view(num, 3, 128, 128), os.path.join(tmp_path,'{}.png'.format(file_name))

def main():
  G = Generator().to(device)
  G.eval()
  checkpoint,_ = load_checkpoint(model_dump_path,-1)
  G.requires_grad_(False)
  G.load_state_dict(checkpoint['G'])
  if not os.path.exists(tmp_path):
    os.mkdir(tmp_path)
  for i in range(30):
    g_noise, g_tag = utils.fake_generator(img_num, 128, device)
    g_input = torch.cat([g_noise, g_tag], dim=1)
    g_input.requires_grad_(True)
    new_path=os.path.join(tmp_path,str(i))
    if not os.path.exists(new_path):
      os.mkdir(new_path)
    generate(G, 'test_{}'.format('hot图'),g_input,img_num,new_path)

if __name__ == '__main__':
  main()
