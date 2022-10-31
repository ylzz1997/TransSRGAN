import argparse
import torch
import numpy as np
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


parser = argparse.ArgumentParser(description="")
parser.add_argument('--tmp_path', type=str, default='../../resource/test3/', help='path of the intermediate files during training')
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


def generate(G, file_name, noise,num,path_epoch):
  '''
  Generate fake image.
  :param G:
  :param file_name:
  :param tags:
  :return: img's tensor and file path.
  '''
  # g_noise = Variable(torch.FloatTensor(1, 128)).to(device).data.normal_(.0, 1)
  # g_tag = Variable(torch.FloatTensor([utils.get_one_hot(tags)])).to(device)
  

  img = G(noise)
  
  vutils.save_image(img.data.view(num, 3, 128, 128),
                    os.path.join(tmp_path,'{}.png'.format(file_name)))
  #print('Saved file in {}'.format(os.path.join(tmp_path, '{}.png'.format(file_name))))
  return img.data.view(num, 3, 128, 128), os.path.join(tmp_path,'{}.png'.format(file_name))

def main():
  G = Generator().to(device)
  G.eval()
  g_noise1, g_tag1 = utils.fake_generator(img_num, 128, device)
  g_noise2, g_tag2 = utils.fake_generator(img_num, 128, device)
  for j in np.arange(0.0,1.1,0.1):
    checkpoint,_ = load_checkpoint(model_dump_path,401)
    g_noise = j*g_noise1 + (1-j)*g_noise2
    g_tag = j*g_tag1 + (1-j)*g_tag2
    G.load_state_dict(checkpoint['G'])
    if not os.path.exists(tmp_path):
      os.mkdir(tmp_path)
    print(torch.cat([g_noise, g_tag], dim=1))
    for i in range(all_num):
      generate(G, '{}'.format(round(j,1)),torch.cat([g_noise, g_tag], dim=1),img_num,j)


if __name__ == '__main__':
  main()
