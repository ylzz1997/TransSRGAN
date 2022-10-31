import argparse
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import os
from DCGAN.generator import Generator
import utils
from data_loader import AnimeFaceDataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description="")
parser.add_argument('--avatar_tag_dat_path', type=str, default='../../resource/avatar_with_tag.dat', help='avatar with tag\'s list path')
parser.add_argument('--tmp_path', type=str, default='../../resource/training_temp_old_DCGAN/', help='path of the intermediate files during training')
parser.add_argument('--tmp_path2', type=str, default='../../resource/training_temp_原图_DCGAN/', help='path of the intermediate files during training')
parser.add_argument('--model_dump_path', type=str, default='../../resource/dcgan_models', help='model\'s save path')
parser.add_argument('--img_num', type=int, default=1, help='img number')
parser.add_argument('--all_num', type=int, default=10000, help='all number')
parser.add_argument('--epoch', type=int, default=100, help='epoch')
opt = parser.parse_args()
tmp_path= opt.tmp_path
tmp_path2= opt.tmp_path2
model_dump_path = opt.model_dump_path
img_num = opt.img_num
all_num = opt.all_num
epoch = opt.epoch
avatar_tag_dat_path = opt.avatar_tag_dat_path 

def load_checkpoint(model_dir,epoch = -1  ):
  models_path = utils.read_newest_model(model_dir)
  if len(models_path) == 0:
    return None, None
  models_path.sort()
  new_model_path = os.path.join(model_dump_path, models_path[epoch])
  checkpoint = torch.load(new_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
  return checkpoint, new_model_path


def generate(G, file_name, tags,num,path_epoch):
  '''
  Generate fake image.
  :param G:
  :param file_name:
  :param tags:
  :return: img's tensor and file path.
  '''
  # g_noise = Variable(torch.FloatTensor(1, 128)).to(device).data.normal_(.0, 1)
  # g_tag = Variable(torch.FloatTensor([utils.get_one_hot(tags)])).to(device)
  g_noise,_ = utils.fake_generator(num, 128, device)

  img = G(g_noise)
  
  vutils.save_image(img.data.view(num, 3, 128, 128),
                    os.path.join(tmp_path,str(path_epoch),'{}.png'.format(file_name)))
  #print('Saved file in {}'.format(os.path.join(tmp_path, '{}.png'.format(file_name))))
  return img.data.view(num, 3, 128, 128), os.path.join(tmp_path,str(path_epoch), '{}.png'.format(file_name))

def main():
  G = Generator(128,64,3).to(device)
  #G.eval()
  '''
  dataset = AnimeFaceDataset(avatar_tag_dat_path=avatar_tag_dat_path,
                                    transform=transforms.Compose([ToTensor()]))
  data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=img_num,
                                                shuffle=True,
                                                num_workers=0, drop_last=True)
  if not os.path.exists(tmp_path):
      os.mkdir(tmp_path)
  if not os.path.exists(tmp_path2):
    os.mkdir(tmp_path2)
  for i in range(all_num):
    _,avatar_img = next(iter(data_loader))
    vutils.save_image(avatar_img.data.view(img_num, 3, avatar_img.size(2), avatar_img.size(3)),
                    os.path.join(tmp_path2,'test_{}.png'.format(i)))
'''

  
  for j in range(21):
    checkpoint,_ = load_checkpoint(model_dump_path,j)
    G.load_state_dict(checkpoint['G'])
    if not os.path.exists(os.path.join(tmp_path,str(j*20))):
      os.mkdir(os.path.join(tmp_path,str(j*20)))
    for i in range(all_num):
      generate(G, 'test_{}'.format(i), ['white hair'],img_num,j*20)


if __name__ == '__main__':
  main()
