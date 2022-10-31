import argparse
import torch
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
parser.add_argument('--tmp_path', type=str, default='../../resource/test/', help='path of the intermediate files during training')
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
  gt = torch.Tensor([[-0.9034,-0.9013,-0.5023,-1.8522,-1.1346,-1.3662,-0.3672,0.5358,0.6458,-0.9587,-0.4894,-0.2873,0.2440,-1.2534,1.2833,-0.0199,-0.8592,2.3169,-0.0566,0.4092,-0.4681,2.1690,1.3590,-0.6330,-1.3200,1.6463,1.2068,0.7990,-0.2561,-0.3366,-0.4298,1.0832,0.0136,0.8167,-0.6856,-0.3877,-1.5278,0.7867,-0.6837,-1.1232,-0.3435,-0.6607,-0.7262,0.4321,-1.9025,0.9178,-0.7466,0.4226,0.7969,1.6355,-0.2874,1.3733,-0.8531,-0.2865,0.9480,-0.8114,0.6774,0.3766,0.9851,-1.2143,0.6910,-0.0334,-0.3650,0.2147,2.4917,-1.0040,-2.3903,-0.0748,1.1615,0.0995,-0.8307,-1.0769,-0.7889,-0.5173,-2.4304,-1.3725,0.7929,0.4666,-0.6591,-0.0849,0.5297,-0.9494,0.3677,-0.1466,-0.9546,-1.3547,0.0513,0.7343,1.2564,-0.6602,-0.8677,-0.8618,0.4110,0.5225,0.4808,2.3342,1.0575,-0.5799,0.6096,0.8289,1.4503,-1.1547,-1.4036,-0.4941,1.4235,0.2839,1.7426,1.4166,-0.9104,-0.6318,-0.4112,0.8838,0.0284,0.0050,-0.5399,0.5794,-0.2463,0.1564,-1.7436,-0.7434,0.1509,-1.3113,-0.1920,-1.3947,0.3475,-0.6302,-1.9498,-0.0444,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,0.0000,1.0000,0.0000,1.0000,0.0000,0.0000,0.0000,1.0000,0.0000,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,1.0000,0.0000,0.0000]]).to(device)
  print(gt)
  for j in range(0,401,10):
    checkpoint,_ = load_checkpoint(model_dump_path,j)
    G.load_state_dict(checkpoint['G'])
    if not os.path.exists(tmp_path):
      os.mkdir(tmp_path)
    for i in range(all_num):
      generate(G, 'test_{}'.format(j),gt,img_num,j)


if __name__ == '__main__':
  main()
