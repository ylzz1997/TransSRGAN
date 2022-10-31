import argparse
from statistics import mode
from WGAN.generator import Generator
from WGAN.discriminator import Discriminator
from data_loader import AnimeFaceDataset
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
import utils
import random
import os
import torchvision.utils as vutils
import logging
import time
import numpy as np

__DEBUG__ = True

# have GPU or not.
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet-GAN")
parser.add_argument('--avatar_tag_dat_path', type=str, default='../../resource/avatar_with_tag.dat', help='avatar with tag\'s list path')

parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.99, help='adam optimizer\'s paramenter')
parser.add_argument('--batch_size', type=int, default=500, help='training batch size for each epoch')
parser.add_argument('--lr_update_cycle', type=int, default=50000, help='cycle of updating learning rate')
parser.add_argument('--max_epoch', type=int, default=400, help='training epoch')
parser.add_argument('--num_workers', type=int, default=0, help='number of data loader processors')
parser.add_argument('--noise_size', type=int, default=128, help='number of G\'s input')
parser.add_argument('--lambda_adv', type=float, default=1.0, help='adv\'s lambda')
parser.add_argument('--lambda_gp', type=float, default=0.5, help='gp\'s lambda')
parser.add_argument('--model_dump_path', type=str, default='../../resource/Wgpgan_models', help='model\'s save path')
parser.add_argument('--verbose', type=bool, default=True, help='output verbose messages')
parser.add_argument('--tmp_path', type=str, default='../../resource/Wgp_training_temp/', help='path of the intermediate files during training')
parser.add_argument('--verbose_T', type=int, default=100, help='steps for saving intermediate file')
parser.add_argument('--logfile', type=str, default='../../resource/wgptraining.log', help='logging path')
parser.add_argument('--gradclip', type=int, default=1, help='Gradient Clipping')

##########################################
# Load params
#
opt = parser.parse_args()
avatar_tag_dat_path = opt.avatar_tag_dat_path
learning_rate = opt.learning_rate
beta_1 = opt.beta_1
batch_size= opt.batch_size
lr_update_cycle = opt.lr_update_cycle
max_epoch = opt.max_epoch
num_workers= opt.num_workers
noise_size = opt.noise_size
lambda_adv = opt.lambda_adv
lambda_gp = opt.lambda_gp
model_dump_path = opt.model_dump_path
verbose = opt.verbose
tmp_path= opt.tmp_path
verbose_T = opt.verbose_T
logfile = opt.logfile
gradclip = opt.gradclip

logger = logging.getLogger()
logger.setLevel(logging.INFO)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log = logging.FileHandler(logfile, mode='w+')
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
log.setFormatter(formatter)

plog = logging.StreamHandler()
plog.setLevel(logging.INFO)
plog.setFormatter(formatter)

logger.addHandler(log)
logger.addHandler(plog)

logger.info('Currently D use {} for calculating'.format(device))
logger.info('Currently G use {} for calculating'.format(device))
#
#
##########################################

def initital_network_weights(element):
  if hasattr(element, 'weight'):
    element.weight.data.normal_(.0, .02)


def adjust_learning_rate(optimizer, iteration):
  lr = learning_rate * (0.1 ** (iteration // lr_update_cycle))
  return lr


class SRGAN():
  def __init__(self):
    logger.info('Set Data Loader')
    self.dataset = AnimeFaceDataset(avatar_tag_dat_path=avatar_tag_dat_path,
                                    transform=transforms.Compose([ToTensor()]))
    self.data_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers, drop_last=True)
    checkpoint, checkpoint_name = self.load_checkpoint(model_dump_path)
    if checkpoint == None:
      logger.info('Don\'t have pre-trained model. Ignore loading model process.')
      logger.info('Set Generator and Discriminator')
      self.G = Generator(128,64,3).to(device)
      self.D = Discriminator(3,64).to(device)
      logger.info('Initialize Weights')
      self.G.apply(initital_network_weights).to(device)
      self.D.apply(initital_network_weights).to(device)
      logger.info('Set Optimizers')
      self.optimizer_G = torch.optim.RMSprop(self.G.parameters(), lr=learning_rate)
      self.optimizer_D = torch.optim.RMSprop(self.D.parameters(), lr=learning_rate)
      self.epoch = 0
    else:
      logger.info('Load Generator and Discriminator')
      self.G = Generator(128,64,3).to(device)
      self.D = Discriminator(3,64).to(device)
      logger.info('Load Pre-Trained Weights From Checkpoint'.format(checkpoint_name))
      self.G.load_state_dict(checkpoint['G'])
      self.D.load_state_dict(checkpoint['D'])
      logger.info('Load Optimizers')
      self.optimizer_G = torch.optim.RMSprop(self.G.parameters(), lr=learning_rate)
      self.optimizer_D = torch.optim.RMSprop(self.D.parameters(), lr=learning_rate)
      self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
      self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
      self.epoch = checkpoint['epoch']
    logger.info('Set Criterion')


  def load_checkpoint(self, model_dir):
    models_path = utils.read_newest_model(model_dir)
    if len(models_path) == 0:
      return None, None
    models_path.sort()
    new_model_path = os.path.join(model_dump_path, models_path[-1])
    if torch.cuda.is_available():
      checkpoint = torch.load(new_model_path)
    else:
      checkpoint = torch.load(new_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    return checkpoint, new_model_path


  def train(self):
    iteration = -1
    logger.info('Currently batch size is {}'.format(batch_size))
    logging.info('Current epoch: {}. Max epoch: {}.'.format(self.epoch, max_epoch))
    while self.epoch <= max_epoch:
      msg = {}
      # adjust_learning_rate(self.optimizer_G, iteration)
      # adjust_learning_rate(self.optimizer_D, iteration)
      for i, (avatar_tag, avatar_img) in enumerate(self.data_loader):
        iteration += 1
        if avatar_img.shape[0] != batch_size:
          logging.warn('Batch size not satisfied. Ignoring.')
          continue
        if verbose:
          if iteration % verbose_T == 0:
            msg['epoch'] = int(self.epoch)
            msg['step'] = int(i)
            msg['iteration'] = iteration
        avatar_img = Variable(avatar_img).to(device)
        avatar_tag = Variable(torch.FloatTensor(avatar_tag)).to(device)
        # D : G = 2 : 1
        # 1. Training D
        # 1.1. use really image for discriminating
        self.D.zero_grad()
        label_p = self.D(avatar_img)
        
        # 1.2. real image's loss
        
        # 1.3. use fake image for discriminating
        g_noise,_ = utils.fake_generator(batch_size, noise_size, device)
        t = np.random.uniform(0,1)
        fake_feat = g_noise
        fake_img = self.G(fake_feat).detach()
        fake_img_ba = Variable(t*fake_img+(1-t)*avatar_img,requires_grad=True)
        fake_label_p = self.D(fake_img)
        loss_D = fake_label_p.mean()-label_p.mean()
        # 1.4. fake image's loss
        fake_label_p_ba = self.D(fake_img_ba)
        gradients = grad(outputs=fake_label_p_ba, inputs=fake_img_ba, grad_outputs=torch.ones(fake_label_p_ba.size()).to(device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].view(fake_img_ba.size(0), -1)
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty.backward()
        loss_D.backward()
        if verbose:
          if iteration % verbose_T == 0:
            msg['discriminator loss'] = float(loss_D)
        # 1.6. update optimizer
        self.optimizer_D.step()

        # 2. Training G
        # 2.1. generate fake image
        self.G.zero_grad()
        g_noise, _ = utils.fake_generator(batch_size, noise_size, device)
        fake_feat = g_noise
        fake_img = self.G(fake_feat)
        fake_label_p = self.D(fake_img)

        # 2.2. calc loss
        loss_g = -fake_label_p.mean()
        loss_g.backward()

        if verbose:
          if iteration % verbose_T == 0:
            msg['generator loss'] = float(loss_g)
        # 2.2. update optimizer
        self.optimizer_G.step()

        if verbose:
          if iteration % verbose_T == 0:
            logger.info('------------------------------------------')
            for key in msg.keys():
              logger.info('{} : {}'.format(key, msg[key]))
        # save intermediate file
        if iteration % verbose_T == 0:
          vutils.save_image(avatar_img.data.view(batch_size, 3, avatar_img.size(2), avatar_img.size(3)),
                            os.path.join(tmp_path, 'real_image_{}.png'.format(str(iteration).zfill(8))))
          g_noise, _ = utils.fake_generator(batch_size, noise_size, device)
          fake_feat = g_noise
          fake_img = self.G(fake_feat)
          vutils.save_image(fake_img.data.view(batch_size, 3, avatar_img.size(2), avatar_img.size(3)),
                            os.path.join(tmp_path, 'fake_image_{}.png'.format(str(iteration).zfill(8))))
          logger.info('Saved intermediate file in {}'.format(os.path.join(tmp_path, 'fake_image_{}.png'.format(str(iteration).zfill(8)))))
      # dump checkpoint
      if self.epoch%20 == 0:
        torch.save({
          'epoch': self.epoch,
          'D': self.D.state_dict(),
          'G': self.G.state_dict(),
          'optimizer_D': self.optimizer_D.state_dict(),
          'optimizer_G': self.optimizer_G.state_dict(),
        }, '{}/checkpoint_{}.tar'.format(model_dump_path, str(self.epoch).zfill(4)))
        logger.info('Checkpoint saved in: {}'.format('{}/checkpoint_{}.tar'.format(model_dump_path, str(self.epoch).zfill(4))))
      self.epoch += 1


if __name__ == '__main__':
  if not os.path.exists(model_dump_path):
    os.mkdir(model_dump_path)
  if not os.path.exists(tmp_path):
    os.mkdir(tmp_path)
  gan = SRGAN()
  gan.train()