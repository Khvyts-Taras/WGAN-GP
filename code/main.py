from torch import optim
import os
import torchvision.utils as vutils
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import math

# Arguments
BATCH_SIZE = 128
IMGS_TO_DISPLAY = 100
EPOCHS = 500
Z_DIM = 100
CHANNELS = 3
N_CRITIC = 5
GRADIENT_PENALTY = 10
LOAD_MODEL = False


model_path = 'model'

os.makedirs('model', exist_ok=True)
os.makedirs('samples', exist_ok=True)

# Method for storing generated images
def generate_imgs(z, epoch=0):
	generator.eval()
	fake_imgs = generator(z)
	fake_imgs_ = vutils.make_grid(fake_imgs, normalize=True, nrow=math.ceil(z.shape[0] ** 0.5))
	vutils.save_image(fake_imgs_, os.path.join('samples', 'sample_' + str(epoch) + '.png'))


# Data loaders
transform = transforms.Compose([transforms.Resize([64, 64]),
								transforms.ToTensor(),
								transforms.Normalize([0.5], [0.5])])


# Конфигурация загрузчика данных
os.makedirs("/data/Places365", exist_ok=True)
dataset = datasets.Places365("/data/Places365", split='val', small=True, download=False, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True,
										  drop_last=True)

# Fix images for viz
fixed_z = torch.randn(IMGS_TO_DISPLAY, Z_DIM)

# Labels
real_label = torch.ones(BATCH_SIZE)
fake_label = torch.zeros(BATCH_SIZE)


class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=3, conv_dim=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, conv_dim * 8, 4, 2, 0, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_dim * 8, conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_dim, channels, 4, 2, 1, bias=True),
            nn.Tanh()
        )
        self._initialize_weights()

    def forward(self, x):
        x = x.reshape([x.shape[0], -1, 1, 1])
        return self.net(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

class Critic(nn.Module):
    def __init__(self, channels=3, conv_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, conv_dim, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(conv_dim * 8, 1, 4, 1, 0, bias=True)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.net(x).squeeze()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


def gradient_penalty(real, fake):
	m = real.shape[0]
	epsilon = torch.rand(m, 1, 1, 1)
	if is_cuda:
		epsilon = epsilon.cuda()
	
	interpolated_img = epsilon * real + (1-epsilon) * fake
	interpolated_out = critic(interpolated_img)

	grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
							   grad_outputs=torch.ones(interpolated_out.shape).cuda() if is_cuda else torch.ones(interpolated_out.shape),
							   create_graph=True, retain_graph=True)[0]
	grads = grads.reshape([m, -1])
	grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
	return grad_penalty


generator = Generator(z_dim=Z_DIM, channels=CHANNELS)
critic = Critic(channels=CHANNELS)

# Load previous model   
if LOAD_MODEL:
	generator.load_state_dict(torch.load(os.path.join(model_path, 'generator.pkl')))
	critic.load_state_dict(torch.load(os.path.join(model_path, 'critic.pkl')))


# Define Optimizers
g_opt = optim.Adam(generator.parameters(), lr=0.001, betas=(0.0, 0.9), weight_decay=2e-5)
c_opt = optim.Adam(critic.parameters(), lr=0.001, betas=(0.0, 0.9), weight_decay=2e-5)

# GPU Compatibility
is_cuda = torch.cuda.is_available()
if is_cuda:
	generator, critic = generator.cuda(), critic.cuda()
	real_label, fake_label = real_label.cuda(), fake_label.cuda()
	fixed_z = fixed_z.cuda()

total_iters = 0
g_loss = d_loss = torch.Tensor([0])
max_iter = len(data_loader)

# Training
for epoch in range(EPOCHS):
	generator.train()
	critic.train()

	for i, data in enumerate(data_loader):

		total_iters += 1

		# Loading data
		x_real, _ = data
		z_fake = torch.randn(BATCH_SIZE, Z_DIM)

		if is_cuda:
			x_real = x_real.cuda()
			z_fake = z_fake.cuda()

		# Generate fake data
		x_fake = generator(z_fake)

		# Train Critic
		fake_out = critic(x_fake.detach())
		real_out = critic(x_real.detach())
		x_out = torch.cat((real_out, fake_out))
		d_loss = (real_out.mean() - fake_out.mean()) + gradient_penalty(x_real, x_fake) * GRADIENT_PENALTY + (x_out ** 2).mean() * 0.0001

		c_opt.zero_grad()
		d_loss.backward()
		c_opt.step()

		# Train Generator
		if total_iters % N_CRITIC == 0:
			z_fake = torch.randn(BATCH_SIZE, Z_DIM)
			if is_cuda:
				z_fake = z_fake.cuda()
			x_fake = generator(z_fake)

			fake_out = critic(x_fake)
			g_loss =  fake_out.mean()

			g_opt.zero_grad()
			g_loss.backward()
			g_opt.step()

		if i % 50 == 0:
			print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
				  + "\titer: " + str(i) + "/" + str(max_iter)
				  + "\ttotal_iters: " + str(total_iters)
				  + "\td_loss:" + str(round(d_loss.item(), 4))
				  + "\tg_loss:" + str(round(g_loss.item(), 4))
				  )
			

	torch.save(generator.state_dict(), os.path.join(model_path, 'generator.pkl'))
	torch.save(critic.state_dict(), os.path.join(model_path, 'critic.pkl'))
	generate_imgs(fixed_z, epoch=epoch + 1)
	

generate_imgs(fixed_z)
