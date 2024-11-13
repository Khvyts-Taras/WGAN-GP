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


BATCH_SIZE = 128
IMGS_TO_DISPLAY = 100
EPOCHS = 50
Z_DIM = 100
N_CRITIC = 5
GRADIENT_PENALTY = 100
LOAD_MODEL = False


model_path = 'models'
samples_path = 'samples'
os.makedirs(model_path, exist_ok=True)
os.makedirs(samples_path, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gradient_penalty(real, fake):
	epsilon = torch.rand(real.shape[0], 1, 1, 1).to(device)
	
	interpolated_img = epsilon * real + (1-epsilon) * fake
	interpolated_out = critic(interpolated_img)

	grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
						  grad_outputs=torch.ones(interpolated_out.shape).to(device),
						  create_graph=True, retain_graph=True)[0]

	grads = grads.reshape([real.shape[0], -1])
	grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
	return grad_penalty


def generate_imgs(z, epoch=0):
	generator.eval()
	fake_imgs = generator(z)
	fake_imgs_ = vutils.make_grid(fake_imgs, normalize=True, nrow=math.ceil(z.shape[0] ** 0.5))
	vutils.save_image(fake_imgs_, os.path.join(samples_path, 'sample_' + str(epoch) + '.png'))


class Gen(nn.Module):
    def __init__(self, z_dim=100, conv_dim=32):
        super(Gen, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, conv_dim*8, 4, 2, 0, bias=False),
            nn.InstanceNorm2d(conv_dim * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(conv_dim*8, conv_dim*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim*4),
            nn.ReLU(),

            nn.ConvTranspose2d(conv_dim*4, conv_dim*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim*2),
            nn.ReLU(),

            nn.ConvTranspose2d(conv_dim*2, conv_dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(),

            nn.ConvTranspose2d(conv_dim, 3, 4, 2, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.reshape([x.shape[0], -1, 1, 1])
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, conv_dim=32):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, conv_dim, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(conv_dim, conv_dim*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(conv_dim*2, conv_dim*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(conv_dim*4, conv_dim*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim*8),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(conv_dim*8, 1, 4, 1, 0, bias=True)
        )

    def forward(self, x):
        return self.net(x).squeeze()




transform = transforms.Compose([transforms.Resize([64, 64]),
								transforms.ToTensor(),
								transforms.Normalize([0.5], [0.5])])
dataset = datasets.CelebA(root='/data/CelebA', split='train', transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)


generator = Gen(Z_DIM).to(device)
critic = Critic().to(device)
 
if LOAD_MODEL:
	generator.load_state_dict(torch.load(os.path.join(model_path, 'generator.pkl')))
	critic.load_state_dict(torch.load(os.path.join(model_path, 'critic.pkl')))

g_opt = optim.Adam(generator.parameters(), lr=0.001, betas=(0.0, 0.9), weight_decay=2e-5)
c_opt = optim.Adam(critic.parameters(), lr=0.001, betas=(0.0, 0.9), weight_decay=2e-5)


total_iters = 0
g_loss = c_loss = torch.Tensor([0])
max_iter = len(data_loader)
fixed_z = torch.randn(IMGS_TO_DISPLAY, Z_DIM).to(device)
for epoch in range(EPOCHS):
	generator.train()
	critic.train()

	for i, data in enumerate(data_loader):
		total_iters += 1

		# Loading data
		x_real = data[0].to(device)
		z = torch.randn(x_real.shape[0], Z_DIM).to(device)
		x_fake = generator(z)

		# Train Critic
		fake_out = critic(x_fake.detach())
		real_out = critic(x_real.detach())
		x_out = torch.cat((real_out, fake_out))
		c_loss = (real_out.mean() - fake_out.mean()) + gradient_penalty(x_real, x_fake)*GRADIENT_PENALTY + (x_out ** 2).mean()*0.0001

		c_opt.zero_grad()
		c_loss.backward()
		c_opt.step()

		# Train Generator
		if total_iters % N_CRITIC == 0:
			z = torch.randn(BATCH_SIZE, Z_DIM).to(device)
			x_fake = generator(z)

			fake_out = critic(x_fake)
			g_loss = fake_out.mean()

			g_opt.zero_grad()
			g_loss.backward()
			g_opt.step()


		if i % 50 == 0:
			print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
				  + "\titer: " + str(i) + "/" + str(max_iter)
				  + "\ttotal_iters: " + str(total_iters)
				  + "\td_loss:" + str(round(c_loss.item(), 4))
				  + "\tg_loss:" + str(round(g_loss.item(), 4))
				  )
			
			generate_imgs(fixed_z, epoch=epoch + 1)

	torch.save(generator.state_dict(), os.path.join(model_path, 'generator.pkl'))
	torch.save(critic.state_dict(), os.path.join(model_path, 'critic.pkl'))
generate_imgs(fixed_z)
