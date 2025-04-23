import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torch.nn.init as init
from torch.autograd import Variable
import datetime
from models.tanogan import LSTMGenerator, LSTMDiscriminator
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scripts.data_loader import get_dataloader, get_dataset, get_windows
from scripts.gan import save_gan, load_gan
from scripts.clf import train_clf, evaluate_clf
from scipy.io import savemat

class Classifier(nn.Module):
    def __init__(self, 
                 data_shape, 
                 label_shape,
                 latent_size=128):
        super().__init__()

        self.input_size = data_shape[0] * data_shape[1]
        self.latent_size = latent_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size, .01),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(self.latent_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        x = torch.flatten(data, 1)
        return self.model(x)


class ArgsTrn:
    workers=4
    batch_size=32
    epochs=20
    lr=0.0002
    cuda = True
    manualSeed=2
    
opt_trn=ArgsTrn()
torch.manual_seed(opt_trn.manualSeed)
cudnn.benchmark = True

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

scaler = MinMaxScaler()
A, A_label = get_dataset('data/ue_jamming_detection/train.csv')
B, B_label = get_dataset('data/ue_jamming_detection/valid.csv')
A[A.columns] = scaler.fit_transform(A)
B[B.columns] = scaler.transform(B)

# A, A_label = get_windows(A, A_label, args.window)
# B, B_label = get_windows(B, B_label, args.window)
A, A_label = A.to_numpy().reshape(-1, 72, 1), A_label.reshape(-1, 1)
B, B_label = B.to_numpy().reshape(-1, 72, 1), B_label.reshape(-1, 1)

# A_0 = A[np.apply_along_axis(np.mean, 1, A_label) == 0]
# A_1 = A[np.apply_along_axis(np.mean, 1, A_label) == 1]

A_train_loader, A_test_loader = get_dataloader(
    A, A_label, device=device, batch_size=opt_trn.batch_size, train_test_split=.2)
A_loader, _ = get_dataloader(
    A, A_label, device=device, batch_size=opt_trn.batch_size)
B_loader, _ = get_dataloader(
    B, B_label, device=device, batch_size=opt_trn.batch_size)

in_dim = A.shape[-1]

netD = LSTMDiscriminator(in_dim=in_dim, device=device).to(device)
netG = LSTMGenerator(in_dim=in_dim, out_dim=in_dim, device=device).to(device)

if os.path.exists('backups/tanogan'):
    load_gan(netG, netD, 'backups/tanogan')
else:

    criterion = nn.BCELoss().to(device)

    optimizerD = optim.Adam(netD.parameters(), lr=opt_trn.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt_trn.lr)
    real_label = 1
    fake_label = 0

    for epoch in range(opt_trn.epochs):
        for i, (x,y) in enumerate(A_train_loader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            #Train with real data
            netD.zero_grad()
            real = x.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            label = torch.full((batch_size, seq_len, 1), real_label, device=device)

            output,_ = netD.forward(real)
            errD_real = criterion(output, label.float())
            errD_real.backward()
            optimizerD.step()
            D_x = output.mean().item()
            
            #Train with fake data
            noise = Variable(init.normal(torch.Tensor(batch_size,seq_len,in_dim),mean=0,std=0.1)).cuda()
            fake,_ = netG.forward(noise)
            output,_ = netD.forward(fake.detach()) # detach causes gradient is no longer being computed or stored to save memeory
            label.fill_(fake_label)
            errD_fake = criterion(output, label.float())
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            noise = Variable(init.normal(torch.Tensor(batch_size,seq_len,in_dim),mean=0,std=0.1)).cuda()
            fake,_ = netG.forward(noise)
            label.fill_(real_label) 
            output,_ = netD.forward(fake)
            errG = criterion(output, label.float())
            errG.backward()
            optimizerG.step()
            D_G_z2 = output.mean().item()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
            % (epoch, opt_trn.epochs, i, len(A_train_loader),
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='')

    os.makedirs('backups/tanogan', exist_ok=True)
    save_gan(netG, netD, 'backups/tanogan')

augment_ratios = [.1, .2, .3, .5, .7, 1., 1.5, 2., 3., 5.]

def gen_synthetic(gen, amount, label):
    noise = torch.randn(amount, 72, 1)
    x_flag, _ = gen(noise)
    y_flag = np.full((amount,1), label)
    return x_flag.detach().cpu().numpy(), y_flag

class Args:
    lr_clf=0.001
    batch_size=256
    epoch_num=20
    block_size=10

args = Args()
outputs = {}

for ratio in augment_ratios:
    ratio_pc = int(ratio*100)
    print('Augmenting with {}% ..'.format(ratio_pc))
    amount = int(A.shape[0] * ratio // 2)

    x_flag_0, y_flag_0 = gen_synthetic(netG, amount, 0)
    x_flag_1, y_flag_1 = gen_synthetic(netG, amount, 1)

    A_flag = np.concatenate([A, x_flag_0, x_flag_1], axis=0)
    A_label_flag = np.concatenate([A_label, y_flag_0, y_flag_1], axis=0)

    A_flag_train_loader, A_flag_test_loader = get_dataloader(A_flag, A_label_flag, device=device, batch_size=args.batch_size, train_test_split=.1)

    clf_augmented = Classifier(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[1, 1],
    ).to(device)

    train_clf(clf_augmented, A_flag_train_loader, A_flag_test_loader, args)

    key_A = f'tanogan_A_{ratio_pc}'
    key_B = f'tanogan_B_{ratio_pc}'

    results_A = evaluate_clf(clf_augmented, A_loader, args)
    results_B = evaluate_clf(clf_augmented, B_loader, args)

    outputs[key_A] = results_A
    outputs[key_B] = results_B

    print('({}) Perf on A set: {}'.format(ratio_pc, results_A))
    print('({}) Perf on B set: {}'.format(ratio_pc, results_B))

savemat('backups/tanogan/eval.mat', outputs)