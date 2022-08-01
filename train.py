import random
import horovod.torch as hvd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import numpy as np
import time
# from scipy.io import loadmat, savemat
from hdf5storage import loadmat
from functional.GradientCompression import GC
from functional.GradientCompression import HGC
import matplotlib.pyplot as plt
import math


def psnr(data_input, reconstruct):
    data_input = (data_input-data_input.min())/(data_input.max()-data_input.min())
    reconstract = (reconstruct-reconstruct.min()) / \
        (reconstruct.max()-reconstruct.min())
    target_data = np.array(data_input)
    ref_data = np.array(reconstruct)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20*np.log10(1.0/rmse)


class Data(Dataset):
    def __init__(self, rank=0):
        super(Data, self).__init__()
        # self.x_input, self.d_input = self.prepare_data_segundo()
        # self.x_input, self.d_input = self.prepare_data_rit()
        self.x_input, self.d_input = self.prepare_data_xa()
        self.nums = int(self.x_input.shape[0])
        self.rank = int(rank)

    def prepare_data_segundo(self):
        load_fn = '/home/worker1/distributed_fpgm_int8/coarse/train_data.mat'
        load_data = loadmat(load_fn)
        x_input = load_data['train_data']
        x_input = np.array(x_input)
        x_input = 2 * ((x_input - x_input.min()) /
                       (x_input.max() - x_input.min()))

        load_fn = '/home/worker1/distributed_fpgm_int8/data/Segundo.mat'
        load_data = loadmat(load_fn)
        d_input = load_data['d']
        d_input = np.array(d_input)
        d_input = 2 * ((d_input - d_input.min()) /
                       (d_input.max() - d_input.min()))
        return x_input, d_input

    def prepare_data_xa(self):
        # load_fn = '/home/worker1/distributed_fpgm_int8/coarse/train_data.mat'
        load_fn = '/home/worker1/distributed_fpgm_int8/data/xiongan_water.mat'
        load_data = loadmat(load_fn)
        x_input = load_data['train_data']
        x_input = np.array(x_input)
        x_input = 2 * ((x_input - x_input.min()) /
                       (x_input.max() - x_input.min()))

        # load_fn = '/home/worker1/distributed_fpgm_int8/data/Segundo.mat'
        # load_d = loadmat(load_fn)
        d_input = load_data['d']
        d_input = np.array(d_input)
        d_input = 2 * ((d_input - d_input.min()) /
                       (d_input.max() - d_input.min()))
        return x_input, d_input

    def prepare_data_rit(self):
        load_fn = '/home/worker1/distributed_fpgm_int8/data/aerorit.mat'
        load_data = loadmat(load_fn)
        x_input = load_data['data']
        x_input = np.array(x_input)
        if len(x_input.shape) == 3:
            x_input = x_input.reshape((x_input.shape[0] * x_input.shape[1], x_input.shape[2]), order='F')
        x_input = (x_input - x_input.min()) / (x_input.max() - x_input.min())

        d_input = load_data['d']
        d_input = np.array(d_input)
        d_input = 2 * ((d_input - d_input.min()) /
                       (d_input.max() - d_input.min()))
        return x_input, d_input

    def __len__(self):
        return self.x_input.shape[0]

    def __getitem__(self, item):
        return self.x_input[item, :]


class AE(torch.nn.Module):
    def __init__(self, dim_data, dim_z, n_hidden=400, n_hidden_d=1000, n_output_d=1):
        super(AE, self).__init__()
        self.dim_data = dim_data
        self.dim_z = dim_z
        self.n_hidden = n_hidden

        # MLP_encoder
        self.encoder1 = torch.nn.Linear(self.dim_data, self.n_hidden)
        self.encoder2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.encoder3 = torch.nn.Linear(self.n_hidden, self.dim_z)
        # MLP_decoker
        self.decoder1 = torch.nn.Linear(self.dim_z, self.n_hidden)
        self.decoder2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.decoder3 = torch.nn.Linear(self.n_hidden, self.dim_data)


    def MLP_encoder(self, x):
        x1 = self.encoder1(x)
        x1 = F.leaky_relu(x1, 0.1, inplace=False)

        x2 = self.encoder2(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=False)

        z = self.encoder3(x2)
        return z

    def MLP_decoder(self, z):
        z1 = self.decoder1(z)
        z1 = F.leaky_relu(z1, 0.1, inplace=False)

        z2 = self.decoder2(z1)
        z2 = F.leaky_relu(z2, 0.1, inplace=False)

        y = self.decoder3(z2)
        y = F.tanh(y)
        return y

    def my_loss(self, y_true, y_pred, data_input):
        d = torch.transpose(data_input, 1, 0)
        A = torch.sum(torch.multiply(y_pred, d), dim=1)
        B = torch.norm(y_pred, p=2, dim=1)
        C = torch.norm(d, p=2)
        defen = torch.div(A, B*C+1e-5)
        s = torch.topk(defen, k=20, dim=0).values
        sam_loss = torch.sum(s)
        mse_loss = F.mse_loss(y_pred, y_true, reduce=True)
        distance_loss = mse_loss + 0.1 * sam_loss
        return distance_loss

    def forward(self, x, data_input, with_decoder=True):
        z = self.MLP_encoder(x)
        if with_decoder == False:
            return z

        y = self.MLP_decoder(z)
        # loss
        R_loss = torch.sum(torch.sum(self.my_loss(x, y, data_input)))

        R_loss = torch.mean(R_loss)

        return y, z, R_loss


class GAN(torch.nn.Module):
    def __init__(self, dim_data, dim_z, n_hidden_d=1000, n_output_d=1):
        super(GAN, self).__init__()
        self.dim_data = dim_data
        self.dim_z = dim_z
        self.n_hidden_d = n_hidden_d
        self.n_output_d = n_output_d
        # discriminator_real
        self.dr1 = torch.nn.Linear(self.dim_z, self.n_hidden_d)
        self.dr2 = torch.nn.Linear(self.n_hidden_d, self.n_hidden_d)
        self.dr3 = torch.nn.Linear(self.n_hidden_d, self.n_output_d)
        # discriminator_fake
        self.df1 = torch.nn.Linear(self.dim_z, self.n_hidden_d)
        self.df2 = torch.nn.Linear(self.n_hidden_d, self.n_hidden_d)
        self.df3 = torch.nn.Linear(self.n_hidden_d, self.n_output_d)

    def discriminator_real(self, z):
        z1 = self.dr1(z)
        z1 = F.leaky_relu(z1, inplace=False)

        z2 = self.dr2(z1)
        z2 = F.leaky_relu(z2, inplace=False)

        y = self.dr3(z2)
        return torch.sigmoid(y), y

    def discriminator_fake(self, z):
        z1 = self.df1(z)
        z1 = F.leaky_relu(z1, inplace=False)

        z2 = self.df2(z1)
        z2 = F.leaky_relu(z2, inplace=False)

        y = self.df3(z2)
        return torch.sigmoid(y), y

    def forward(self, z, with_G=True):
        z_simples = np.random.randn(self.dim_data, self.dim_z)
        z_simples = torch.tensor(z_simples, dtype=torch.float32)
        z_simples = z_simples.cuda()
        z_real = z_simples
        z_fake = z
        D_real, D_real_logits = self.discriminator_real(z_real)
        D_fake, D_fake_logits = self.discriminator_fake(z_fake)
        # D_real_logits.requires_grad = False
        # D_fake_logits.requires_grad = False
        # discriminator loss
        D_loss_real = torch.mean(F.binary_cross_entropy_with_logits(D_real_logits, torch.ones_like(D_real_logits)))
        D_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(D_fake_logits, torch.zeros_like(D_fake_logits)))
        D_loss = 0.5 * (D_loss_real + D_loss_fake)
        # generator loss
        G_loss = torch.mean(F.binary_cross_entropy_with_logits(D_fake_logits, torch.ones_like(D_fake_logits)))

        if with_G == False:
            D_loss = torch.mean(D_loss)
            return D_loss
        D_loss = torch.mean(D_loss)
        G_loss = torch.mean(G_loss)
        return D_loss, G_loss


def main(with_GC: bool = True, save_module: bool = False, batch_size: int = 200000):
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    project_path = '/home/worker1/distributed_fpgm_int8/'
    """ parameters """
    hvd.init()
    world_size = hvd.size()
    rank = hvd.rank()
    print('run in {} workers({})'.format(world_size, rank))
    torch.cuda.set_device(hvd.local_rank())

    print('reading data ...')
    dataset = Data()
    print('done')

    dim_z = 50
    dim_data = dataset.x_input.shape[1]
    # batch_size = int(dataset.num)
    batch_size = 250000
    print('batch_size:{}'.format(batch_size))
    n_epochs = 20
    learn_rate = 1e-4
    steps = math.ceil(dataset.nums/batch_size)

    # train_simpler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    module_ae = AE(dim_data=dim_data, dim_z=dim_z)
    module_ae.cuda()
    module_gan = GAN(dim_data=dim_data, dim_z=dim_z)
    module_gan.cuda()

    opt_ae = torch.optim.Adam(params=module_ae.parameters(), lr=learn_rate)
    opt_d = torch.optim.Adam(params=module_gan.parameters(), lr=learn_rate)
    opt_g = torch.optim.Adam(params=module_gan.parameters(), lr=learn_rate)
    opt_ae = hvd.DistributedOptimizer(opt_ae, named_parameters=module_ae.named_parameters(prefix='ae'), backward_passes_per_step=steps*3)
    opt_d = hvd.DistributedOptimizer(opt_d, named_parameters=module_gan.named_parameters(prefix='gen_d'), backward_passes_per_step=steps*3)
    opt_g = hvd.DistributedOptimizer(opt_g, named_parameters=module_gan.named_parameters(prefix='gen_g'), backward_passes_per_step=steps*3)

    hvd.broadcast_parameters(module_ae.state_dict(prefix='ae'), root_rank=0)
    hvd.broadcast_parameters(module_gan.state_dict(prefix='gen_d'), root_rank=0)
    hvd.broadcast_parameters(module_gan.state_dict(prefix='gen_g'), root_rank=0)

    cr = 0.5
    grad_compression = GC(compress_rate=1, distance_rate=cr)
    sta = time.time()

    loss_plot = {'R': [], 'D': [], 'G': []}
    for epoch in range(n_epochs):
        st = time.time()
        loss_per_epoch_r = 0
        loss_per_epoch_d = 0
        loss_per_epoch_g = 0
        if epoch == 0:
            with_GC = True
        for index, batch_data in enumerate(train_dataloader):
            x = batch_data.cuda()
            x = x.float()
            data_input = torch.tensor(dataset.d_input, dtype=torch.float32).cuda()
            data_input = data_input.float()
            y, _, R_loss = module_ae(x, data_input)
            R_loss.backward(retain_graph=True)
            loss_per_epoch_r = loss_per_epoch_r + R_loss.detach().cpu().numpy()
        if with_GC:
            for _, layer in enumerate(module_ae.parameters()):
                if torch.sum(layer.grad.data) != 0:
                    grad_compression.compression(layer)
        ttt=time.time()
        opt_ae.step()
        opt_ae.zero_grad()
        print(time.time()-ttt)

        for index, batch_data in enumerate(train_dataloader):
            x = batch_data.cuda()
            x = x.float()
            data_input = torch.tensor(dataset.d_input, dtype=torch.float32).cuda()
            data_input = data_input.float()
            z = module_ae(x, data_input, with_decoder=False)
            D_loss = module_gan(z, with_G=False)
            D_loss.backward(retain_graph=True)
            loss_per_epoch_d = loss_per_epoch_d + D_loss.detach().cpu().numpy()
        if with_GC:
            for _, layer in enumerate(module_ae.parameters()):
                if torch.sum(layer.grad.data) != 0:
                    grad_compression.compression(layer)
        opt_d.step()
        opt_d.zero_grad()

        for index, batch_data in enumerate(train_dataloader):
            x = batch_data.cuda()
            x = x.float()
            data_input = torch.tensor(dataset.d_input, dtype=torch.float32).cuda()
            data_input = data_input.float()
            z = module_ae(x, data_input, with_decoder=False)
            _, G_loss = module_gan(z)
            G_loss.backward()
            loss_per_epoch_g = loss_per_epoch_g + G_loss.detach().cpu().numpy()
        if with_GC:
            for _, layer in enumerate(module_ae.parameters()):
                if torch.sum(layer.grad.data) != 0:
                    grad_compression.compression(layer)
        opt_g.step()
        opt_g.zero_grad()

        et = time.time()
        loss_per_epoch_r = loss_per_epoch_r / (index+1)
        loss_per_epoch_d = loss_per_epoch_d / (index+1)
        loss_per_epoch_g = loss_per_epoch_g / (index+1)
        print('epoch:[{}/{}],cost:{}s\n    R_loss:{}\n    D_loss:{}\n    G_loss:{}'
              .format(epoch+1, n_epochs, et-st, loss_per_epoch_r, loss_per_epoch_d, loss_per_epoch_g))
        loss_plot['R'].append(loss_per_epoch_r)
        loss_plot['D'].append(loss_per_epoch_d)
        loss_plot['G'].append(loss_per_epoch_g)

    print('cost:{}'.format(time.time()-sta))

    if save_module:
        path = project_path + 'result/xa_GCC_new_' + str(cr) + '.pth'
        torch.save(module_ae.state_dict(), path)

    plt.figure()
    x = np.linspace(1, n_epochs+1, n_epochs)
    plt.plot(x, loss_plot['R'], color='red', label='R_loss')
    plt.plot(x, loss_plot['D'], color='blue', label='D_loss')
    plt.plot(x, loss_plot['G'], color='green', label='G_loss')
    plt.legend()
    image_save_path = project_path + 'result/xa_GCC_new_' + str(cr) + '.png'
    plt.savefig(image_save_path)
    # plt.show()


if __name__ == '__main__':
    main(with_GC=False, save_module=True)
