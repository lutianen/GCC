import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
# from scipy.io import loadmat, savemat
from hdf5storage import loadmat, savemat
# from sklearn.metrics import roc_curve, roc_auc_score


class Data(Dataset):
    def __init__(self):
        super(Data, self).__init__()
        self.x_input, self.data_input = self.prepare_data()

    def prepare_data_segundo(self):
        load_fn = '/home/worker1/distributed_fpgm_int8/data/Segundo.mat'
        load_data = loadmat(load_fn)
        x_input = load_data['data']
        x_input = np.array(x_input)
        if len(x_input.shape) == 3:
            x_input = x_input.reshape((x_input.shape[0]*x_input.shape[1], x_input.shape[2]), order='F')
        x_input = 2 * ((x_input - x_input.min()) /
                       (x_input.max() - x_input.min()))

        load_fn = '/home/worker1/distributed_fpgm_int8/data/Segundo.mat'
        load_data = loadmat(load_fn)
        d_input = load_data['d']
        d_input = np.array(d_input)
        d_input = 2 * ((d_input - d_input.min()) /
                       (d_input.max() - d_input.min()))
        return x_input, d_input

    def prepare_data(self):
        load_fn = '/home/worker1/DATASETS/xiongan/xiongan.mat'
        load_data = loadmat(load_fn)
        x_input = load_data['data']
        x_input = np.array(x_input)
        if len(x_input.shape) == 3:
            x_input = x_input.reshape((x_input.shape[0]*x_input.shape[1], x_input.shape[2]), order='F')
        x_input = (x_input - x_input.min()) / (x_input.max() - x_input.min())

        load_fn = '/home/worker1/distributed_fpgm_int8/data/xiongan_water.mat'
        load_d = loadmat(load_fn)
        d_input = load_d['d']
        d_input = np.array(d_input)
        d_input = 2 * ((d_input - d_input.min()) /
                       (d_input.max() - d_input.min()))

        return x_input, d_input

    def prepare_data_rit(self):
        load_fn = '/home/worker1/DATASETS/AeroRIT/image_hsi_radiance.mat'
        load_data = loadmat(load_fn)
        x_input = load_data['data']
        x_input = np.array(x_input)
        if len(x_input.shape) == 3:
            x_input = x_input.reshape((x_input.shape[0] * x_input.shape[1], x_input.shape[2]), order='F')
        x_input = (x_input - x_input.min()) / (x_input.max() - x_input.min())

        load_fn = '/home/worker1/distributed_fpgm_int8/data/aerorit_car.mat'
        load_data = loadmat(load_fn)
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
    def __init__(self, dim_data, dim_z, n_hidden=400):
        super(AE, self).__init__()
        self.dim_data = dim_data
        self.dim_z = dim_z
        self.n_hidden = n_hidden
        # MLP_encoder
        self.encoder1 = torch.nn.Linear(self.dim_data, self.n_hidden)
        self.encoder2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.encoder3 = torch.nn.Linear(self.n_hidden, self.dim_z)
        # MLP_decoder
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
        SAM = torch.randint(1, [1, 1]).cuda()
        num = y_true.shape[0]
        print(num)
        for i in range(num):
            d = torch.transpose(data_input, 1, 0)
            A = torch.sum(torch.multiply(y_pred[i, :], d))
            B = torch.norm(y_pred[i, :], p=2)
            C = torch.norm(d, p=2)
            defen = torch.div(A, B*C+1e-5)
            defen = torch.reshape(defen, [1, 1])
            SAM = torch.cat([SAM, defen], 0)
        s = torch.topk(SAM[1:, :], k=20, dim=0).values
        sam_loss = torch.sum(s)
        mse_loss = F.mse_loss(y_pred, y_true, reduce=True)
        distance_loss = mse_loss + 0.1*sam_loss
        return distance_loss

    def forward(self, x, data_input, with_decoder=True):
        z = self.MLP_encoder(x)
        if with_decoder == False:
            return z

        y = self.MLP_decoder(z)
        # loss
        R_loss = torch.sum(torch.sum(self.my_loss(x, y, data_input)))
        # R_loss = torch.sum(torch.sum(F.mse_loss(x, y)))

        R_loss = torch.mean(R_loss)
        # D_loss = torch.mean(D_loss)
        # G_loss = torch.mean(G_loss)

        return y, z, R_loss# , D_loss, G_loss


def detection(map, data, reconstruct_result):
    reconstruct_result = (reconstruct_result - reconstruct_result.min()) / \
                         (reconstruct_result.max() - reconstruct_result.min())
    lamda = 10
    max = 4
    norm_a = np.linalg.norm(reconstruct_result, ord=2, axis=0)
    norm_b = np.linalg.norm(data, ord=2, axis=0)
    dot = np.sum(np.multiply(reconstruct_result, data), axis=0)
    sam = np.arccos(dot / (norm_a * norm_b))

    project_path = '/home/worker1/distributed_fpgm_int8/'
    result_coarse = loadmat(project_path+'coarse/result_coarse.mat')['result_coarse']

    # engine = matlab.engine.start_matlab()
    # output = engine.nonlinear(result_coarse, lamda, max)
    # B = sam * output

    # FPR, TPR, thresholds = roc_curve(map, B)
    # auc = roc_auc_score(map, B)


def main():
    project_path = '/home/worker1/distributed_fpgm_int8/'

    torch.cuda.set_device(0)

    test_dataset = Data()
    test_dataloader = DataLoader(test_dataset, batch_size=310000,shuffle=False)

    dim_data = test_dataset.x_input.shape[1]
    dim_z = 50

    # module_path = project_path + 'result/xa_size2GCTruesumdlr.pth'
    module_path = project_path + 'result/xa_GCC_new_0.9.pth'
    savepath = project_path + 'result/xa_GCC_new_0.9.mat'

    module_ae = AE(dim_data, dim_z)
    module_ae.load_state_dict(torch.load(module_path))
    module_ae.cuda()
    module_ae.eval()

    with torch.no_grad():
        for batch_index, data in enumerate(test_dataloader):
            data_input = torch.tensor(test_dataset.data_input, dtype=torch.float32).cuda()
            data_input = data_input.float()
            data = data.cuda()
            data = data.float()
            y, _, _ = module_ae(data, data_input)
            y = y.cpu().numpy()
            if batch_index == 0:
                y_pred = y
            else:
                y_pred = np.concatenate((y_pred, y), axis=0)
        savemat(savepath, {'y': y_pred}, format='7.3')


if __name__ == '__main__':
    main()
