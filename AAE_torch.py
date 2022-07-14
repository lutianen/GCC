import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import os
from torch.autograd import Variable
parser = argparse.ArgumentParser(description = 'torch AAE')
parser.add_argument('--batch_size', type = int, default = 10000, metavar = 'N')
parser.add_argument('--epochs', type = int, default = 5000, metavar = 'N')
parser.add_argument('--learning_rate', type = int, default = 0.001, metavar = 'N')
args = parser.parse_args()
cuda = torch.cuda.is_available()
class MyDataset(Dataset):
    def __init__(self, path):
        super(MyDataset, self).__init__()
        self.data_path = path
        self.data = loadmat(self.data_path)['data'].astype(np.float32)
        self.map = loadmat(self.data_path)['map']
        self.norm()
        self.size = self.map.shape
        self.data_dim = self.data.shape[-1]
        self.num_sampales = self.size[0]*self.size[1]
        self.reshape()
    def norm(self):
        self.data = (self.data-np.min(self.data))/(np.max(self.data)-np.min(self.data))

    def reshape(self):
        self.data = np.reshape(self.data, [self.num_sampales, self.data_dim])
        self.map = np.reshape(self.map,[self.num_sampales,-1])

    def get_map(self):
        return self.map, self.map.shape

    def get_dim(self):
        return self.data_dim

    def __getitem__(self,idx):
        return  [self.data[idx,:], self.map[idx,:]]

    def __len__(self):
        return self.num_sampales

class Q_net(nn.Module):
    def __init__(self, size):
        self.size = size
        super(Q_net, self).__init__()
        self.linear_1 = nn.Linear(size, 500)
        self.norm_1 = nn.BatchNorm1d(500)
        self.linear_2 = nn.Linear(500, 300)
        self.norm_2 = nn.BatchNorm1d(300)
        self.linear_3 = nn.Linear(300, 20)

    def forward(self, inputs):
#y = nn.functional.dropout(self.linear_1(inputs), p = 0.2, training = self.training)
        y = self.linear_1(inputs)
        y = self.norm_1(y)
        y = nn.functional.relu(y)
        
#y = nn.functional.dropout(self.linear_2(y), p = 0.2, training = self.training)
        y = self.linear_2(y)
        y = self.norm_2(y)
        y = nn.functional.relu(y)

        y = self.linear_3(y)
    
        return  y
        
class P_net(nn.Module):
    def __init__(self,size):
        self.size = size
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(20, 300)
        self.norm1 = nn.BatchNorm1d(300)
        self.lin2 = nn.Linear(300, 500)
        self.norm2 = nn.BatchNorm1d(500)
        self.lin3 = nn.Linear(500, size)

    def forward(self, x):
        x = self.lin1(x)
#x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.norm1(x)
        x = nn.functional.relu(x)
        x = self.lin2(x)
#x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.norm2(x)
        x = self.lin3(x)
        return nn.functional.sigmoid(x)

class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(20, 300)
        self.norm1 = nn.BatchNorm1d(300)
        self.lin2 = nn.Linear(300, 500)
        self.norm2 = nn.BatchNorm1d(500)
        self.lin3 = nn.Linear(500, 1)

    def forward(self, x):
#x = nn.functional.dropout(self.lin1(x), p=0.2, training=self.training)
        x = self.lin1(x)
        x = self.norm1(x)
        x = nn.functional.relu(x)
        x = self.lin2(x)
        x = self.norm2(x)
#x = nn.functional.dropout(self.lin2(x), p=0.2, training=self.training)
        x = nn.functional.relu(x)
        return nn.functional.sigmoid(self.lin3(x))

def train(fileName):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filename = fileName
    TINY = 1e-15
    path = '/home/data/zhangxin/abu/' + filename + '.mat'
    dataset = MyDataset(path)    
    Q = Q_net(dataset.data_dim).cuda()
    P = P_net(dataset.data_dim).cuda()
    D_guss = D_net_gauss().cuda()

    Q_encoder = torch.optim.SGD(Q.parameters(),lr=args.learning_rate)
    P_decoder = torch.optim.SGD(P.parameters(),lr=args.learning_rate)
    Q_generator = torch.optim.SGD(Q.parameters(),lr=(args.learning_rate / 2))
    D_guss_solver = torch.optim.SGD(D_guss.parameters(),lr=(args.learning_rate / 2))
    
    train_loader = DataLoader(dataset,batch_size=args.batch_size)
    mse_loss = nn.MSELoss()
    time_s = time.time()
    for epoch in range(args.epochs):
        st = time.time()
        for i,load in enumerate(train_loader):
            load=load
            ori = load[0]
            map = load[1]
            ori = ori.to(DEVICE)
            map = map.to(DEVICE)
            
            P.zero_grad()
            Q.zero_grad()
            D_guss.zero_grad()
            
            z_sample = Q(ori)
            x_sample = P(z_sample)
            
            recon_loss = mse_loss(ori, x_sample)
            recon_loss.backward()
            Q_encoder.step()
            P_decoder.step()
            
            P.zero_grad()
            Q.zero_grad()
            D_guss.zero_grad()
            
            Q.eval()
            z_real_gauss = Variable(torch.randn(ori.shape[0], 20) * 5.)
            z_real_gauss = z_real_gauss.cuda()
            z_fake_gauss = Q(ori)
            
            D_real_gauss = D_guss(z_real_gauss)
            D_fake_gauss = D_guss(z_fake_gauss)
            
            D_loss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))
            D_loss.backward()
            D_guss_solver.step()
            
            P.zero_grad()
            Q.zero_grad()
            D_guss.zero_grad()
            
            Q.train()
            
            z_fake_gauss = Q(ori)
            
            D_fake_gauss = D_guss(z_fake_gauss)
            G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))
            G_loss.backward()
            Q_generator.step()
            
            P.zero_grad()
            Q.zero_grad()
            D_guss.zero_grad()
            if (epoch+1)%(args.epochs / 10) == 0:
                print('epoch' + str(epoch + 1) + '/' + str(args.epochs) + ':' + str(recon_loss) +
                '   '+ str(D_loss) + '   ' + str(G_loss))
        print('cost:{}'.format(time.time()-st))
    time_e = time.time()
    savemat(filename + '-AAE.mat', {'code':(z_fake_gauss.cpu().detach().numpy()),'map':(map.cpu().detach().numpy())})
    with open("AAE_time_ori.txt","a+")  as fp:
        fp.write("%s | epoch : %d | time : %f s | loss : %f  |  loss : %f  | loss : %f  \n"%(filename,args.epochs,time_e-time_s,recon_loss,D_loss ,G_loss))
if __name__ == '__main__':
    file_list = os.listdir( '/home/data/zhangxin/abu/' )
    for i  in range(len(file_list)):
        train(file_list[i].split('.')[0])   
#train_distri('abu-airport-1')
    #train_horovod('abu-airport-1')
    #train_grace('abu-airport-1')
