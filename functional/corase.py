import numpy as np
# from scipy.io import loadmat, savemat
from hdf5storage import loadmat, savemat
import matplotlib.pyplot as plt


def CME(data_path):
    # load dataset
    load_fn = loadmat(data_path)
    data = load_fn['data']
    data = np.array(data)
    data.astype(np.float32)
    ground_true = load_fn['map']

    # plt.figure()
    # plt.imshow(data[:,:,0])
    # plt.show()

    # generate known target sample -> d
    ground_true_index = np.where(ground_true == 1)
    # a = np.random.randint(len(ground_true_index[0]), size=1)
    d = data[ground_true_index[0], ground_true_index[1], :]
    d = np.mean(d, axis=0)
    d = d.reshape((d.size, 1))
    ground_true = np.zeros(ground_true.shape)
    ground_true[ground_true_index[0], ground_true_index[1]] = 1

    # # CME
    # Coarse detection
    if len(data.shape) == 3:
        size = data.shape
        data = data.reshape(size[0]*size[1], size[2])
    else:
        return
    data_r = np.matmul(np.transpose(data), data) / data.shape[0]
    w_x = np.linalg.solve(data_r, d)
    w = w_x / (np.matmul(np.transpose(d), w_x))
    w = np.transpose(w)
    z = np.matmul(w, np.transpose(data))
    z = np.reshape(z, [size[0], size[1]])
    result_coarse = np.abs(z)
    # Binarization
    k = 0.15
    result_binary = result_coarse.copy()
    binary_index = np.where(result_coarse > k)
    result_binary[binary_index[0], binary_index[1]] = 1
    binary_index = np.where(result_coarse < 1)
    result_binary[binary_index[0], binary_index[1]] = 0
    binary_index2 = np.where(np.reshape(result_coarse, [size[0]*size[1], 1]) < 1)
    # plt.figure()
    # plt.imshow(result_binary)
    # plt.show()
    # plt.figure()
    # plt.imshow(result_coarse)
    # plt.show()

    # get train data
    ratio = 0.75
    imax = len(binary_index2[0])
    background = data[binary_index2[0], :]
    np.random.shuffle(background)
    m = int(ratio * imax)
    train_data = background[0:m+1, :]
    val_data = background[m+1:, :]

    # save data
    savemat('/home/worker1/distributed_fpgm_int8/data/aerorit_car.mat', {'train_data': train_data, 'val_data': val_data, 'd': d, 'result_coarse': result_coarse, 'map': ground_true}, format='7.3')


def main():
    path = '/home/worker1/DATASETS/AeroRIT/image_hsi_radiance.mat'
    CME(path)


if __name__ == '__main__':
    main()
