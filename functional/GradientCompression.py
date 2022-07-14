import torch
import numpy as np
from scipy.spatial import distance


class GC():
    def __init__(self, compress_rate, distance_rate):
        self.compress_rate = compress_rate
        self.distance_rate = distance_rate

    def compression(self, item):
        size = item.grad.data.size()
        if len(size) == 1:
            return
        length = 1
        for i in range(len(size)):
            length = length * size[i]
        #mat = self.get_filter_grad_codebook(weight_grad_torch=item.grad.data, compress_rate=self.compress_rate, length=length)
        similar_matrix = self.get_filter_grad_similar(weight_grad_torch=item.grad.data, compress_rate=self.compress_rate,
                                                      distance_rate=self.distance_rate, length=length)
        #mat = torch.FloatTensor(mat)
        #similar_matrix = torch.FloatTensor(similar_matrix)
        #item.grad.data = self.do_grad_mask(item=item, size=size, length=length, mat=mat, similar_matrix=similar_matrix)
        item.grad.data = self.do_grad_similar_mask(item=item, size=size, length=length,
                                                   similar_matrix=similar_matrix)

    def get_filter_grad_codebook(self, weight_grad_torch, compress_rate, length):
        codebook = np.ones(length)

        # convolution
        if len(weight_grad_torch.size()) == 4:
            filter_pruned_num = int(weight_grad_torch.size()[0] * (1 - compress_rate))
            weight_grad_vec = weight_grad_torch.view(weight_grad_torch.size()[0], -1)
            norm2 = torch.norm(weight_grad_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_grad_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_grad_torch.size()[1] * weight_grad_torch.size()[2] * weight_grad_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

        # full connection
        elif len(weight_grad_torch.size()) == 2:
            filter_pruned_num = int(weight_grad_torch.size()[0] * (1 - compress_rate))
            weight_grad_vec = weight_grad_torch.view(weight_grad_torch.size()[0], -1)
            norm2 = torch.norm(weight_grad_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]

            kernel_length = weight_grad_torch.size()[1]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

        else:
            pass
        return codebook

    def get_filter_grad_similar(self, weight_grad_torch, compress_rate, distance_rate, length, dist_type="mean"):
        codebook = torch.ones(length).cuda()

        if len(weight_grad_torch.size()) == 4:
            filter_pruned_num = int(weight_grad_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_grad_torch.size()[0] * distance_rate)
            weight_grad_vec = weight_grad_torch.view(weight_grad_torch.size()[0], -1)

            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_grad_vec, 2, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_grad_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]

            # distance using numpy function
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_grad_vec, 0,
                                                       indices).cpu().numpy()
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm,
                                                'euclidean') 
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            kernel_length = weight_grad_torch.size()[1] * weight_grad_torch.size()[2] * weight_grad_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0

        elif len(weight_grad_torch.size()) == 2:
            # filter_pruned_num = int(weight_grad_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(length * distance_rate)
            weight_grad_vec = weight_grad_torch

            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = torch.cdist(weight_grad_vec, weight_grad_vec,
                                                p=2)  
                similar_sum = torch.sum(torch.abs(similar_matrix), dim=0)
                similar_small_index = torch.argsort(similar_sum)[: similar_pruned_num]
                codebook[similar_small_index] = 0
            elif dist_type == "mean":
                similar_matrix = torch.abs(weight_grad_vec.view(-1) - torch.mean(weight_grad_vec.view(-1)))
                similar_small_index = torch.argsort(similar_matrix)[: similar_pruned_num]
                codebook[similar_small_index] = 0


            # for distance similar: get the filter index with largest similarity == small distance
            # similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            # similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            # similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            # kernel_length = weight_grad_torch.size()[1]
            # for x in range(0, len(similar_index_for_filter)):
            #     codebook[similar_index_for_filter[x] * kernel_length:
            #              (similar_index_for_filter[x] + 1) * kernel_length] = 0

        else:
            pass
        return codebook

    def do_grad_mask(self, item, size, length, mat, similar_matrix):
        a = item.grad.data.view(length)
        # reverse the mask of model
        # b = a * (1 - self.mat[index])
        if a.is_cuda:
            mat = mat.cuda()
            similar_matrix = similar_matrix.cuda()
        b = a * mat
        b = b * similar_matrix
        # item.grad.data = b.view(size)
        return b.view(size)

    def do_grad_similar_mask(self, item, size, length, similar_matrix):
        a = item.grad.data.view(length)
        # reverse the mask of model
        # b = a * (1 - self.mat[index])
        if a.is_cuda:
            similar_matrix = similar_matrix.cuda()
        b = a * similar_matrix
        # item.grad.data = b.view(size)
        return b.view(size)


class HGC():
    def __init__(self, compress_rate, distance_rate):
        self.compress_rate = compress_rate
        self.distance_rate = distance_rate

    def compression(self, item):
        size = item.grad.data.size()
        if len(size) == 1:
            return
        length = 1
        for i in range(len(size)):
            length = length * size[i]
        similar_matrix = self.get_filter_grad_similar(weight_grad_torch=item.grad.data, compress_rate=self.compress_rate,
                                                      distance_rate=self.distance_rate, length=length)
        similar_matrix = torch.FloatTensor(similar_matrix)
        item.grad.data = self.do_grad_similar_mask(item=item, size=size, length=length,
                                                   similar_matrix=similar_matrix)

    def get_filter_grad_similar(self, weight_grad_torch, compress_rate, distance_rate, length, dist_type="l2"):
        codebook = np.ones(length)

        # 卷积层
        if len(weight_grad_torch.size()) == 4:
            filter_pruned_num = int(weight_grad_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_grad_torch.size()[0] * distance_rate)
            weight_grad_vec = weight_grad_torch.view(weight_grad_torch.size()[0], -1)

            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_grad_vec, 2, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_grad_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]

            # distance using numpy function
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_grad_vec, 0,
                                                       indices).cpu().numpy() 
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm,
                                                'euclidean') 
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            kernel_length = weight_grad_torch.size()[1] * weight_grad_torch.size()[2] * weight_grad_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0

        elif len(weight_grad_torch.size()) == 2:
            filter_pruned_num = int(weight_grad_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_grad_torch.size()[0] * distance_rate)
            weight_grad_vec = weight_grad_torch.view(weight_grad_torch.size()[0], -1)
            weight_grad_vec = weight_grad_vec.cpu().numpy()
            weight_grad_h = np.fft.fft2(weight_grad_vec)
            weight_grad_p = np.angle(weight_grad_h)
            weight_grad_p = torch.FloatTensor(weight_grad_p).cuda()

            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_grad_p, 1, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_grad_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]
            # codebook[filter_large_index] = 0

            # distance using numpy function
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_grad_p, 0,
                                                       indices).cpu().numpy()  
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm,
                                                'euclidean')  
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            kernel_length = weight_grad_torch.size()[1]
            for x in range(0, len(similar_index_for_filter)):
                codebook[similar_index_for_filter[x] * kernel_length:
                         (similar_index_for_filter[x] + 1) * kernel_length] = 0

        else:
            pass
        return codebook

    def do_grad_similar_mask(self, item, size, length, similar_matrix):
        a = item.grad.data.view(length)
        # reverse the mask of model
        # b = a * (1 - self.mat[index])
        if a.is_cuda:
            similar_matrix = similar_matrix.cuda()
        b = a * similar_matrix
        # item.grad.data = b.view(size)
        return b.view(size)
