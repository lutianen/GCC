clc, clear
close all
%% Load the hyperspectral image and ground truth
addpath('../functions')
addpath('../data')
% load Segundo
% load ../coarse/result_coarse
%load ../result/reconstruct_result
load('xiongan.mat')
load('xiongan_water.mat')
load('../result/xa_GCC_new_0.9.mat')
% load('image_hsi_radiance.mat')
% load('../result/rit_GCC_new_0.8.mat')
tic
data=double(data);
[w, h, bs] = size(data);
data = hyperNormalize(data);
data_r = hyperConvert2d(data)';
% reconstruct_result = hyperNormalize(reconstruct_result);
reconstruct_result = hyperNormalize(y);
%% Parameters setup
lamda = 10;
max = 4;% FOR AeroRIT dataset,max=200;FOR XiongAn dataset,max=4
%% Difference
for i = 1: w*h
    sam(i)= hyperSam(data_r(i,:), reconstruct_result(i,:));
end 
sam = reshape(sam , w, h);
SAM = hyperNormalize( sam );
%% Binary the difference 
% output  = nonlinear(SAM, lamda, max );% FOR AeroRIT dataset
output  = nonlinear(result_coarse, lamda, max );% FOR XiongAn dataset
B = SAM.* output;
res=B;

toc
[FPR,TPR,thre] = myPlot3DROC( map, B);
auc = -trapz(FPR,TPR);
fpr = -trapz(FPR,thre);
figure, imagesc(B), axis image, axis off
