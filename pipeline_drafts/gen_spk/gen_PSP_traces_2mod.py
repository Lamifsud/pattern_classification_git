#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:38:14 2020

@author: matgilson
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression


frmt_grph = 'png'
cols = ['r','b','gray']
cols2 = [[0.5,0,0],[0,0,0.5]]

scaling_aff = 0.5

# time parameters
T = 200.0
dt = 1.0
v_t = np.arange(0.0,T,dt)
n_T = v_t.size
n_T2 = int(n_T/2)
v_t2 = np.arange(0.0,T/2,dt)

N = 3 # 3 neurons

# trial repetitions
n_rep_aff = 2
n_rep = 100

# stimuli properties
n_stim = 2
t_stim = np.zeros([n_stim,2]) # start/end times of stimuli
t_stim[0,:] = 10.0, 80.0
t_stim[1,:] = 110.0, 170.0

# generate double-exponential kernel
def gen_kernel(t0, tau1, tau2):
    thres_v_t = np.maximum(0, v_t-t0)
    return (np.exp(-thres_v_t/tau1)-np.exp(-thres_v_t/tau2)) / (tau1-tau2)

alpha = np.zeros([n_stim,N]) # scaling factor for responses
# stim 0
alpha[0,:] = 7.0, 6.5, 1.0
# stim 1
alpha[1,:] = 1.5, 7.0, 6.0

tau_rise_decay = np.zeros([n_stim,N,2]) # rise/decay times for responses
lat_start_end = np.zeros([n_stim,2]) # latency ranges (for all neurons)

if True:
    # rate modulation: slow modulation = long kernels, fixed latency
    # spikes are distributed Poisson-like during the long bumps, the smoothed instantaneous firing rate should be stable across trials

    # stim 0
    tau_rise_decay[0,0,:] = 4, 12
    tau_rise_decay[0,1,:] = 5, 15
    tau_rise_decay[0,2,:] = 2, 9
    # stim 1
    tau_rise_decay[1,0,:] = 4, 16
    tau_rise_decay[1,1,:] = 3, 19
    tau_rise_decay[1,2,:] = 1.5, 8

    # stim 0
    lat_start_end[0,:] = 1.0, 1.0
    # stim 1
    lat_start_end[1,:] = 1.0, 1.0

else:
    # synchrony modulation: fast modulation = sharp kernels, variable latency
    # spikes tend to happen during the short modulation, but their latencies vary from trial to trial

    # stim 0
    tau_rise_decay[0,0,:] = 1.0, 2.5
    tau_rise_decay[0,1,:] = 1.0, 3.0
    tau_rise_decay[0,2,:] = 1.0, 5.0
    # stim 1
    tau_rise_decay[1,0,:] = 1.0, 4.0
    tau_rise_decay[1,1,:] = 1.0, 2.0
    tau_rise_decay[1,2,:] = 1.0, 2.0
    
    # stim 0
    lat_start_end[0,:] = 1.0, 50.0
    # stim 1
    lat_start_end[1,:] = 1.0, 50.0

offset = 0.2

# index of color for plotting
v_t_col = np.zeros([n_T], dtype=int)
for i_t, t in enumerate(v_t):
    if t>t_stim[0,0] and t<t_stim[0,1]:
        v_t_col[i_t] = 0
    elif t>t_stim[1,0] and t<t_stim[1,1]:
        v_t_col[i_t] = 1
    else:
        v_t_col[i_t] = 2

# line styles for lagged covariances
ls = ['-','--',':']


#%% generate spike trains

# smoothed spike trains
smoothed_rate_ts = np.zeros([N,n_T2,2*n_rep])
tau_smooth = 5.0
v_smooth = np.arange(-4*tau_smooth, 4*tau_smooth+dt*0.5, dt)
kernel_smooth = np.exp(-v_smooth**2 / tau_smooth**2 / 2.0)

# lagged (cross-)covariance
L = 30
v_bins = np.arange(-L,L,dt)
n_L = v_bins.size
n_L2 = int(n_L/2)
cov = np.zeros([N,N,n_L,n_stim,n_rep])


# loop over repetitions for generating n_rep samples, with 1 red and 1 blue stimulus each repetition
for i_rep in range(n_rep):

    inst_rate = np.zeros([N,n_T])
    for i in range(N):
        inst_rate[i,:] += offset
    for i_stim in range(n_stim):
        t_start, t_stop = lat_start_end[i_stim]
        lat = t_start + np.random.rand() * (t_stop - t_start)
        for i in range(N):
            inst_rate[i,:] += alpha[i_stim,i] * gen_kernel(t_stim[i_stim,0] + lat, tau_rise_decay[i_stim,i,0], tau_rise_decay[i_stim,i,1])

    # gen spike train and apply smoothing
    spk_ts = []
    for i in range(N):
        # binary vector
        v_spk_bin = np.random.rand(n_T)<inst_rate[i,:]*dt*0.5
        # list of spike times
        spk_ts += [np.argwhere(v_spk_bin).flatten()]
        # apply smoothing, first half of trial is red stimulus, second half is blus stimulus
        smoothed_rate_ts[i,:,i_rep] = np.convolve(v_spk_bin[:n_T2], kernel_smooth, 'same')
        smoothed_rate_ts[i,:,i_rep+n_rep] = np.convolve(v_spk_bin[n_T2:], kernel_smooth, 'same')
        
    # lagged covariancefor each stimulus (first half, second half)
    for i in range(N):
        for j in range(N):
            for t_i in spk_ts[i]:
                for t_j in spk_ts[j]:
                    if t_i<n_T2 and t_j<n_T2:
                        # covariance for red stimulus
                        l = t_i - t_j
                        if l>=-L and l<L:
                            i_l = l+n_L2
                            cov[i,j,i_l,0,i_rep] += 1
                    elif t_i>=n_T2 and t_j>n_T2:
                        # covariance for blue stimulus
                        l = t_i - t_j
                        if l>-L and l<L:
                            i_l = l+n_L2
                            cov[i,j,i_l,1,i_rep] += 1
        
    # plot two repetitions to see variability across trials
    if i_rep<n_rep_aff:
        plt.figure(figsize=[6,4])
        for i in range(N):
            # plot PSP
            plt.plot((v_t[0],v_t[-1]), (i,i), c='k', lw=0.5, ls='-')
            ind_t = v_t<=t_stim[0,0] 
            plt.plot(v_t[ind_t], scaling_aff*inst_rate[i,ind_t]+i, c=cols[2], lw=1, ls='--')
            ind_t = np.logical_and(v_t>=t_stim[0,0], v_t<=t_stim[0,1])
            plt.plot(v_t[ind_t], scaling_aff*inst_rate[i,ind_t]+i, c=cols[0], lw=1, ls='--')
            ind_t = np.logical_and(v_t>=t_stim[0,1], v_t<=t_stim[1,0])
            plt.plot(v_t[ind_t], scaling_aff*inst_rate[i,ind_t]+i, c=cols[2], lw=1, ls='--')
            ind_t = np.logical_and(v_t>=t_stim[1,0], v_t<=t_stim[1,1])
            plt.plot(v_t[ind_t], scaling_aff*inst_rate[i,ind_t]+i, c=cols[1], lw=1, ls='--')
            ind_t = v_t>=t_stim[1,1] 
            plt.plot(v_t[ind_t], scaling_aff*inst_rate[i,ind_t]+i, c=cols[2], lw=1, ls='--')
            for i_t in spk_ts[i]:
        #        print(i_t)
                t_spk = v_t[i_t]
                plt.plot((t_spk,t_spk), (i,i+0.3), c=cols[v_t_col[i_t]], lw=1.5)
        plt.yticks(range(N), ['n1','n2','n3'])
        plt.axis(xmin=0, xmax=T, ymin=-0.5, ymax=N)
        plt.xlabel('time (ms)')
        plt.ylabel('neuron responses')
        plt.tight_layout()
        # plt.close()

# calculate metrics: smoothed covariance/correlation, and averages over trial period
smoothed_corr_ts = np.einsum('itk, jtk -> ijtk', smoothed_rate_ts, smoothed_rate_ts)
mask_tri = np.tri(N, N, -1, dtype=bool)

av_rate = smoothed_rate_ts.sum(axis=1)
av_corr = smoothed_corr_ts.sum(axis=2)


#%% plots of smoothed rate and correlation trajectories, average over the 100 trials

plt.figure(figsize=[6,4])
for i in range(N):
    for k in range(n_stim):
        plt.plot(v_t2, smoothed_rate_ts[i,:,k*n_rep:(k+1)*n_rep].mean(axis=1)+i*2, c=cols[k], lw=1, ls='-')
plt.title('smoothed rate average over trials')


plt.figure(figsize=[6,4])
cnt = 0
for i in range(N):
    for j in range(i):
        for k in range(n_stim):
            plt.plot(v_t2, smoothed_corr_ts[i,j,:,k*n_rep:(k+1)*n_rep].mean(axis=1)+cnt*5, c=cols[k], lw=1, ls='-')
        cnt += 1
plt.title('smoothed corr average over trials')


#%% plot of lagged covariances

# red stimulus
plt.figure(figsize=[6,4])
plt.plot(np.arange(-n_L2,n_L2), cov[0,1,:,0,:].sum(axis=-1), c='r', ls='-')
plt.plot(np.arange(-n_L2,n_L2), cov[0,2,:,0,:].sum(axis=-1), c='r', ls='--')
plt.plot(np.arange(-n_L2,n_L2), cov[1,2,:,0,:].sum(axis=-1), c='r', ls=':')
plt.xlabel('time lag (ms)')
plt.ylabel('covar')
plt.tight_layout()

# red stimulus
plt.figure(figsize=[6,4])
plt.plot(np.arange(-n_L2,n_L2), cov[0,1,:,1,:].sum(axis=-1), c='b', ls='-')
plt.plot(np.arange(-n_L2,n_L2), cov[0,2,:,1,:].sum(axis=-1), c='b', ls='--')
plt.plot(np.arange(-n_L2,n_L2), cov[1,2,:,1,:].sum(axis=-1), c='b', ls=':')
plt.xlabel('time lag (ms)')
plt.ylabel('covar')
plt.tight_layout()


#%% plot PCA for the four metrics

# metrics from smoothed spike trains
pca = PCA(n_components=2)

pca_ts_rate = pca.fit_transform(smoothed_rate_ts.reshape(N*n_T2,2*n_rep).T)

pca_ts_corr = pca.fit_transform(smoothed_corr_ts[mask_tri,:,:].reshape(int(N*(N-1)/2)*n_T2,2*n_rep).T)

pca_av_rate = pca.fit_transform(av_rate.reshape(N,2*n_rep).T)

pca_av_corr = pca.fit_transform(av_corr[mask_tri,:].reshape(int(N*(N-1)/2),2*n_rep).T)


plt.figure()
for k in range(2):
    plt.scatter(pca_ts_rate[k*n_rep:(k+1)*n_rep,0], pca_ts_rate[k*n_rep:(k+1)*n_rep,1], color=cols[k])
plt.title('smoothed rate')


plt.figure()
for k in range(2):
    plt.scatter(pca_ts_corr[k*n_rep:(k+1)*n_rep,0], pca_ts_corr[k*n_rep:(k+1)*n_rep,1], color=cols[k])
plt.title('smoothed corr')

plt.figure()
for k in range(2):
    plt.scatter(pca_av_rate[k*n_rep:(k+1)*n_rep,0], pca_av_rate[k*n_rep:(k+1)*n_rep,1], color=cols[k])
plt.title('av rate')


plt.figure()
for k in range(2):
    plt.scatter(pca_av_corr[k*n_rep:(k+1)*n_rep,0], pca_av_corr[k*n_rep:(k+1)*n_rep,1], color=cols[k])
plt.title('av corr')


#%% decoding red versus blue

n_splits = 10
acc = np.zeros([n_splits,4])

labels = np.zeros([2*n_rep], dtype=int)
labels[n_rep:] = 1

sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2)

mlr = LogisticRegression(penalty='l2', C=1.0)

for i_split in range(n_splits):
    for ind_train, ind_test in sss.split(labels, labels):
        # split labels
        y_train, y_test = labels[ind_train], labels[ind_test]
        # split data, train and test
        # smoothed rate
        X_train, X_test = pca_ts_rate[ind_train,:], pca_ts_rate[ind_test,:]
        mlr.fit(X_train, y_train)
        acc[i_split,0] = mlr.score(X_test, y_test)
        # smoothed corr
        X_train, X_test = pca_ts_corr[ind_train,:], pca_ts_corr[ind_test,:]
        mlr.fit(X_train, y_train)
        acc[i_split,1] = mlr.score(X_test, y_test)
        # av rate
        X_train, X_test = pca_av_rate[ind_train,:], pca_av_rate[ind_test,:]
        mlr.fit(X_train, y_train)
        acc[i_split,2] = mlr.score(X_test, y_test)
        # av corr
        X_train, X_test = pca_av_corr[ind_train,:], pca_av_corr[ind_test,:]
        mlr.fit(X_train, y_train)
        acc[i_split,3] = mlr.score(X_test, y_test)

print('decoding accuracies:')
print('mean acc ts rate / ts corr / av rate / av corr:', acc.mean(axis=0))
print('std acc ts rate / ts corr / av rate / av corr:', acc.std(axis=0))


