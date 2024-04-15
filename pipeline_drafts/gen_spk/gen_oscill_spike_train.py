#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:38:14 2020

@author: matgilson
"""

# generating spikes from oscillatory instantaneous rate profiles

import numpy as np
import matplotlib.pyplot as plt

# time parameters
T = 70.
dt = 1.
v_t = np.arange(0,T,dt)
n_T = v_t.size

N = 3 # 3 neurons

f = 2*np.pi/20 # 100 Hz
a = 0.35

inst_rate = np.zeros([N,n_T])
for i in range(N):
    inst_rate[i,:] = a * (1+np.cos(f*v_t))


frmt_grph = 'eps'
cols = ['r','b','gray']


plt.figure(figsize=[3,2])
for i in range(N):
    # plot PSP
    plt.plot((v_t[0],v_t[-1]), (i,i), c='k', lw=0.5, ls='-')
    plt.plot(v_t, inst_rate[i,:]+i, c=cols[-1], lw=1, ls='--')
    # gen spike train
    spk_ts = np.argwhere(np.random.rand(n_T)<inst_rate[i,:]*dt*0.2).flatten()
    for i_t in spk_ts:
#        print(i_t)
        t_spk = v_t[i_t]
        plt.plot((t_spk,t_spk), (i,i+0.3), c=cols[-1], lw=1.5)

plt.yticks(range(N), ['n1','n2','n3'])
plt.axis(xmin=0, xmax=T, ymin=-0.5, ymax=N)
plt.xlabel('time (ms)')
plt.ylabel('neuron responses')
plt.tight_layout()


