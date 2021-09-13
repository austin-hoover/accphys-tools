#!/usr/bin/env python
# coding: utf-8

# # SNS injection painting

# In[35]:


import sys
import importlib
import os
from os.path import join

import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm import trange
import proplot as pplt 

sys.path.append('..')
from tools import plotting as myplt
from tools import animation as myanim
from tools import utils
from tools import beam_analysis as ba

plt.rcParams['animation.html'] = 'jshtml'
plt.rcParams['savefig.dpi'] = 'figure'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.alpha'] = 0.04
plt.rcParams['axes.grid'] = False


utils.delete_files_not_folders('_output/figures/')


# ## Matched eigenvector 

# In[36]:


# matched_eigvec = np.load('matched_eigenvector.npy')
# matched_env_params = np.load('matched_env_params.npy')

# eps = 40e-6 # intrinsic emittance [mm mrad]
# n_parts = 10000
# radii = np.sqrt(np.random.uniform(0, 4 * eps, size=(n_parts, 1)))
# phases = np.linspace(0, 2 * np.pi, n_parts).reshape(n_parts, 1) # eigenvector phase [rad]
# eigvecs = np.tile(matched_eigvec, (n_parts, 1))
# X = np.real(radii * eigvecs * np.exp(-1j * phases))

# myplt.corner(1e3 * X, 1e3 * matched_env_params);


# In[37]:


folder = '_output/data/'


# ## Injection region closed orbit

# In[38]:


inj_region_coords_t0 = np.load(join(folder, 'inj_region_coords_t0.npy'))
inj_region_coords_t1 = np.load(join(folder, 'inj_region_coords_t1.npy'))
inj_region_positions_t0 = np.load(join(folder, 'inj_region_positions_t0.npy'))
inj_region_positions_t1 = np.load(join(folder, 'inj_region_positions_t1.npy'))
inj_region_positions_t0 -= 0.5 * inj_region_positions_t0[-1]
inj_region_positions_t1 -= 0.5 * inj_region_positions_t1[-1]
inj_region_coords_t0 *= 1000. # convert to mm-mrad
inj_region_coords_t1 *= 1000. # convert to mm-mrad


# In[39]:


fig, ax = pplt.subplots(figsize=(6, 2.5))
ax.plot(inj_region_positions_t0, inj_region_coords_t0[:, 0])
ax.plot(inj_region_positions_t0, inj_region_coords_t0[:, 2])
ax.format(cycle='colorblind')
ax.plot(inj_region_positions_t1, inj_region_coords_t1[:, 0], ls='--', lw=1)
ax.plot(inj_region_positions_t1, inj_region_coords_t1[:, 2], ls='--', lw=1)
ax.format(title='Injection region closed orbit')
ax.legend(labels=('x (t = 0 ms)', 'y (t = 0 ms)', 'x (t = 1 ms)', 'y (t = 1 ms)'), 
          ncols=1, loc=(1.02, 0), handlelength=1.5);
ax.axvline(0, c='k', lw=0.75, alpha=0.1)
ax.axhline(0, c='k', lw=0.75, alpha=0.1)
ax.format(xlabel='s [m]', ylabel='[mm]') 

# Plot kicker positions
# hkick_positions = [-12.8738290, -10.1838290, 11.13989262, 13.82989262]
# vkick_positions = [-11.7138290, -9.64382900, 10.59989262, 12.66989262]
# for hkick_position in hkick_positions:
#     ax.axvline(hkick_position, color='pink', zorder=0)
# for vkick_position in vkick_positions:
#     ax.axvline(vkick_position, color='grey', zorder=0)
    
plt.savefig('_output/figures/inj_region_closed_orbit.png', facecolor='white', dpi=500)


# ## Kicker strengths

# In[40]:


kicker_angles_t0 = np.loadtxt(folder + 'kicker_angles_t0.dat')
kicker_angles_t1 = np.loadtxt(folder + 'kicker_angles_t1.dat')
kicker_names = ['ikickh_a10', 'ikickv_a10', 'ikickh_a11', 'ikickv_a11',
                'ikickv_a12', 'ikickh_a12', 'ikickv_a13', 'ikickh_a13']


# In[41]:


def waveform(t, k0, k1):
    return k0 - (k0 - k1)*np.sqrt(t)


# In[42]:


t = np.linspace(0, 1, 1000)

fig, axes = pplt.subplots(nrows=4, ncols=2, figsize=(3.5, 6))
for k0, k1, name, ax in zip(kicker_angles_t0, kicker_angles_t1, kicker_names, axes):
    ax.plot(t, 1000 * waveform(t, k0, k1), c='k')
    ax.format(title=name)
axes.format(ylabel='Amplitude', suptitle='Kicker angles', xlabel='time [ms]')
plt.savefig('_output/figures/kicker_angles.png', facecolor='w', dpi=500)


# In[43]:


fig, axes = pplt.subplots(nrows=4, ncols=2, figsize=(3.5, 6))
for k0, k1, name, ax in zip(kicker_angles_t0, kicker_angles_t1, kicker_names, axes):
    ax.plot(t, waveform(t, 1.0, k1/k0), c='k')
    ax.format(title=name)
axes.format(ylabel='Amplitude', suptitle='Kicker waveforms', xlabel='time [ms]')
plt.savefig('_output/figures/kicker_waveforms.png', facecolor='w', dpi=500)


# ## Beam statistics

# In[44]:


coords = utils.load_stacked_arrays(join(folder, 'coords.npz'))


# In[45]:


moments_list = []
for X in tqdm(coords):
    Sigma = np.cov(X[:, :4].T)
    moments_list.append(ba.mat2vec(Sigma))
moments_list = np.array(moments_list)
    
stats = ba.StatsReader()
stats.read_moments(moments_list)


# In[46]:


fig, ax = pplt.subplots(figsize=(3.5, 2.5))
plt_kws = dict(legend=False)
stats.twiss2D[['eps_x','eps_y']].plot(ax=ax, **plt_kws)
stats.twiss4D[['eps_1','eps_2']].plot(ax=ax, **plt_kws)
ax.legend(labels=[r'$\varepsilon_{}$'.format(v) for v in ['x', 'y', '1', '2']], 
          ncols=1, loc='upper left')
ax.format(ylabel='[mm mrad]', xlabel='Turn number', grid=True);
plt.savefig('_output/figures/emittances.png', facecolor='w', dpi=300)


# In[47]:


exey = (stats.twiss2D['eps_x'] * stats.twiss2D['eps_y']).values
e1e2 = (stats.twiss4D['eps_1'] * stats.twiss4D['eps_2']).values

fig, ax = pplt.subplots(figsize=(3.5, 2.5))
ax.plot(e1e2, color='red8')
ax.plot(exey, color='blue8')
ax.legend(labels=[r'$\varepsilon_1\varepsilon_2$', r'$\varepsilon_x\varepsilon_y$'],
          ncols=1, loc='upper left')
ax.format(xlabel='Turn number', ylabel=r'[mm$^2$ mrad$^2$]', 
          grid=True)
plt.savefig('_output/figures/emittances_4D.png', facecolor='w', dpi=300)


# In[48]:


fig, axes = pplt.subplots(nrows=2, figsize=(3.5, 3.5))
stats.twiss2D[['eps_x','eps_y']].plot(ax=axes[0], **plt_kws)
stats.twiss4D[['eps_1','eps_2']].plot(ax=axes[0], **plt_kws)
axes[1].plot(e1e2, color='red8')
axes[1].plot(exey, color='blue8')
axes[0].legend(labels=[r'$\varepsilon_{}$'.format(v) for v in ['x', 'y', '1', '2']], 
               ncols=1, loc='upper left')
axes[1].legend(labels=[r'$\varepsilon_1\varepsilon_2$', r'$\varepsilon_x\varepsilon_y$'],
               ncols=1, loc='upper left')
axes[0].format(ylabel='[mm mrad]', xlabel='Turn number');
for ax in axes:
    ax.grid(axis='y')
plt.savefig('_output/figures/emittances_combined.png', facecolor='w', dpi=300)


# In[49]:


fig, ax = pplt.subplots(figsize=(3.5, 2.5))
ax.plot(1.0 - np.sqrt((e1e2) / (exey)), c='k')
ax.format(xlabel='Turn number', 
          title=r'C = 1 - $\sqrt{\frac{\varepsilon_1\varepsilon_2}{\varepsilon_x\varepsilon_y}}$', 
          grid=True)
plt.savefig('_output/figures/coupling_factor.png', facecolor='w', dpi=300)


# In[50]:


fig, axes = pplt.subplots(nrows=3, figsize=(3.5, 5.0), spany=False, aligny=True)
columns = (['beta_x','beta_y'], ['alpha_x','alpha_y'], ['eps_x','eps_y'])
ylabels = (r'$\beta$ [m]', r'$\alpha$ [rad]', r'$\varepsilon$ [mm $\cdot$ mrad]')
for ax, col in zip(axes, columns):
    stats.twiss2D[col].plot(ax=ax, **plt_kws)
axes.format(xlabel='Turn number', grid=True)
axes[0].format(title='2D Twiss parameters')
myplt.set_labels(axes, ylabels, 'ylabel')
plt.savefig('_output/figures/twiss2D.png', facecolor='w', dpi=300)


# In[51]:


fig, axes = pplt.subplots(nrows=2, figsize=(3.5, 3.33), spany=False, aligny=True)
stats.twiss4D['u'].plot(color='k', ax=axes[0], **plt_kws)
stats.twiss4D['nu'].plot(color='k', ax=axes[1], **plt_kws)
axes.format(grid=True)
axes[0].format(ylabel='u')
axes[1].format(ylabel=r'$\nu$', yformatter='deg')
plt.savefig('_output/figures/u_and_nu.png', facecolor='w', dpi=300)



# ## TBT coordinates 

# In[54]:


X = coords[-1]
fig, ax = pplt.subplots(figsize=(4, 1.5))
ax.hist(X[:, 4], histtype='stepfilled', bins='auto', color='black')
ax.set_xlabel("z [m]")
ax.set_xlim(-248/2, 248/2)
plt.savefig('_output/figures/z.png', facecolor='white', dpi=300)


# In[55]:


fig, ax = pplt.subplots(figsize=(4, 1.5))
ax.hist(X[:, 5], histtype='stepfilled', bins='auto', color='black')
ax.set_xlabel(r"$\delta$E")
plt.savefig('_output/figures/dE.png', facecolor='white', dpi=300)


# In[56]:


foil_pos = (46.8, 49.2)
coords_foil_frame = []
for X in coords:
    Y = np.copy(X)
    Y[:, 0] -= foil_pos[0]
    Y[:, 2] -= foil_pos[1]
    coords_foil_frame.append(Y)
    
    
    
import seaborn as sns

sns.pairplot(pd.DataFrame(coords_foil_frame[-0], columns=['x', 'xp', 'y', 'yp', 'z', 'dE']), 
             kind='hist', corner=True, plot_kws=dict(cmap='binary'), diag_kws=dict(color='k'))
plt.savefig('_output/figures/init_dist.png', facecolor='white', dpi=300)

sns.pairplot(pd.DataFrame(coords_foil_frame[-1], columns=['x', 'xp', 'y', 'yp', 'z', 'dE']), 
             kind='hist', corner=True, plot_kws=dict(cmap='binary'), diag_kws=dict(color='k'))
plt.savefig('_output/figures/final_dist.png', facecolor='white', dpi=300)


# In[57]:


limits = ((-90, 35), (-6, 6)) # ((x_min, x_max), (y_min, y_max))
turn = -1
axes = myplt.corner(
    coords_foil_frame[turn][:, :4], 
    limits=None, 
    zero_center=False,
    pad=0.1,
    samples=50000,
    diag_kws=dict(color='k'),
    text='Turn {}'.format(turn),
    kind='hist', cmap='dusk_r',
#     kind='scatter', s=10,
)
# for i in range(1, 4):
#     for j in range(i):
#         ax = axes[i, j]
#         ax.scatter(0, 0, c='r', s=10, zorder=99)
#         axes[i, j].axvline(0, c='w', lw=0.75, alpha=0.04, zorder=0)
#         axes[i, j].axhline(0, c='w', lw=0.75, alpha=0.04, zorder=0)
plt.savefig('_output/figures/corner.png', facecolor='w', dpi=300)


# In[58]:


X = coords[-1]
h, _ = np.histogram(X[:, 0], bins='auto')
bins = int(len(h))


# In[59]:


anim = myanim.corner(
    coords_foil_frame, figsize=7, skip=9, keep_last=True,
#     limits=[(-85, 25), (-5, 5), (-55, 55), (-5, 5)],
    zero_center=False, pad=0,
    samples=len(X), 
    text_fmt='Turn {}', 
    diag_kws=dict(color='black'), 
#     kind='scatter', color='black', ms=0.3,
    kind='hist', cmap='dusk_r', bins=bins
    )
anim.save('_output/figures/corner.mp4', dpi=350, fps=10)



i = 299
X_onepart = np.array([X[i, :4] for X in coords_foil_frame])

axes = myplt.corner(X_onepart, zero_center=False)
plt.savefig('_output/figures/corner_part{}.png'.format(i), facecolor='w', dpi=300)

# In[63]:


anim = myanim.corner_onepart(
    X_onepart[:50], show_history=True, skip=0, pad=0.35, text_fmt='Turn = {}', 
    zero_center=False,
)
anim.save('_output/figures/corner_part{}.mp4'.format(i), dpi=350, fps=5)

