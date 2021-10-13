#!/usr/bin/env python
# coding: utf-8

# # SNS injection painting

# In[ ]:


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
import seaborn as sns

sys.path.append('..')
from tools import animation as myanim
from tools import beam_analysis as ba
from tools import plotting as myplt
from tools import utils


# ## Settings

# In[ ]:


plt.rcParams['animation.html'] = 'jshtml'
plt.rcParams['savefig.dpi'] = 'figure'
plt.rcParams['savefig.transparent'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.alpha'] = 0.04
plt.rcParams['axes.grid'] = False
savefig_kws = dict(facecolor='white', dpi=300)
n_bins_hist2d = 50


# In[ ]:


folder = '_output/data/'
location = 'injection point' # {'injection point', 'rtbt entrance'}


# In[ ]:


utils.delete_files_not_folders('_output/figures/')


# ## Matched eigenvector 

# In[ ]:


# matched_eigvec = np.load('matched_eigenvector.npy')
# matched_env_params = np.load('matched_env_params.npy')

# eps = 40e-6 # intrinsic emittance [mm mrad]
# n_parts = 10000
# radii = np.sqrt(np.random.uniform(0, 4 * eps, size=(n_parts, 1)))
# phases = np.linspace(0, 2 * np.pi, n_parts).reshape(n_parts, 1) # eigenvector phase [rad]
# eigvecs = np.tile(matched_eigvec, (n_parts, 1))
# X = np.real(radii * eigvecs * np.exp(-1j * phases))

# myplt.corner(1e3 * X, 1e3 * matched_env_params);


# ## Injection region closed orbit

# In[ ]:


inj_region_coords_t0 = np.load(join(folder, 'inj_region_coords_t0.npy'))
inj_region_coords_t1 = np.load(join(folder, 'inj_region_coords_t1.npy'))
inj_region_positions_t0 = np.load(join(folder, 'inj_region_positions_t0.npy'))
inj_region_positions_t1 = np.load(join(folder, 'inj_region_positions_t1.npy'))
inj_region_coords_t0 *= 1000. # convert to mm-mrad
inj_region_coords_t1 *= 1000. # convert to mm-mrad


# In[ ]:


fig, ax = pplt.subplots(figsize=(6, 2.5))
ax.plot(inj_region_positions_t0, inj_region_coords_t0[:, 0], label='x (initial)')
ax.plot(inj_region_positions_t0, inj_region_coords_t0[:, 2], label='y (initial)')
ax.format(cycle='colorblind')
ax.plot(inj_region_positions_t1, inj_region_coords_t1[:, 0], ls='--', lw=1, label='x (final)')
ax.plot(inj_region_positions_t1, inj_region_coords_t1[:, 2], ls='--', lw=1, label='y (final)')
ax.format(title='Injection region closed orbit')
ax.legend(ncols=1, loc=(1.02, 0), handlelength=1.5);
ax.format(xlabel='s [m]', ylabel='[mm]') 
ax.grid(axis='y')
plt.savefig('_output/figures/inj_region_closed_orbit.png', **savefig_kws)


# ## Kicker strengths

# In[ ]:


# def get_perveance(kin_energy, mass, line_density):
#     classical_proton_radius = 1.53469e-18 # m
#     gamma = 1 + (kin_energy / mass) # Lorentz factor
#     beta = np.sqrt(1 - (1 / gamma)**2) # velocity/speed_of_light
#     return (2 * classical_proton_radius * line_density) / (beta**2 * gamma**3)

# ring_length = 248.0
# zlim = (135.0 / 360.0) * ring_length
# zmin, zmax = -zlim, zlim
# bunch_length = zmax - zmin

# # Production beam
# kin_energy = 1.0 # [GeV]
# mass = 0.93827231 # [kg]
# intensity = 1.5e14
# Q = get_perveance(kin_energy, mass, intensity / bunch_length)
# xmax = ymax = 26.0
# area = xmax * ymax
# print('Q = {}'.format(Q))
# density_production = Q / area
# print('Q / area = {}'.format(density_production))

# # SCBD
# kin_energy = 1.0 # [GeV]
# mass = 0.93827231 # [kg]
# max_intensity = 1.5e14
# max_n_turns = 1000.
# xmax = ymax = 26.0
# area = xmax * ymax
# densities = []
# turns = np.linspace(1, max_n_turns, 1000)
# for t in turns:
#     tau = t / max_n_turns
#     intensity = max_intensity * tau
#     Q = get_perveance(kin_energy, mass, intensity / bunch_length)
#     densities.append(Q / area)
# densities = np.array(densities)
    
# fig, ax = pplt.subplots()
# ax.plot(turns, densities, color='black');
# ymin, ymax = ax.get_ylim()
# alpha = 0.15
# ax.fill_between(turns, 0., density_production, color='green', alpha=alpha)
# ax.fill_between(turns, density_production, ymax, color='red', alpha=alpha)
# ax.format(grid=True, ylim=(0, ax.get_ylim()[1]),
#           xlabel='# turns to reach full beam size', ylabel=r'Q / area [mm$^{-2}$]')
# plt.savefig('_output/figures/perveance_scaling_{}_{}.png'.format(kin_energy, xmax))


# In[ ]:


kicker_angles_t0 = np.loadtxt(folder + 'kicker_angles_t0.dat')
kicker_angles_t1 = np.loadtxt(folder + 'kicker_angles_t1.dat')
kicker_names = ['ikickh_a10', 'ikickv_a10', 'ikickh_a11', 'ikickv_a11',
                'ikickv_a12', 'ikickh_a12', 'ikickv_a13', 'ikickh_a13']


# In[ ]:


def waveform(t, k0, k1):
    return k0 - (k0 - k1)*np.sqrt(t)


# In[ ]:


t = np.linspace(0, 1, 1000)

fig, axes = pplt.subplots(nrows=4, ncols=2, figsize=(3.5, 6))
for k0, k1, name, ax in zip(kicker_angles_t0, kicker_angles_t1, kicker_names, axes):
    ax.plot(t, 1000 * waveform(t, k0, k1), c='k')
    ax.format(title=name)
axes.format(ylabel='Amplitude', suptitle='Kicker angles', xlabel='time [ms]')
plt.savefig('_output/figures/kicker_angles.png', **savefig_kws)


# In[ ]:


fig, axes = pplt.subplots(nrows=4, ncols=2, figsize=(3.5, 6))
for k0, k1, name, ax in zip(kicker_angles_t0, kicker_angles_t1, kicker_names, axes):
    ax.plot(t, waveform(t, 1.0, k1/k0), c='k')
    ax.format(title=name)
axes.format(ylabel='Amplitude', suptitle='Kicker waveforms', xlabel='time [ms]')
plt.savefig('_output/figures/kicker_waveforms.png', **savefig_kws)


# ## Beam statistics

# In[ ]:


suffix = ''
if location == 'rtbt entrance':
    suffix = '_rtbt_entrance'
filename = 'coords{}.npz'.format(suffix)
coords = utils.load_stacked_arrays(join(folder, filename))
for i in trange(len(coords)):
    coords[i][:, 5] *= 1000. # convert dE to [MeV]


# In[ ]:


moments_list = []
for X in tqdm(coords):
    Sigma = np.cov(X[:, :4].T)
    moments_list.append(ba.mat2vec(Sigma))
moments_list = np.array(moments_list)
    
stats = ba.StatsReader()
stats.read_moments(moments_list)


# In[ ]:


fig, ax = pplt.subplots(figsize=(3.5, 2.5))
plt_kws = dict(legend=False)
stats.twiss2D[['eps_x','eps_y']].plot(ax=ax, **plt_kws)
stats.twiss4D[['eps_1','eps_2']].plot(ax=ax, **plt_kws)
ax.legend(labels=[r'$\varepsilon_{}$'.format(v) for v in ['x', 'y', '1', '2']], 
          ncols=1, loc='upper left')
ax.format(ylabel='[mm mrad]', xlabel='Turn number', grid=True);
plt.savefig('_output/figures/emittances.png', **savefig_kws)


# In[ ]:


exey = (stats.twiss2D['eps_x'] * stats.twiss2D['eps_y']).values
e1e2 = (stats.twiss4D['eps_1'] * stats.twiss4D['eps_2']).values

fig, ax = pplt.subplots(figsize=(3.5, 2.5))
g1 = ax.plot(e1e2, color='red')
g2 = ax.plot(exey, color='blue')
ax.legend([g1, g2], labels=[r'$\varepsilon_1\varepsilon_2$', r'$\varepsilon_x\varepsilon_y$'],
          ncols=1, loc='upper left')
ax.format(xlabel='Turn number', ylabel=r'[mm$^2$ mrad$^2$]', 
          grid=True)
plt.savefig('_output/figures/emittances_4D.png', **savefig_kws)


# In[ ]:


fig, axes = pplt.subplots(nrows=2, figsize=(4.0, 4.0), spany=False)
g1 = axes[0].plot(stats.twiss2D['eps_x'])
g2 = axes[0].plot(stats.twiss2D['eps_y'])
g3 = axes[0].plot(stats.twiss4D['eps_1'])
g4 = axes[0].plot(stats.twiss4D['eps_2'])
axes[0].legend(handles=[g1, g2, g3, g4],
               labels=[r'$\varepsilon_{}$'.format(v) for v in ['x', 'y', '1', '2']],
               ncols=1, loc='r')
g1 = axes[1].plot(e1e2, color='red')
g2 = axes[1].plot(exey, color='blue8')
axes[1].legend(handles=[g1, g2],
               labels=[r'$\varepsilon_1\varepsilon_2$', r'$\varepsilon_x\varepsilon_y$'],
               ncols=1, loc='r')
axes[0].format(ylabel='[mm mrad]', xlabel='Turn number')
axes[1].format(ylabel=r'[mm$^2$ mrad$^2$]')
for ax in axes:
    ax.grid(axis='y')
plt.savefig('_output/figures/emittances_combined.png', **savefig_kws)


# In[ ]:


fig, ax = pplt.subplots(figsize=(3.5, 2.5))
C = 1.0 - (e1e2) / (exey)
ax.plot(C, c='k')
ax.format(xlabel='Turn number', 
          title=r'C = 1 - $\frac{\varepsilon_1\varepsilon_2}{\varepsilon_x\varepsilon_y}$', 
          grid=True)
plt.savefig('_output/figures/coupling_factor.png', **savefig_kws)


# In[ ]:


fig, axes = pplt.subplots(nrows=3, figsize=(3.5, 5.0), spany=False, aligny=True)
columns = (['beta_x','beta_y'], ['alpha_x','alpha_y'], ['eps_x','eps_y'])
ylabels = (r'$\beta$ [m]', r'$\alpha$ [rad]', r'$\varepsilon$ [mm mrad]')
for ax, col in zip(axes, columns):
    stats.twiss2D[col].plot(ax=ax, **plt_kws)
axes.format(xlabel='Turn number', grid=True)
axes[0].format(title='2D Twiss parameters')
myplt.set_labels(axes, ylabels, 'ylabel')
plt.savefig('_output/figures/twiss2D.png', **savefig_kws)


# In[ ]:


fig, axes = pplt.subplots(nrows=2, figsize=(3.5, 3.33), spany=False, aligny=True)
stats.twiss4D['u'].plot(color='k', ax=axes[0], **plt_kws)
stats.twiss4D['nu'].plot(color='k', ax=axes[1], **plt_kws)
axes.format(grid=True)
axes[0].format(ylabel='u')
axes[1].format(ylabel=r'$\nu$', yformatter='deg')
plt.savefig('_output/figures/u_and_nu.png', **savefig_kws)


# In[ ]:


stats.twiss2D.head()


# In[ ]:


stats.twiss4D.head()


# ## Tunes 

# In[ ]:


# mass = 0.93827231 # [GeV/c^2]
# kin_energy = 0.8 # [GeV]
# alpha_x = 0.06951453814317858
# alpha_y = 0.01091131703789978
# beta_x = 12.243573284689077
# beta_y = 12.030511575868042
# tune_calc = ba.TuneCalculator(mass, kin_energy, alpha_x, alpha_y, beta_x, beta_y)


# In[ ]:


# tunes_list = []
# # turns = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499]
# turns = [0, 5, 9]
# for t in tqdm(turns):
#     tunes = tune_calc.get_tunes(coords[t - 1], coords[t])
#     tunes_list.append(tunes)


# In[ ]:


# tunes = np.copy(tunes_list[2])
# tunes += 6.0
# tunes[np.where(tunes > 6.5)] -= 1.
# lim = (5.9, 6.1)
# g = sns.jointplot(
#     x=tunes[:, 0], y=tunes[:, 1],
#     xlim=lim, ylim=lim, height=4.0,
#     kind='hist', 
#     joint_kws=dict(cmap='binary'),
#     marginal_kws=dict(ec=None, color='black', bins='auto')
# )
# plt.show()


# In[ ]:


fig, ax = pplt.subplots(figsize=(5, 2))
ax.hist(coords[-1][:, 4], bins='auto', color='black');
ax.set_xlim(-248/2, 248/2)


# In[ ]:


# twiss = np.load('twiss.npy')
# df = pd.DataFrame(twiss, columns=['s', 'nux', 'nuy', 'alpha_x', 'alpha_y', 'beta_x', 'beta_y'])
# df
# fig, ax = pplt.subplots(figsize=(12, 3))
# ax.plot(s, beta_x)
# ax.plot(s, beta_y)
# plt.show()


# ## TBT coordinates 

# In[ ]:


foil_pos = (48.6, 46.0)
coords_foil_frame = []
for X in coords:
    Y = np.copy(X)
    Y[:, 0] -= foil_pos[0]
    Y[:, 2] -= foil_pos[1]
    coords_foil_frame.append(Y)


# In[ ]:


turn = -1
pad = -0.1
bins = 75
hist_height_frac = 0.6
plot_kws = dict(bins=bins, pad=pad, hist_height_frac=hist_height_frac)


# In[ ]:


axes = myplt.corner(coords_foil_frame[-1][:, :4], **plot_kws)
plt.savefig('_output/figures/corner4D_turn{}.png'.format(turn), **savefig_kws)


# In[ ]:


axes = myplt.corner(coords_foil_frame[-1], **plot_kws)
plt.close()
limits = [ax.get_xlim() for ax in axes[-1, :]]
limits[4] = (-248/2, 248/2)
axes = myplt.corner(coords_foil_frame[-1], limits=limits, **plot_kws)
plt.savefig('_output/figures/corner6D_turn{}.png'.format(turn), **savefig_kws)


# In[ ]:


anim_kws = dict(skip=19, keep_last=True, text_fmt='Turn = {}', 
                limits=limits, 
                bins=bins, pad=pad, hist_height_frac=hist_height_frac)


# In[ ]:


anim = myanim.corner(coords_foil_frame, dims=4, **anim_kws)
anim.save('_output/figures/corner4D.mp4', dpi=350, fps=5)


# In[ ]:


anim = myanim.corner(coords_foil_frame, dims=6, **anim_kws)
anim.save('_output/figures/corner6D.mp4', dpi=350, fps=5)


# In[ ]:


i = 0
X_onepart = np.array([X[i, :] for X in coords_foil_frame])

axes = myplt.corner(X_onepart, kind='scatter', c='steelblue', pad=0.1)
plt.savefig('_output/figures/corner_part{}.png'.format(i), **savefig_kws)


# In[ ]:


# anim = myanim.corner_onepart(
#     X_onepart[:50], show_history=True, skip=0, pad=0.35, text_fmt='Turn = {}', 
#     zero_center=False, history_kws=dict(ms=5, color='lightgrey'),
# )
# anim.save('_output/figures/corner_part{}.mp4'.format(i), dpi=350, fps=5)

