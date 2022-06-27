import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse


plot_kws = dict(fill=False, lw=0.75, linestyle='--', alpha=0.5)
a = 0.1
n_steps = 6

times = np.linspace(0.0, 1.0, n_steps)

inj_kws = dict(marker='.', color='red8', ms=9, zorder=99)



# Anti-correlated painting
fig, ax = pplt.subplots(figsize=(3, 3))
xmax = 1.5
ax.format(xlim=(-xmax, xmax), ylim=(-xmax, xmax), aspect=1)

for i in range(n_steps):
    t = times[i]
    x = np.sqrt(t)
    y = np.sqrt(1.0 - t)
    width = a + 2 * x
    height = a + 2 * y
    rectangle = Rectangle((-0.5 * width, -0.5 * height), width, height, **plot_kws)
    ax.add_patch(rectangle)

    ax.plot(x, y, **inj_kws)

ellipse = Ellipse((0., 0.), 2.0 + a, 2.0 + a, alpha=0.2)
ax.add_patch(ellipse)

arrow_length = 1.2
arrow_kws = dict(width=0.01, head_width=0.075, color='black', zorder=0)
# ax.arrow(0., 0., arrow_length, 0., **arrow_kws)
# ax.arrow(0., 0., 0., arrow_length, **arrow_kws)
# ax.annotate(r'$x_{co}$', xy=(1.35, -0.15), fontsize=12)
# ax.annotate(r'$y_{co}$', xy=(-0.15, 1.35), fontsize=12)

myplt.despine(ax, 'all')
pplt.rc['savefig.transparent'] = True
plt.savefig('_output/figures/paint_anticorrelated.png', **savefig_kws)



# Correlated painting
fig, ax = pplt.subplots(figsize=(3, 3))
xmax = 1.5
ax.format(xlim=(-xmax, xmax), ylim=(-xmax, xmax), aspect=1)
for t in times:
    x = np.sqrt(t)
    y = np.sqrt(t)
    width = a + 2 * x
    height = a + 2 * y
    rectangle = Rectangle((-0.5 * width, -0.5 * height), width, height, **plot_kws)
    ax.add_patch(rectangle)
    
    ax.plot(x, y, **inj_kws)
    
rectangle = Rectangle((-1.0 - 0.5 * a, -1.0 - 0.5 * a), 2.0 + a, 2.0 + a, alpha=0.2)
ax.add_patch(rectangle)

# ax.arrow(0., 0., arrow_length, 0., **arrow_kws)
# ax.arrow(0., 0., 0., arrow_length, **arrow_kws)
# ax.annotate(r'$x_{co}$', xy=(1.35, -0.15), fontsize=12)
# ax.annotate(r'$y_{co}$', xy=(-0.15, 1.35), fontsize=12)

myplt.despine(ax, 'all')
plt.savefig('_output/figures/paint_correlated.png', **savefig_kws)





# Anti-correlated painting
fig, ax = pplt.subplots(figsize=(3, 3))
xmax = 1.5
ax.format(xlim=(-xmax, xmax), ylim=(-xmax, xmax), aspect=1)
for t in times:
    x = np.sqrt(t)
    y = np.sqrt(t)
    width = a + 2 * x
    height = a + 2 * y
    ellipse = Ellipse((0., 0.), width, height, **plot_kws)
    ax.add_patch(ellipse)
    
    ax.plot(x, 0., **inj_kws)
    
ellipse = Ellipse((0., 0.), 2.0 + a, 2.0 + a, alpha=0.2)
ax.add_patch(ellipse)


ax.arrow(1.0, 0.0, -1.0, 0.0, color='black', head_width=0.05, length_includes_head=True, zorder=999)
# arrow_kws['zorder'] = 0
# x = y = 0.
# ax.arrow(x, y, arrow_length, 0., **arrow_kws)
# ax.arrow(x, y, 0., arrow_length, **arrow_kws)
# ax.annotate(r'$x_{co}$', xy=(1.35, -0.15), fontsize=12)
# ax.annotate(r'$y_{co}$', xy=(-0.15, 1.35), fontsize=12)

foil_width = 0.15
x_foil = 1.0
rectangle = Rectangle((x_foil - 0.5 * foil_width + 0.5 * a, -a/2), foil_width, 1.0, color='darkgoldenrod', alpha=0.4)
ax.add_patch(rectangle)

myplt.despine(ax, 'all')
plt.savefig('_output/figures/paint_elliptical.png', **savefig_kws)