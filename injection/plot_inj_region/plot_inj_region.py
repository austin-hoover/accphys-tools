# python3 -m pip install proplot
from os.path import join
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import proplot as pplt

pplt.rc['grid'] = False


folder = "data/"

# Load the closed orbit trajectory.
inj_region_coords_t0 = np.load(join(folder, "inj_region_coords_t0.npy"))
inj_region_coords_t1 = np.load(join(folder, "inj_region_coords_t1.npy"))
inj_region_positions_t0 = np.load(join(folder, "inj_region_positions_t0.npy"))
inj_region_positions_t1 = np.load(join(folder, "inj_region_positions_t1.npy"))
inj_region_coords_t0 *= 1000.0  # convert to mm-mrad
inj_region_coords_t1 *= 1000.0  # convert to mm-mrad

# Load kicker angles and names.
kicker_angles_t0 = np.loadtxt(join(folder, "kicker_angles_t0.dat"))
kicker_angles_t1 = np.loadtxt(join(folder, "kicker_angles_t1.dat"))
kicker_names = [
    "ikickh_a10",
    "ikickv_a10",
    "ikickh_a11",
    "ikickv_a11",
    "ikickv_a12",
    "ikickh_a12",
    "ikickv_a13",
    "ikickh_a13",
]

# Load node positions.
file = open(join(folder, 'injection_region_node_positions.txt'), 'r')
names, positions, widths = [], [], []
for line in file:
    name, start, stop = line.rstrip().split(" ")
    names.append(name)
    start = float(start)
    stop = float(stop)
    positions.append(0.5 * (start + stop))
    widths.append(stop - start)
file.close()


def lens(pos, width, height, foc=True, **kws):
    """Return diamond representing a quadrupole lens."""
    if foc:
        coords = [
            (pos - 0.5 * width, -0.5 * height),
            (pos + 0.5 * width, -0.5 * height),
            (pos - 0.5 * width, +0.5 * height),
            (pos + 0.5 * width, +0.5 * height),
        ]
    else:
        coords = [
            (pos - 0.5 * width, 0.0),
            (pos, -0.5 * height),
            (pos + 0.5 * width, 0.0),
            (pos, +0.5 * height),
        ]
    return patches.Polygon(coords, **kws)


def is_quad(name):
    return name.startswith("q") and "sc" not in name


def is_kicker(name):
    return name.startswith("ikick")


def is_corrector(name):
    return "dchv" in name or "dmcv" in name


def is_foil(name):
    return name == "inj_mid"


def get_number(name):
    for i, character in enumerate(name):
        if character.isnumeric():
            return int(name[i:])


start = names.index("inj_start")
stop = names.index("inj_end")
names = names[start:] + names[:stop]
positions = list(np.subtract(positions[start:], positions[-1])) + positions[:stop]
widths = widths[start:] + widths[:stop]

fig, axes = pplt.subplots(nrows=2, figsize=(6.0, 3), height_ratios=(1.0, 0.35), spany=False)
pad = 2.0
axes.format(
    xlim=(min(positions) - pad, max(positions) + pad),
    xlabel="Distance from injection point [m]",
)
axes[0].format(ylabel='[mm]')

# Lattice elements
height = 0.4
ytextpos = 0.28
ax = axes[1]
colorblind = pplt.Cycle('colorblind').by_key()['color']
colors = {
    "quad": "black",
    "hkicker": colorblind[0],
    "vkicker": colorblind[1],
    "foil": "gold",
    "vcorrector": colorblind[2],
}
for position, name, width in zip(positions, names, widths):
    if is_quad(name):
        focusing = "h" in name
        patch = lens(position, width, height, focusing, color=colors["quad"])
    elif is_kicker(name):
        if "h" in name:
            color = colors["hkicker"]
        else:
            color = colors["vkicker"]
        patch = patches.Rectangle(
            (position - 0.5 * width, -0.5 * height), width, height, color=color
        )
    elif is_corrector(name):
        color = colors["vcorrector"]
        width = 0.04
        patch = patches.Rectangle(
            (position - 0.5 * width, -0.5 * height), width, height, color=color
        )
    elif is_foil(name):
        color = colors["foil"]
        width = 0.02
        patch = patches.Rectangle(
            (position - 0.5 * width, -0.5 * height), width, height, color=color
        )
    else:
        continue
    ax.add_patch(patch)
ax.axhline(0.0, color="black", alpha=0.1, lw=0.2)
ax.format(ylim=(-height, height), yticks=[])

# Closed orbit.
ax = axes[0]
_pos = np.linspace(positions[0], positions[-1], len(inj_region_coords_t0))
ax.plot(_pos, inj_region_coords_t0[:, 0], label="x (initial)")
ax.plot(_pos, inj_region_coords_t0[:, 2], label="y (initial)")
ax.format(cycle="colorblind")
ax.plot(_pos, inj_region_coords_t1[:, 0], ls="--", lw=1, label="x (final)")
ax.plot(_pos, inj_region_coords_t1[:, 2], ls="--", lw=1, label="y (final)")
ax.legend(ncols=1, loc=(1.02, 0), handlelength=1.5, framealpha=0)
plt.savefig('inj_region_elements')