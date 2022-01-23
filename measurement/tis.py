"""Analysis of beam images from the SNS target imaging system."""
import collections
from datetime import datetime
import sys

import numpy as np
from scipy import optimize as opt
from skimage import filters
from skimage import measure


PIXEL_WIDTH = 1.0 / 1.77 # [mm]


def get_edges(centers):
    width = np.diff(centers)[0]
    return np.hstack([centers - 0.5 * width, [centers[-1] + 0.5 * width]])


class TargetImage:

    def __init__(self, Z):
        self.Z = Z
        self.Zf = None
        self.n_rows, self.n_cols = Z.shape
        self.xx = np.array(list(range(self.n_rows))).astype(float)
        self.yy = np.array(list(range(self.n_cols))).astype(float)
        self.xx -= np.mean(self.xx)
        self.yy -= np.mean(self.yy)
        self.width = abs(self.xx[-1] - self.xx[0])
        self.height = abs(self.yy[-1] - self.yy[0])
        self.pixel_width = PIXEL_WIDTH
        self.set_pixel_width(PIXEL_WIDTH)
        
        self.xedges = get_edges(self.xx)
        self.yedges = get_edges(self.yy)
                
        self.X, self.Y = np.meshgrid(self.xx, self.yy)
        self.Zfit = None
        self.cov = None
        self.mean_x, self.mean_y = None, None
        self.cx, self.cy, self.angle = None, None, None
        
    def set_pixel_width(self, pixel_width):
        """Needs to be modified."""
        if type(pixel_width) in [int, float]:
            pixel_width = [pixel_width, pixel_width]
        self.xx *= pixel_width[0]
        self.yy *= pixel_width[1]
        self.xedges = get_edges(self.xx)
        self.yedges = get_edges(self.yy)
        
    def filter(self, sigma, **kws):
        self.Zf = filters.gaussian(self.Z, sigma=sigma, **kws)
        return self.Zf
            
    def fit_gauss2d(self, use_filtered=False):
        Z = self.Zf if use_filtered else self.Z 
        Zfit, params = fit_gauss2d(self.X, self.Y, Z.T)
        self.Zfit = Zfit.T
        sig_xx, sig_yy, sig_xy, mean_x, mean_y, amp = params
        self.c1, self.c2, self.angle = rms_ellipse_dims(sig_xx, sig_yy, sig_xy)
        self.cov = np.array([[sig_xx, sig_xy], [sig_xy, sig_yy]])
        self.mean_x = mean_x
        self.mean_y = mean_y
        
    def estimate_moments(self, use_filtered=False):
        Z = np.copy(self.Zf) if use_filtered else np.copy(self.Z)
        coords = np.c_[self.X.ravel(), self.Y.ravel()]
        x, y = coords.T
        f = Z.T.flatten()
        return estimate_moments_2d(f, x, y)
    
    
def estimate_moments(Z, xcenters=None, ycenters=None):
    if xcenters is None:
        xcenters = np.arange(Z.shape[0])
    if ycenters is None:
        ycenters = np.arange(Z.shape[1])
    X, Y = np.meshgrid(xcenters, ycenters)
    coords = np.c_[X.ravel(), Y.ravel()]
    x, y = coords.T
    f = Z.T.flatten()
    return estimate_moments_2d(f, x, y)

    
def estimate_moments_1d(f, x):
    mean = np.sum(f * x) / np.sum(f)
    sig2 = np.sum(f * (x - mean)**2) / np.sum(f)
    return mean, sig2


def estimate_moments_2d(f, x, y):
    mean_x = np.sum(f * x) / np.sum(f)
    mean_y = np.sum(f * y) / np.sum(f)
    sig_xx = np.sum(f * (x - mean_x)**2) / np.sum(f)
    sig_yy = np.sum(f * (y - mean_y)**2) / np.sum(f)
    sig_xy = np.sum(f * (x - mean_x) * (y - mean_y)) / np.sum(f)
    return mean_x, mean_y, sig_xx, sig_yy, sig_xy

    
def process_array(array, make_square=False):
    """Process the target image PV data.
    
    The target image PV data is an array of shape (80000,). We reshape this
    into an array Z of shape (400, 200) so that Z[i, j] corresponds to point
    (x = x[i], y = y[j]).
    """
    Z = array.reshape(200, 400) 
    Z = Z.T # rows for x, columns for y
    Z = np.flip(Z, axis=1) # Z[i, j] is for (x[i], y[j])
    if make_square:
        pad = np.zeros((400, 100))
        Z = np.hstack([pad, Z, pad])
    return Z    


def read_file(filename, n_avg='all', thresh=0, make_square=False):
    """Read an image file.

    Each row in the file is an image take at a different beam pulse. We need
    to remove blank images and then average over the images.
    
    Parameters
    ----------
    n_avg : int or 'all'
        The number of images to include in the average.
    thresh : int
        If the number of nonzero pixels in an image is less than `thresh`, 
        exclude that image from the average.
    make_square : bool
        Whether to pad the y dimension with zeros to make the image square.
        
    Returns
    -------
    Image
        The average over the images.
    """
    arrays = np.loadtxt(filename)
    if arrays.ndim == 1:
        arrays = [arrays]
    Z_list = [process_array(array, make_square) for array in arrays]

    # Remove duplicates.
    n = len(Z_list)
    Z_list = np.unique(Z_list, axis=0)
    if len(Z_list) < n:
        print('Excluding {} duplicates.'.format(n - len(Z_list)))
    
    # Remove blanks.
    n = len(Z_list)
    Z_list = [Z for Z in Z_list if np.count_nonzero(Z > 0.) > thresh]
    if len(Z_list) < n:
        print('Excluding {} blanks'.format(n - len(Z_list)))
        
    # Return the average of the remaining images.
    if n_avg == 'all':
        n_avg = len(Z_list)
    return TargetImage(np.mean(Z_list[:n_avg], axis=0))

        
def read_files(filenames, **read_file_kws):
    """Load images and sort by timestamp."""
    TFile = collections.namedtuple('TFile', ['filename', 'timestamp'])
    tfiles = []
    for filename in filenames:
        datetime_str = filename.split('image_')[-1].split('.dat')[0]
        date_str, time_str = datetime_str.split('_')
        times = []
        times += [int(s) for s in date_str.split('.')]
        times += [int(s) for s in time_str.split('.')]
        tfiles.append(TFile(filename, datetime(*times)))
    tfiles = sorted(tfiles, key=lambda tfile: tfile.timestamp)
    return [read_file(tfile.filename, **read_file_kws) for tfile in tfiles]


def fit_gauss2d(X, Y, Z):
    """Fit a 2D Gaussian to an image."""
    def gauss2d(XY, sig_xx, sig_yy, sig_xy, mean_x, mean_y, amp):
        X, Y = XY
        x = X - mean_x
        y = Y - mean_y
        det = sig_xx * sig_yy - sig_xy**2
        Z = amp * np.exp(-0.5*(sig_yy*x**2 + sig_xx*y**2 - 2*sig_xy*x*y) / det)
        return Z.ravel()
    XY = (X, Y)
    p0 = (1., 1., 0., 1., 1., 1.)
    params, _ = opt.curve_fit(gauss2d, XY, Z.ravel(), p0=p0)
    Zfit = gauss2d(XY, *params).reshape(Z.shape)
    return Zfit, params


def rms_ellipse_dims(sig_xx, sig_yy, sig_xy):
    """Return semi-axes and tilt angle of the RMS ellipse in the x-y plane."""
    angle = -0.5 * np.arctan2(2 * sig_xy, sig_xx - sig_yy)
    sn, cs = np.sin(angle), np.cos(angle)
    c1 = np.sqrt(abs(sig_xx*cs**2 + sig_yy*sn**2 - 2*sig_xy*sn*cs))
    c2 = np.sqrt(abs(sig_xx*sn**2 + sig_yy*cs**2 + 2*sig_xy*sn*cs))
    return c1, c2, angle