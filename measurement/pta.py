"""Read Profile Tools and Analysis (PTA) wire-scanner files."""
from datetime import datetime
import numpy as np
from scipy import interpolate


DIAG_WIRE_ANGLE = np.radians(-45.0)


def string_to_list(string):
    """Convert string to list of floats.
    
    '1 2 3' -> [1.0, 2.0, 3.0])
    """
    return [float(token) for token in string.split()]


def split(items, token):
    """Split `items` into sublists, excluding `token`.

    Example:
    >>> items = ['cat', 'dog', 'x', 'tree', 'bark']
    >>> split_list(items, 'x')
    [['cat', 'dog'], ['tree', 'bark']]
    """
    indices = [i for i, item in enumerate(items) if item == token]
    sublists = []
    if items[0] != token:
        sublists.append(items[: indices[0]])
    for lo, hi in zip(indices[:-1], indices[1:]):
        sublists.append(items[lo + 1 : hi])
    if items[-1] != token:
        sublists.append(items[indices[-1] + 1 :])
    return sublists


def get_sig_xy(sig_xx, sig_yy, sig_uu, diag_wire_angle):
    """Compute cov(x, y) from horizontal, vertical, and diagonal wires.
    
    Diagonal wire angle should be in radians.
    """
    phi = np.radians(90.0) + diag_wire_angle
    sn, cs = np.sin(phi), np.cos(phi)
    sig_xy = (sig_uu - sig_xx * (cs**2) - sig_yy * (sn**2)) / (2 * sn * cs)
    return sig_xy


class Stat:
    """Container for a signal parameter.
    
    Attributes
    ----------
    name : str
        Parameter name.
    rms : float
        Parameter value from rms calculation.
    fit : float
        Parameter value from Gaussian fit.
    """

    def __init__(self, name, rms, fit):
        self.name, self.rms, self.fit = name, rms, fit


class Signal:
    """Container for profile signal.
    
    Attributes
    ----------
    pos : list
        Wire positions.
    raw : list
        Raw signal amplitudes at each position.
    fit : list
        Gaussian fit amplitudes at each position.
    stats : dict
        Each key is a different statistical parameter: ('Area', 'Mean', etc.). 
        Each value is a Stat object that holds the parameter name, rms value, 
        and Gaussian fit value.
    """

    def __init__(self, pos, raw, fit, stats):
        self.pos, self.raw, self.fit, self.stats = pos, raw, fit, stats


class Profile:
    """Stores data from single wire-scanner.
    
    Attributes
    ----------
    hor, ver, dia : Signal
        Signal object for horizontal, vertical and diagonal wire.
    signals : list
        List of [hor, ver, dia] signals.
    diag_wire_angle : float
        Angle of diagonal wire above the x axis.
    """

    def __init__(self, pos, raw, fit=None, stats=None, diag_wire_angle=DIAG_WIRE_ANGLE):
        """Constructor.
        
        Parameters
        ----------
        pos : [xpos, ypos, upos]
            Position lists for each wire.
        raw : [xraw, yraw, uraw]
            List of raw signal amplitudes for each wire.
        fit : [xfit, yfit, ufit]
            List of Gaussian fit amplitudes for each wire.
        stats : [xstats, ystats, ustats]
            List of stats dictionaries for each wire.
        """
        self.diag_wire_angle = diag_wire_angle
        xpos, ypos, upos = pos
        xraw, yraw, uraw = raw
        if fit is None:
            xfit = yfit = ufit = None
        else:
            xfit, yfit, ufit = fit
        if stats is None:
            xstats = ystats = ustats = None
        else:
            xstats, ystats, ustats = stats
        self.hor = Signal(xpos, xraw, xfit, xstats)
        self.ver = Signal(ypos, yraw, yfit, ystats)
        self.dia = Signal(upos, uraw, ufit, ustats)
        self.signals = [self.hor, self.ver, self.dia]
        
    def get_signal(self, dim='x'):
        """Return the signal along the specified axis."""
        return {'x': self.hor, 'y': self.ver, 'u': self.dia}[dim]


class Measurement(dict):
    """Dictionary of profiles for one measurement.

    Each key in this dictionary is a wire-scanner ID; each value is a Profile.
    
    Attributes
    ----------
    filename : str
        Full path to the PTA file.
    filename_short : str
        Only include the filename, not the full path.
    timestamp : datetime
        Represents the time at which the data was taken.
    pvloggerid : int
        The PVLoggerID of the measurement (this gives a snapshot of the machine state).
    node_ids : list[str]
        The ID of each wire-scanner. (These are the dictionary keys.)
    moments : dict
        The [<x^2>, <y^2>, <xy>] moments at each wire-scanner.
    transfer_mats : dict
        The linear 4x4 transfer matrix from a start node to each wire-scanner. 
        The start node is determined in the function call `get_transfer_mats`.
    """

    def __init__(self, filename):
        dict.__init__(self)
        self.filename = filename
        self.filename_short = filename.split("/")[-1]
        self.timestamp = None
        self.pvloggerid = None
        self.node_ids = None
        self.moments, self.transfer_mats = dict(), dict()
        self.read_pta_file()

    def read_pta_file(self):
        # Store the timestamp on the file.
        date, time = self.filename.split("WireAnalysisFmt-")[-1].split("_")
        time = time.split(".pta")[0]
        year, month, day = [int(token) for token in date.split(".")]
        hour, minute, second = [int(token) for token in time.split(".")]
        self.timestamp = datetime(year, month, day, hour, minute, second)

        # Collect lines corresponding to each wire-scanner
        file = open(self.filename, "r")
        lines = dict()
        ws_id = None
        for line in file:
            line = line.rstrip()
            if line.startswith("RTBT_Diag"):
                ws_id = line
                continue
            if ws_id is not None:
                lines.setdefault(ws_id, []).append(line)
            if line.startswith("PVLoggerID"):
                self.pvloggerid = int(line.split("=")[1])
        file.close()
        self.node_ids = sorted(list(lines))

        # Read the lines
        for node_id in self.node_ids:
            # Split lines into three sections:
            #     stats: statistical signal parameters;
            #     raw: wire positions and raw signal amplitudes;
            #     fit: wire positions and Gaussian fit amplitudes.
            # There is one blank line after each section.
            sep = ""
            lines_stats, lines_raw, lines_fit = split(lines[node_id], sep)[:3]

            # Remove headers and dashed lines beneath headers.
            lines_stats = lines_stats[2:]
            lines_raw = lines_raw[2:]
            lines_fit = lines_fit[2:]

            # The columns of the following array are ['pos', 'yraw', 'uraw', 'xraw',
            # 'xpos', 'ypos', 'upos']. (NOTE: This is not the order that is written
            # in the file header.)
            data_arr_raw = [string_to_list(line) for line in lines_raw]
            pos, yraw, uraw, xraw, xpos, ypos, upos = np.transpose(data_arr_raw)

            # This next array is the same, but it contains 'yfit', 'ufit', 'xfit',
            # instead of 'yraw', 'uraw', 'xraw'.
            data_arr_fit = [string_to_list(line) for line in lines_fit]
            pos, yfit, ufit, xfit, xpos, ypos, upos = np.transpose(data_arr_fit)

            # Get statistical signal parameters. (Headers don't give true ordering.)
            xstats, ystats, ustats = dict(), dict(), dict()
            for line in lines_stats:
                tokens = line.split()
                name = tokens[0]
                vals = [float(val) for val in tokens[1:]]
                s_yfit, s_yrms, s_ufit, s_urms, s_xfit, s_xrms = vals
                xstats[name] = Stat(name, s_xrms, s_xfit)
                ystats[name] = Stat(name, s_yrms, s_yfit)
                ustats[name] = Stat(name, s_urms, s_ufit)

            self[node_id] = Profile(
                [xpos, ypos, upos],
                [xraw, yraw, uraw],
                [xfit, yfit, ufit],
                [xstats, ystats, ustats],
            )

    def get_moments(self):
        """Store/return dictionary of measured moments at each profile."""
        self.moments = dict()
        for node_id in self.node_ids:
            profile = self[node_id]
            sig_xx = profile.hor.stats["Sigma"].rms**2
            sig_yy = profile.ver.stats["Sigma"].rms**2
            sig_uu = profile.dia.stats["Sigma"].rms**2
            sig_xy = get_sig_xy(sig_xx, sig_yy, sig_uu, profile.diag_wire_angle)
            self.moments[node_id] = [sig_xx, sig_yy, sig_xy]
        return self.moments


def read_files(filenames):
    """Read a list of wire-scanner files and sort them by timestamp."""
    measurements = [Measurement(filename) for filename in filenames]
    measurements = sorted(measurements, key=lambda measurement: measurement.timestamp)
    measurements = [
        measurement
        for measurement in measurements
        if measurement.pvloggerid > 0 and measurement.pvloggerid is not None
    ]
    return measurements


class DictOfLists(dict):
    """A dictionary of lists."""
    def __init__(self):
        dict.__init__(self)

    def add(self, key, value):
        if key not in self:
            self[key] = []
        self[key].append(value)


def get_moments_dict(measurements):
    """Construct a dictionary of rms moments from a list of measurements."""
    if type(measurements) is not list:
        measurements = [measurements]
    moments_dict = DictOfLists()
    for measurement in measurements:
        measurement.get_moments()
        for node_id in measurement.node_ids:
            moments_dict.add(node_id, measurement.moments[node_id])
    for node_id in moments_dict:
        moments_dict[node_id] = np.array(moments_dict[node_id])
    return moments_dict


def processed_profiles(
    measurements, 
    ws_id,
    dim="x",
    width=None,
    width_units='data',
    n_interp=None,
):
    """Return ndarray of centered profiles
    
    Parameters
    ----------
    measurements : list[Measurements], shape (n,)
        The measurements containing the profiles.
    ws_id : str
        The wire-scanner at which to extract the profiles.
    dim : {'x', 'y', 'u'}
        The projection axis.
    width : float
        The width of the projection axis.
    width_units : {'data', 'std'}
        If 'std', `width` is a multiple of the maximum 
        standard deviation of the profiles.
    n_interp : int
        The number of points to use along the projection axis. Linear
        interpolation is used. If None, no interpolation is performed.
    
    Returns
    -------
    pos : ndarray, shape (k,)
        The position array.
    data : ndarray, shape (k, 3)
        The horizontal (x), vertical (y) and diagonal (u) profiles.
    """
    data, sigmas = [], []
    for measurement in measurements:
        profile = measurement[ws_id]
        signal = profile.get_signal(dim)
        pos = np.copy(signal.pos) - signal.stats['Mean'].rms
        data.append(np.copy(signal.raw))
        sigmas.append(signal.stats['Sigma'].rms)
    data = np.array(data)
    sigmas = np.array(sigmas)
     
    if n_interp is None:
        n_interp = len(pos)
        
    if width_units == 'std':
        width *= np.max(sigmas)
        
    f = interpolate.interp1d(pos, data, bounds_error=False, fill_value=0.0)
    pos = np.linspace(-0.5 * width, 0.5 * width, n_interp)
    data = f(pos)
    return pos, data
