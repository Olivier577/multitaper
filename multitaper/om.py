import numpy as np
import pandas as pd
import xarray as xr

import multitaper.utils as utils
from multitaper.mtcross import MTCross


def cross_spectrogram(
    data, dt, twin, olap=0.5, nw=3.5, kspec=5, fmin=0.0, fmax=-1.0,
    iadapt=0, vn=None, lamb=None, wl=0.0
):
    """
    Computes a cross-spectrogram with consecutive multitaper estimates.
    Returns both Thomson's multitaper and the Quadratic multitaper estimate

    **Parameters**

    data : array_like (npts, 2)
        Time series or sequence
    dt : float
        Sampling interval in seconds of the time series.
    twin : float
        Time duration in seconds of each segment for a single multitaper estimate.
    olap : float, optional
        Overlap requested for the segment in building the spectrogram.
        Defaults = 0.5, values must be (0.0 - 0.99).
        Overlap rounds to the nearest integer point.
    nw : float, optional
        Time-bandwidth product for Thomson's multitaper algorithm.
        Default = 3.5
    kspec : int, optional
        Number of tapers for avearaging the multitaper estimate.
        Default = 5
    fmin : float, optional
        Minimum frequency to estimate the spectrogram, otherwise returns the
        entire spectrogram matrix.
        Default = 0.0 Hz
    fmax : float, optional
        Maximum frequency to estimate the spectrogram, otherwise returns the
        entire spectrogram matrix.
        Default = 0.5/dt Hz (Nyquist frequency)
    iadapt : integer, optional
        User defined, determines which method for multitaper averaging to use.
        Default = 0
        0 - Adaptive multitaper
        1 - Eigenvalue weights
        2 - Constant weighting
    wl : float, optional
        water-level for stabilizing deconvolution (transfer function).
        defined as proportion of mean power of Syy

    **Returns**

    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Quad : ndarray
        Spectrogram of x using the quadratic multitaper estimate.
    MT : ndarray
        Spectrogram of x using Thomson's multitaper estimate.

    By default, the last axis of Quad/MT corresponds to the segment times.

    **See Also**

    MTSpec: Multitaper estimate of a time series.

    **Notes**

    The code assumes a real input signals and thus mainly returns the positive
    frequencies. For a complex input signals, code qould require adaptation.

    **References**

       Prieto, G.A. (2022). The multitaper spectrum analysis package in Python.
       Seism. Res. Lett In review.

    **Examples**

    To do

    |

    """

    data_is_df = False
    if isinstance(data, pd.DataFrame):
        data_is_df = True
        if data.index.freq is not None:
            dt = pd.to_timedelta(data.index.freq).total_seconds()

    if fmax <= 0.0:
        fmax = 0.5 / dt

    nwin = int(np.round(twin / dt))
    if olap <= 0.0:
        njump = nwin
    else:
        njump = int(np.round(twin * (1.0 - olap) / dt))

    npts = np.shape(data)[0]
    nmax = npts - nwin
    nvec = np.arange(0, nmax, njump)
    t = nvec * dt  # window starting times in seconds
    nspec = len(nvec)
    nvars = 5

    if data_is_df:
        t_middle = data.index[0] + pd.to_timedelta(t, unit='seconds')  # window middle datetime
        data = data.to_numpy()
    else:
        t_middle = t + twin // 2  # window middle times in seconds    

    print("Window length %5.1fs and overlap %2.0f%%" % (twin, olap * 100))
    print("Total number of cross-spectral estimates", nspec)
    print("Frequency band of interest (%5.2f-%5.2f)Hz" % (fmin, fmax))

    if vn is None or lamb is None:
        vn, lamb = utils.dpss(nwin, nw, kspec)

    for i in range(nspec):
        if (i + 1) % 10 == 0:
            print("Loop ", i + 1, " of ", nspec)

        i1 = nvec[i]
        i2 = i1 + nwin

        cpsd = MTCross(
            data[i1: i2 + 1, 0],  # x: independent/explanatory variable
            data[i1: i2 + 1, 1],  # y: dependent/response variable 
            nw,
            kspec,
            dt,
            iadapt=iadapt,
            vn=vn,
            lamb=lamb,
            wl=wl,
        )

        freq2 = cpsd.freq

        nf = len(freq2)

        if i == 0:

            fres = np.where((freq2 >= fmin) & (freq2 <= fmax))[0]
            nf = len(fres)
            f = freq2[fres]

            ds = xr.DataArray(
                np.ones((nf, nspec, nvars), dtype=complex) * np.nan,
                dims=('f', 't', 'var'),
                coords=dict(f=f.flatten(), t=t_middle, var=[
                            'Sxx', 'Syy', 'Sxy', 'Syx', 'Cxy'])
            )

        if np.isnan(data[i1: i2 + 1, :]).sum() > 0.1 * nwin:
            continue

        ds.sel(var='Sxx')[:, i] = cpsd.Sxx[fres, 0]
        ds.sel(var='Syy')[:, i] = cpsd.Syy[fres, 0]
        ds.sel(var='Sxy')[:, i] = cpsd.Sxy[fres, 0]
        ds.sel(var='Syx')[:, i] = cpsd.Syx[fres, 0]
        ds.sel(var='Cxy')[:, i] = cpsd.cohe[fres, 0]


    return ds