import itertools
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

    data : array_like (npts, nvars)
        Time series or sequence
    dt : complex
        Sampling interval in seconds of the time series.
    twin : complex
        Time duration in seconds of each segment for a single multitaper estimate.
    olap : complex, optional
        Overlap requested for the segment in building the spectrogram.
        Defaults = 0.5, values must be (0.0 - 0.99).
        Overlap rounds to the nearest integer point.
    nw : complex, optional
        Time-bandwidth product for Thomson's multitaper algorithm.
        Default = 3.5
    kspec : int, optional
        Number of tapers for avearaging the multitaper estimate.
        Default = 5
    fmin : complex, optional
        Minimum frequency to estimate the spectrogram, otherwise returns the
        entire spectrogram matrix.
        Default = 0.0 Hz
    fmax : complex, optional
        Maximum frequency to estimate the spectrogram, otherwise returns the
        entire spectrogram matrix.
        Default = 0.5/dt Hz (Nyquist frequency)
    iadapt : integer, optional
        User defined, determines which method for multitaper averaging to use.
        Default = 0
        0 - Adaptive multitaper
        1 - Eigenvalue weights
        2 - Constant weighting
    wl : complex, optional
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

    if fmax <= 0.0:
        fmax = 0.5 / dt

    nwin = int(np.round(twin / dt))
    if olap <= 0.0:
        njump = nwin
    else:
        njump = int(np.round(twin * (1.0 - olap) / dt))

    npts, nvars = np.shape(data)
    nmax = npts - nwin
    nvec = np.arange(0, nmax, njump)
    t = nvec * dt
    nspec = len(nvec)

    pairs = list(itertools.combinations(range(nvars), 2))
    npairs = len(pairs)

    print("Number of variables", nvars)
    print("Number of variable pairs", npairs)
    print("Window length %5.1fs and overlap %2.0f%%" % (twin, olap * 100))
    print("Total number of cross-spectral estimates", nspec)
    print("Frequency band of interest (%5.2f-%5.2f)Hz" % (fmin, fmax))

    vn, theta = utils.dpss(nwin, nw, kspec)

    for j, (k, l) in enumerate(pairs):
        print("Loop over pair ", j + 1, " of ", npairs)

        for i in range(nspec):
            if (i + 1) % 10 == 0:
                print("Loop ", i + 1, " of ", nspec)

            i1 = nvec[i]
            i2 = i1 + nwin

            cpsd = MTCross(
                data[i1: i2 + 1, k],
                data[i1: i2 + 1, l],
                nw,
                kspec,
                dt,
                iadapt=iadapt,
                vn=vn,
                lamb=theta,
                wl=wl,
            )

            freq2 = cpsd.freq

            nf = len(freq2)

            if j == i == 0:
                fres = np.where((freq2 >= fmin) & (freq2 <= fmax))[0]
                nf = len(fres)
                f = freq2[fres]
                Sxx = np.zeros((nf, nspec, npairs), dtype=complex)
                Syy = np.zeros((nf, nspec, npairs), dtype=complex)
                Sxy = np.zeros((nf, nspec, npairs), dtype=complex)
                Syx = np.zeros((nf, nspec, npairs), dtype=complex)
                cohe = np.zeros((nf, nspec, npairs), dtype=float)
                trf = np.zeros((nf, nspec, npairs), dtype=complex)

            Sxx[:, i, j] = cpsd.Sxx[fres, 0]
            Syy[:, i, j] = cpsd.Syy[fres, 0]
            Sxy[:, i, j] = cpsd.Sxy[fres, 0]
            Sxy[:, i, j] = cpsd.Sxy[fres, 0]
            Syx[:, i, j] = cpsd.Syx[fres, 0]
            cohe[:, i, j] = cpsd.cohe[fres, 0]
            trf[:, i, j] = cpsd.trf[fres, 0]

    return xr.DataArray(
        dict(Sxx=Sxx, Syy=Syy, Sxy=Sxy, Syx=Syx, cohe=cohe, trf=trf),
        coords=dict(t=t, f=f)
    )
