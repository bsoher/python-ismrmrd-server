#!/usr/bin/env python

# Copyright (c) 2024-2025 Brian J Soher - All Rights Reserved
#
# Redistribution and use in source and binary forms, with or without
# modification, are not permitted without explicit permission.

# Python modules
import logging

# 3rd party modules
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.fft import fftshift as shift
from scipy.fft import ifftshift as ishift
from scipy.fft import fft, ifft, fftn

# Our modules


#logger = logging.getLogger((__name__).split('.')[-1])   # just module name
#logger.setLevel(logging.DEBUG)




def inline_get_traj():
    """
    This trajectory array created from 'd:\\Midas\\Bin\\epsi\\g100_r130_sim.dat'

    """
    full_traj = False  # trajectory mode, 0 = one cycle, 1 = file covers whole multi-echo train

    k_traj = np.array([
        7.6293945e-06, 3.1311557e-02, 1.2515242e-01, 2.8131762e-01, 4.9945346e-01, 7.7906573e-01, 1.1195210e+00, 1.5200483e+00,
        1.9797397e+00, 2.4975543e+00, 3.0723190e+00, 3.7027318e+00, 4.3873649e+00, 5.1246667e+00, 5.9129672e+00, 6.7504807e+00,
        7.6353106e+00, 8.5654516e+00, 9.5387964e+00, 1.0553142e+01, 1.1606188e+01, 1.2695551e+01, 1.3818761e+01, 1.4973276e+01,
        1.6156481e+01, 1.7365692e+01, 1.8598171e+01, 1.9851130e+01, 2.1121725e+01, 2.2407082e+01, 2.3704287e+01, 2.5010403e+01,
        2.6322470e+01, 2.7637888e+01, 2.8953304e+01, 3.0268723e+01, 3.1584141e+01, 3.2899559e+01, 3.4214977e+01, 3.5530396e+01,
        3.6845814e+01, 3.8161232e+01, 3.9476650e+01, 4.0792068e+01, 4.2107491e+01, 4.3422909e+01, 4.4738327e+01, 4.6053745e+01,
        4.7369164e+01, 4.8684582e+01, 5.0000000e+01, 5.1315418e+01, 5.2630836e+01, 5.3946251e+01, 5.5261669e+01, 5.6577087e+01,
        5.7892506e+01, 5.9207924e+01, 6.0523342e+01, 6.1838760e+01, 6.3154179e+01, 6.4469597e+01, 6.5785011e+01, 6.7100433e+01,
        6.8415848e+01, 6.9731270e+01, 7.1046684e+01, 7.2362106e+01, 7.3677521e+01, 7.4989586e+01, 7.6295708e+01, 7.7592911e+01,
        7.8878265e+01, 8.0148857e+01, 8.1401817e+01, 8.2634308e+01, 8.3843521e+01, 8.5026718e+01, 8.6181236e+01, 8.7304443e+01,
        8.8393806e+01, 8.9446854e+01, 9.0461197e+01, 9.1434540e+01, 9.2364685e+01, 9.3249519e+01, 9.4087029e+01, 9.4875328e+01,
        9.5612640e+01, 9.6297272e+01, 9.6927689e+01, 9.7502449e+01, 9.8020264e+01, 9.8479958e+01, 9.8880493e+01, 9.9220947e+01,
        9.9500557e+01, 9.9718689e+01, 9.9874855e+01, 9.9968704e+01, 1.0000001e+02, 9.9968704e+01, 9.9874855e+01, 9.9718689e+01,
        9.9500557e+01, 9.9220947e+01, 9.8880493e+01, 9.8479958e+01, 9.8020264e+01, 9.7502449e+01, 9.6927689e+01, 9.6297272e+01,
        9.5612640e+01, 9.4875328e+01, 9.4087029e+01, 9.3249519e+01, 9.2364685e+01, 9.1434540e+01, 9.0461197e+01, 8.9446854e+01,
        8.8393806e+01, 8.7304443e+01, 8.6181236e+01, 8.5026718e+01, 8.3843513e+01, 8.2634300e+01, 8.1401817e+01, 8.0148857e+01,
        7.8878258e+01, 7.7592903e+01, 7.6295700e+01, 7.4989586e+01, 7.3677521e+01, 7.2362106e+01, 7.1046684e+01, 6.9731270e+01,
        6.8415848e+01, 6.7100433e+01, 6.5785011e+01, 6.4469597e+01, 6.3154179e+01, 6.1838760e+01, 6.0523342e+01, 5.9207924e+01,
        5.7892506e+01, 5.6577087e+01, 5.5261669e+01, 5.3946251e+01, 5.2630829e+01, 5.1315411e+01, 4.9999992e+01, 4.8684574e+01,
        4.7369156e+01, 4.6053738e+01, 4.4738319e+01, 4.3422901e+01, 4.2107483e+01, 4.0792065e+01, 3.9476646e+01, 3.8161228e+01,
        3.6845810e+01, 3.5530392e+01, 3.4214973e+01, 3.2899555e+01, 3.1584139e+01, 3.0268721e+01, 2.8953302e+01, 2.7637884e+01,
        2.6322466e+01, 2.5010401e+01, 2.3704285e+01, 2.2407080e+01, 2.1121723e+01, 1.9851128e+01, 1.8598169e+01, 1.7365688e+01,
        1.6156477e+01, 1.4973273e+01, 1.3818760e+01, 1.2695549e+01, 1.1606185e+01, 1.0553139e+01, 9.5387945e+00, 8.5654488e+00,
        7.6353083e+00, 6.7504783e+00, 5.9129643e+00, 5.1246638e+00, 4.3873620e+00, 3.7027290e+00, 3.0723162e+00, 2.4975517e+00,
        1.9797370e+00, 1.5200455e+00, 1.1195184e+00, 7.7906317e-01, 4.9945089e-01, 2.8131506e-01, 1.2514986e-01, 3.1308994e-02],
        dtype=np.float32)

    return k_traj


def inline_idl_interpolate(xx, arr, cubic=-0.5, missing=None):

    g = np.clip(cubic, -1.0, 0.0)

    xi = np.zeros([4,], dtype=np.int16)
    n1 = len(arr)
    nx = len(xx)

    is2d = True if len(arr.shape) > 1 else False

    rshape = [xx.shape[0],arr.shape[1]] if is2d else [xx.shape[0],1]
    res = np.ndarray(shape=rshape, dtype=arr.dtype)

    dat = np.expand_dims(arr, axis=1) if not is2d else arr

    for j in range(nx):
        x = xx[j]
        if x < 0:
            if missing is not None:
                res[j,:] = missing
            else:
                res[j,:] = dat[0,:]
        elif x < n1-1:
            ix = np.floor(x)
            xi[0] = ix - 1
            xi[1] = ix
            xi[2] = ix + 1
            xi[3] = ix + 2

            # make in range

            xi = np.clip(xi, 0, n1-1)

            dx = x - xi[1]
            d2 = dx*dx
            d3 = d2*dx
            omd = 1 - dx
            omd2 = omd*omd
            omd3 = omd2*omd
            opd = 1 + dx
            opd2 = opd*opd
            opd3 = opd2*opd
            dmd = 2 - dx
            dmd2 = dmd*dmd
            dmd3 = dmd2*dmd
            c1 = ((g + 2) * d3 - (g + 3) * d2 + 1)
            c2 = ((g + 2) * omd3 - (g + 3) * omd2 + 1)
            c0 = (g * opd3 - 5 * g * opd2 + 8 * g * opd - 4 * g)
            c3 = (g * dmd3 - 5 * g * dmd2 + 8 * g * dmd - 4 * g)
            res[j,:] = c1 * dat[xi[1],:] + c2 * dat[xi[2],:] + c0 * dat[xi[0],:] + c3 * dat[xi[3],:]

        elif x < n1:
            res[j,:] = dat[n1-1,:]
        else:
            if missing is not None:
                res[j,:] = missing
            else:
                res[j,:] = dat[n1-1,:]

    return np.squeeze(res)


def inline_calc_average_center_of_mass(data):
    """
    Calc center of mass left to right, then right to left, return midway point.

    """
    x1, y1, slope1 = inline_calc_center_of_mass(data)
    x2, y2, slope2 = inline_calc_center_of_mass(data[::-1])

    # Slope from test will be +ve, but we need to swap to -ve
    slope2 = -slope2
    x2 = data.size - x2 - 1

    # compute X values for Y=0
    ax1 = float(x1) - y1 / slope1
    ax2 = float(x2) - y2 / slope2

    return (ax1 + ax2) / 2


def inline_calc_center_of_mass(data):
    """
    Find index for point at half total amplitude, return index, value at
    that point and the slope at that point.

    """
    tmp = data.cumsum()
    xx = (tmp > tmp[-1]/2.0).nonzero()[0][0]
    if xx < 2: raise ValueError("Center of mass index < 2")
    return xx, tmp[xx], (tmp[xx]-tmp[xx-1])


def inline_calc_first_echo_shift(nx, data, channel):
    '''
    calculates shift of positions for 1st odd and even echos from center point.
    Uses center-of-mass to find echo center. The value returned for even echo
    shift is as the data is acquired, i.e. no flip applied. The function used
    for the correction is flipped later to match the flip of the even echo data.

    '''
    nx2 = float(nx) / 2.0

    offs2 = np.zeros(2, dtype="float32")  # for odd and even echo
    offs2[0] = inline_calc_average_center_of_mass(abs(data[0:nx]))
    offs2[1] = inline_calc_average_center_of_mass(abs(data[nx:nx * 2]))
    offs2[1] = nx - offs2[1]

    test1 = offs2[0] < nx2 - 0.5 or offs2[0] > nx2 + 0.5
    test2 = offs2[1] < nx2 - 0.5 or offs2[1] > nx2 + 0.5

    if test1 or test2:
        schan = f'{channel:2d}'
        soff0 = f'{offs2[0]:4.6f}'
        soff1 = f'{offs2[1]:4.6f}'
        snx2  = f'{int(nx / 2):3d}'
        logging.info(' : Inline Chan ' + schan + ' Odd/Even echo at points ' + soff0 + ' / ' + soff1 + ' (should be ' + snx2 + ')')

    echoshift = offs2 - nx2  # +ve value shift to right

    return echoshift


def inline_calc_freq_drift(block, data, ichan):
    '''
    Get FID using peaks from zero k data, based on the calculated echo
    positions. Do for odd echos only

    '''
    nt, nx = block.nt, block.nx
    nt2, nx2 = int(nt / 2), int(nx / 2)
    indx0 = np.arange(0, nt, 2)  # index for odd-echos into echo-shifts array

    indx = np.round(block.echo_shifts[ichan, 0]).astype(int) + int(nx2)
    indx = indx + indx0 * nx
    fid = data[indx]
    nt2 = nt2 / 2  # now half the number of points
    fid[0] = fid[0] / 2.

    fid = shift(fft(shift(fid), overwrite_x=True))
    indx = np.argmax(abs(fid)) - nt2
    hz = indx * block.sw / block.nt

    return hz


def inline_init_traj_corr(block):
    '''
    The REF acq data must be stored for this to work

    Read in k-space trajectory file and run echo position and drift
    calculations. Store in block for SI use.

    RETURN:
      k_data = k-space trajectory for one odd/even echo pair. Read once at
      start of program and then used together with the echo shift positions to
      set up interpolation points for each channel.

    '''
    nx, ny, nz, nt, nchan = block.nx, block.ny, block.nz, block.nt, block.nchannels
    block.echo_shifts = np.zeros([nchan, 2], dtype=np.float32)

    k_traj = inline_get_traj()

    freq_shift = 0.0
    for ichan in range(nchan):

        data = block.ref[ichan, :, :]
        data = data.flatten()

        shifts = inline_calc_first_echo_shift(nx, data, ichan)
        block.echo_shifts[ichan, :] = shifts

        freq_shift += inline_calc_freq_drift(block, data, ichan)

    # Get average value of frequency shift and multiply by 1.7 to estimate final value.
    # Empirical modification that is somewhere between linear and exponential - see Ebel 2004

    val = freq_shift / nchan
    freq_shift = 1.7 * val
    block.frequency_drift_value = freq_shift

    msg  = '\n    : inline Frequency drift at k-space center = ' + f'{val:.2f}' + ' Hz.'
    msg += '\n    : inline Total frequency drift estimate    = ' + f'{freq_shift:.2f}' + ' Hz.'
    msg += '\n    : inline Final frequency drift estimate    = ' + f'{freq_shift:.2f}' + ' Hz. \n'
    logging.info(msg)

    return k_traj


def inline_init_interp_kx(block):
    '''
    Change the kxt-trajectory into the points for resampling the data using
    the interpolate function. This corresponds to equal kx steps.  Also
    rearrange into odd and even. Correct if the k-space function has drift,

    '''
    nt, nx, os, nchan = block.nt, block.nx, block.os, block.nchannels
    nx_out = int(block.nx / block.os)

    k_traj = block.k_traj
    k_traj = k_traj.reshape(2, nx)

    samples = np.arange(nx + 1)
    samples2 = np.arange(nx_out) * os

    low = int(np.ceil(k_traj[0,0]) / block.os)
    high = int(np.floor(k_traj[1,0]) / block.os)
    high = high if high < nx_out else nx_out

    samples2 = samples2[low: high]
    nresamp = len(samples2)

    xino = np.zeros((nchan, nresamp), dtype=np.float32)  # odd
    xine = np.zeros((nchan, nresamp), dtype=np.float32)  # even

    for ichan in range(nchan):

        # Calculate for Odd Echo
        # - set up k-space for current echo, include 1st point of next echo
        # - shift the time point by the amount the echo is offset
        # - get points to interpolate the samples at
        # - echo shift is actual echo position, not correction, so no need
        #    to make negative
        temp = np.r_[k_traj[0, :], k_traj[1, 0]]
        time_samples = samples + block.echo_shifts[ichan, 0]
        func = splrep(temp, time_samples, s=0)
        xino[ichan, :] = splev(samples2, func, der=0)

        # Calc for Even Echo
        # - reverse the echo data and also kx function
        temp = np.r_[k_traj[1, :], k_traj[0, 0]]
        temp = temp[::-1]
        time_samples = samples + block.echo_shifts[ichan, 1]
        func = splrep(temp, time_samples)
        xine[ichan, :] = splev(samples2, func)

    return xino, xine


def inline_init_process_kt(block, reverse=False):
    '''
    Precompute phase correction terms used in process_kt(). This is to
    convert samples for each echo in the ky-t EPSI function to the same point
    in time using the Fourier shift theory. See Metzger and Hu on the IFT.

    '''
    nchan = block.nchannels
    nt = block.nt
    sw = block.sw  # this sweepwidth is for all echos
    ts = 1.0 / sw  # time of one echo readout in sec

    nt2 = int(nt/2)  # number of echo pairs
    ts2 = ts/2
    nx_out = int(block.nx / block.os)

    # find next data size for 2^N
    npow = 1
    while (nt2 / pow(2, npow)) > 2: npow += 1
    npow += 1  # this is the 2^N lower than the number, go to next higher
    if (pow(2, npow) - nt2 > 10):  # try 10 as minimum padding size to allow 500--> 512
        pad_leng = pow(2, npow) - nt2
    else:
        pad_leng = pow(2, npow+1) - nt2

    block.csa_pad_length = pad_leng

    nt3 = nt2 + pad_leng  # size for DFT to do IFT echo combination

    atr = np.arange(nx_out) * ts / nx_out
    atr = atr - ts2

    if not reverse:
        atr = -atr

    atr_e = atr[::-1]  # for even echoes time axis is reversed

    # odd/even time-shift correction depends on whether echo_output or
    # separate echos, 0 - add odd/even echos, 1 - output separate files

    # bjs default - block.echo_output == 0
    dt0_o = -atr
    dt0_e = -atr_e - ts

    # The sweepwidth here [rad/s] is that of all echos, i.e. before adding odd/even
    # Spectral coordinates in Hz, 0.0 in center
    ww = ((np.arange(nt3) / nt3) - 0.5) * sw
    ww = ww * np.pi

    expo = np.zeros((nchan, nt3, nx_out), dtype=np.complex128)  # odd
    expe = np.zeros((nchan, nt3, nx_out), dtype=np.complex128)  # even

    for ichan in range(nchan):

        # bjs default - block.echo_drift_corr > 0:
        echoshift = block.echo_shifts[ichan, 0:2]           # in points, take 1st odd & even only
        echoshift = echoshift * block.sampling_interval     # convert to sec.
        dt_o = dt0_o - echoshift[0]
        dt_e = dt0_e - echoshift[1]

        #  Precompute exp term for ChemShift phase correction
        expo[ichan, :, :] = np.exp(1j * np.outer(ww, dt_o))
        expe[ichan, :, :] = np.exp(1j * np.outer(ww, dt_e))

    return expo, expe


def inline_process_kx(block, data):
    ''' 1D interpolate regrid of EPSI nx data '''

    nt, nx, ny, nz, nchan = block.nt, block.nx, block.ny, block.nz, block.nchannels
    nx_out = int(block.nx/block.os)
    nt2 = int(nt/2)
    nresamp = len(block.xino[0, :])

    nzf1 = int(int(nx_out-nresamp+1)/2)     # changed from interp_kx() in epsi_util.py
    if nzf1 > 0:
        raise ValueError('inline_process_kx(): nzf1>0, nx-out and nresamp mismatch.')

    for ichan in range(nchan):
        tmp_odd = []
        tmp_evn = []

        xino = block.xino[ichan, :]
        xine = block.xine[ichan, :]

        for k in range(nt):

            if k != nt-1:
                tmp = np.r_[ data[ichan,k,:].flatten(), data[ichan,k+1,0] ]
            else:
                tmp = np.r_[ data[ichan,k,:].flatten(), 0+0j ]

            if (k%2) != 0:
                tmp = tmp[::-1]
                tmp_evn.append(tmp)
            else:
                tmp_odd.append(tmp)

        odd_arr = np.array(tmp_odd).T
        res_odd = inline_idl_interpolate(xino, odd_arr, cubic=-0.5, missing=0.0)

        evn_arr = np.array(tmp_odd).T
        res_evn = inline_idl_interpolate(xine, evn_arr, cubic=-0.5, missing=0.0)

        if xino[0] < 0:       # match IDL interpolate(missing=0) keyword option
            res_odd[0,:] = 0+0j

        if xine[0] < 0:
            res_evn[0,:] = 0+0j

        data[ichan, :, nx_out:] *= 0.0
        for kk in range(nt2):
            data[ichan, kk*2,  0:nx_out] = res_odd[:, kk]
            data[ichan, kk*2+1,0:nx_out] = res_evn[:, kk]

    return data


def inline_process_kt(block, data_in):
    '''
    average odd and even echoes, phase correct for change in sampling time as
    function of kx, transpose data from [x,t,y,z] to [t,x,y,z]

    '''
    nt, ny, nz, nchan = block.nt, block.ny, block.nz, block.nchannels
    nx_out = int(block.nx / block.os)
    nt2 = int(nt/2)
    pad_leng = block.csa_pad_length

    data_out = np.zeros([nchan, nx_out, nt2], dtype=np.complex64)

    od = np.arange(0, nt, 2, dtype=int)
    ev = np.arange(0, nt, 2, dtype=int) + 1
    ntot = nt2 + pad_leng
    tmpoo = np.zeros(ntot, dtype="complex64")
    tmpee = np.zeros(ntot, dtype="complex64")

    for ichan in range(nchan):
        expo = block.expo[ichan, :, :]
        expe = block.expe[ichan, :, :]

        for x in range(nx_out):  # loop over kx points

            # TODO bjs - could lose x-loop and do 1D shifts on 1 axis of 2D array - for speed
            #  - bjs, could maybe even do all dims at once with careful indexing expo/expe

            tmpoo[0:nt2] = data_in[ichan, od, x].flatten()
            tmpee[0:nt2] = data_in[ichan, ev, x].flatten()

            if block.echo_average_fix:
                temp = (tmpoo[0:3] + tmpee[0:3]) * 0.5

            tmpoo[nt2::] = 0.0          # make 2^N with end of FID 0.0
            tmpoo[0] = 0.5 * tmpoo[0]   # divide 1st point of FID by 2 for FFT
            tmpoo = shift(fft(tmpoo, overwrite_x=True))
            tmpoo = tmpoo * expo[:, x]  # apply CS phase correction

            tmpee[nt2::] = 0.0
            tmpee[0] = 0.5 * tmpee[0]
            tmpee = shift(fft(tmpee, overwrite_x=True))
            tmpee = tmpee * expe[:, x]

            tmpoo = (tmpoo + tmpee) * 0.5
            tmpoo = ifft(ishift(tmpoo), overwrite_x=True)  # inverse FFT
            tmpoo[0] = 2.0 * tmpoo[0]  # restore previously applied /2 for 1st point

            # Replace initial points with unprocessed odd+even-echo. This corrects
            # incorrect initial FID values due to frequency-dependent phase shift
            if block.echo_average_fix:
                tmpoo[0:3] = temp[0:3]

            data_out[ichan, x, :] = tmpoo[0:nt2]  # select just FID part, not zerofilled tail

    return data_out


def inline_apply_freq_drift(block, data):
    '''
    Correct EPSI data for frequency drift during acquisition. Frequency shift
    assumed linear change from start to end.

    '''
    nt, nx, ny, nz = block.nt, block.nx, block.ny, block.nz
    fd = block.frequency_drift_value
    sw = block.sw
    iy = block.curr_yindx
    iz = block.curr_zindx

    t = np.arange(int(nt/2), dtype=np.float32) / (sw / 2)
    t = -1j * 2 * np.pi * t
    df = float(fd / (nz * ny))  # Hz/acquisition

    iz2 = iz * ny
    f = df * (iz2 + iy)  # correction (Hz)

    data = data * np.exp(f * t)

    return data


def inline_do_epsi(block, data_in):
    """
    - data_in is a numpy array of one EPI readout, either water OR metab
    - data_in - [ncha, nt, nx]
    - data_out - reduced to [ncha, nt/2, nx/2]

    """

    data_in  = inline_process_kx(block, data_in)
    data_out = inline_process_kt(block, data_in)
    data_out = inline_apply_freq_drift(block, data_out)    # def to set.frequency_drift_corr > 0

    return data_out

# -----------------------------------------------------------------------------
# test code

def _test():
    pass


if __name__ == '__main__':
    _test()


