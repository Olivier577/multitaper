import math
import numpy as np


cpdef sft(double [:] x, double om):

    cdef int n = len(x)

    cdef int np1 = n+1
    cdef int l = math.floor(6.0*om/(2.0*math.pi))
    cdef double a = 0.0
    cdef double b
    cdef double c
    cdef double d = 0.0
    cdef double e = 0.0
    cdef double st, ct
    
    cdef int k, i

    if l == 0:

        # recursion for low frequencies (.lt. nyq/3)

        b = -4.0*math.sin(om/2.0)**2
        for k in range(1, np1):
            i = n-k
            c = a
            d = e
            a = x[i]+b*d+c
            e = a+d
      
    elif (l == 1):

        #regular goertzal algorithm for intermediate frequencies

        b = 2.0*math.cos(om)
        for k in range(1, np1):
            i = n-k
            a = x[i]+b*e-d
            d = e
            e = a

    else:
        # recursion for high frequencies (> 2*fnyq/3)
      
        b = 4.0*math.cos(om/2.0)**2
        for k in range(1, np1):
            i = n-k
            c = a
            d = e
            a = x[i]+b*d-c
            e = a-d

    st = -math.sin(om)*d
    ct = a-b*d/2.0

    return ct, st


cpdef set_xint(int ising):

    cdef int nomx = 8
    cdef int lomx = 256
    #cdef double w[8][257]
    w = np.zeros((8,257))
    cdef double [:, :] w_ = w
    #cdef double x[257]
    x = np.zeros(257)
    cdef double [:] x_ = x
    #x_[:] = 0.
    #w_[:,:] = 0.

    cdef double pi = math.pi
    cdef int n = 2
    cdef int index, nx, nhalf, i, k
    cdef double pin, t, si, ck, rk, rn

    for index in range(nomx):
        n *= 2
        rn = <double>n
        nx = n-2
        if index == 0:
            nx = 4
      
        pin = pi/rn
        nhalf = <int>(n/2)
        for i in range(nhalf+1):
            t = <double>i*pin
            si = 0.0
            for k in range(0,nx+1,2):
                ck=4.0
                if k == 0:
                    ck = 2.0
                rk = <double>k
                si = si+ck*math.cos(rk*t)/(1.0-rk*rk)

            if (i==0 or i==nhalf):
                si *= 0.5

            t = math.cos(t)

            if ising == 2:
                t = 0.5*pi*(1.0 +t)
                si = si*0.5 * math.sin(t)*pi
                t = math.cos(t)
                x_[i] = 0.5 *(1.0 +t)
                w_[index,i] = 0.5 *si/rn
            elif ising == 1:
                x_[i] = 0.5 *(1.0 +t)
                w_[index,i] = 0.5 *si/rn
        # end i loop
    # end index loop         
      
    return w, x