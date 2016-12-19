#!/usr/bin/env python
'''
Defines classes and functions used by some other scripts inside this folder. 


Copyright (C) 2016 Espen Hagen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


'''
import numpy as np
import scipy.signal as ss
import pylab as pl

def PrPz(r0, z0, r1, z1, r2, z2, r3, z3):
    '''intersection point for infinite lines'''
    Pr = ((r0*z1 - z0*r1)*(r2 - r3) - (r0 - r1)*(r2*z3 - r3*z2)) / \
                        ((r0 - r1)*(z2 - z3) - (z0 - z1)*(r2-r3))
    Pz = ((r0*z1 - z0*r1)*(z2 - z3) - (z0 - z1)*(r2*z3 - r3*z2)) / \
                        ((r0 - r1)*(z2 - z3) - (z0 - z1)*(r2-r3))

    if Pr >= r0 and Pr <= r1 and Pz >= z0 and Pz <= z1:
        hit = True
    elif Pr <= r0 and Pr >= r1 and Pz >= z0 and Pz <= z1:
        hit = True
    elif Pr >= r0 and Pr <= r1 and Pz <= z0 and Pz >= z1:
        hit = True
    elif Pr <= r0 and Pr >= r1 and Pz <= z0 and Pz >= z1:
        hit = True
    else:
        hit = False

    return [Pr, Pz, hit]

def true_lam_csd(c, dr=100, z=None):
    '''Return CSD from membrane currents as function along the coordinates
    of the electrode along z-axis. Inputs; c: cell.Cell object, dr: radius,
    z-coordinates of electrode.'''
    if type(z) != type(np.ndarray(shape=0)):
        raise ValueError, 'type(z) should be a np.ndarray'

    dz = abs(z[1] - z[0])
    CSD = np.zeros((z.size, c.tvec.size,))
    r_end = np.sqrt(c.xend**2 + c.yend**2)
    r_start = np.sqrt(c.xstart**2 + c.ystart**2)
    V = dz * np.pi * dr * dr

    for i in xrange(len(z)):
        aa0 = c.zstart < z[i] + dz/2
        aa1 = c.zend < z[i] + dz/2
        bb0 = c.zstart >= z[i] - dz/2
        bb1 = c.zend >= z[i] - dz/2
        cc0 = r_start < dr
        cc1 = r_end < dr
        ii = aa0 & bb0 & cc0 #startpoint inside V
        jj = aa1 & bb1 & cc1 #endpoint inside V

        for j in xrange(c.zstart.size):
            isum = 0.
            #calc fraction of source being inside control volume from 0-1
            if ii[j] and jj[j]:
                CSD[i,] = CSD[i, ] + c.imem[j, ] / V
            elif ii[j] and not jj[j]: #startpoint in V
                z0 = c.zstart[j]
                r0 = r_start[j]
                z1 = c.zend[j]
                r1 = r_end[j]
                L2 = (r1 - r0)**2 + (z1 - z0)**2

                z2 = [z[i]-dz/2, z[i]+dz/2, z[i]-dz/2]
                r2 = [0, 0, dr]
                z3 = [z[i]-dz/2, z[i]+dz/2, z[i]+dz/2]
                r3 = [dr, dr, dr]

                P = []
                for k in xrange(3):
                    P.append(PrPz(r0, z0, r1, z1, r2[k], z2[k], r3[k], z3[k]))
                    if P[k][2]:
                        vL2 = (P[k][0] - r0)**2 + (P[k][1] -z0)**2
                        frac = np.sqrt(vL2 / L2)
                CSD[i,] = CSD[i, ] + frac * c.imem[j, ] / V
            elif jj[j] and not ii[j]: #endpoint in V
                z0 = c.zstart[j]
                r0 = r_start[j]
                z1 = c.zend[j]
                r1 = r_end[j]
                L2 = (r1 - r0)**2 + (z1 - z0)**2

                z2 = [z[i]-dz/2, z[i]+dz/2, z[i]-dz/2]
                r2 = [0, 0, dr]
                z3 = [z[i]-dz/2, z[i]+dz/2, z[i]+dz/2]
                r3 = [dr, dr, dr]

                P = []
                for k in xrange(3):
                    P.append(PrPz(r0, z0, r1, z1, r2[k], z2[k], r3[k], z3[k]))
                    if P[k][2]:
                        vL2 = (r1 - P[k][0])**2 + (z1 - P[k][1])**2
                        frac = np.sqrt(vL2 / L2)
                CSD[i,] = CSD[i, ] + frac * c.imem[j, ] / V
            else:
                pass

    return CSD, z

def tru_sphershell_csd(c, r=np.arange(20, 500, 20)):
    '''Calculate the true csd for spherical shells around origo'''
    r_end = np.sqrt(c.xend*c.xend + c.yend*c.yend + c.zend*c.zend)
    r_start = np.sqrt(c.xstart*c.xstart + c.ystart*c.ystart + c.zstart*c.zstart)
    if r[0] > 0:
        r = np.concatenate((np.array([0]), r))
    V = 4 * np.pi / 3 * (r[1:]**3 - r[:-1]**3)
    currents = np.zeros((V.size, c.tvec.size))
    CSD = np.zeros((V.size, c.tvec.size))

    i = 0
    for v in V:
        aa = r_start < r[i+1]
        bb = r_start >= r[i]

        cc = r_end < r[i+1]
        dd = r_end >= r[i]

        ii = aa & bb    #start inside V
        jj = cc & dd    #end inside V

        #loop over elements
        for j in xrange(c.xstart.size):
            isum = 0
            if ii[j] and jj[j]:
                currents[i,] += c.imem[j, ]
            elif ii[j] and not jj[j]:
                r0 = r_start[j]
                r1 = r_end[j]
                L2 = (r0 - r1)**2
                if r1 < r[i]:
                    frac = np.sqrt( (r0 - r[i])**2 / L2)
                elif r1 >= r[i+1]:
                    frac = np.sqrt( (r[i+1] - r0)**2 / L2 )
                currents[i, ] += frac * c.imem[j, ]
            elif not ii[j] and jj[j]:
                r0 = r_start[j]
                r1 = r_end[j]
                L2 = (r0 - r1)**2
                if r0 < r[i]:
                    frac = np.sqrt( (r1 - r[i])**2 / L2)
                elif r0 >= r[i+1]:
                    frac = np.sqrt( (r[i+1] - r1)**2 / L2 )
                currents[i, ] += frac * c.imem[j, ]
            else:
                pass
        CSD[i, ] = currents[i, ] / v
        i += 1

    return CSD, r


def donut_csd(c, dr = 20, rmax=500, dz = 20, zlim = 500):
    '''Return the true CSD in donuts, with the z-axis in the center'''
    z = np.arange(-zlim, zlim + dz, dz)
    r = np.arange(0, rmax + dr, dr)
    cr = np.sqrt(c.xmid**2 + c.ymid**2)
    V = np.zeros((z.size, r.size-1))
    u = {}
    for i in xrange(z.size):
        for j in xrange(r.size-1):
            V[i, j] = dz * np.pi * (r[j+1]**2 - r[j]**2)

    CSD = np.zeros((z.size, r.size-1, c.tvec.size))

    for i in xrange(z.size):
        aa = c.zmid < z[i] + dz/2
        bb = c.zmid >= z[i] - dz/2
        for j in xrange(r.size-1):
            cc = cr < r[j + 1]
            dd = cr >= r[j]
            u[i, j] = np.where(aa & bb & cc & dd)

    for i in xrange(z.size):
        for j in xrange(r.size-1):
            if len(u[i, j]) > 0:
                CSD[i, j, ] = c.imem[u[i, j], ].sum(axis=1) / V[i, j]

    return z, r, CSD


class Signal:
    """ LFPy signal processing """
    def __init__(self, dt=2**-4):
        print 'LFPy signal processing loaded!'
        self.dt = dt
        print '"dt" in class Signal is %g \n' % self.dt
        self.data = {}

    def filter_signal(self, signal, dt=2**-4, order=1, filter_type='low',  \
            ftype='butter', fcut=100., R=0.5, convolution=True):
        '''Return filtered signal using a specified type of filter'''
        x = signal
        samplefreq = 1000./dt
        Wn = fcut/samplefreq
        #print('normalized Wn = '+str(Wn))
        if Wn > 1.:
            print('Sample frequency %i is less than the cut-off frequency %i!' \
        % (samplefreq, fcut))
            print('Filtered signal may be a mess!')
        N = order

        if ftype=='butter':
            [b, a] = ss.butter(N, Wn, btype=filter_type)
        elif ftype=='cheby1':
            [b, a] = ss.cheby1(N, R, Wn, btype=filter_type)
        elif ftype == 'boxcar':
            b = ss.boxcar(N)
            a = np.array([b.sum()])
        elif ftype == 'hamming':
            b = ss.hamming(N)
            a = np.array([b.sum()])
        elif ftype == 'triangular':
            b = ss.triang(N)
            a = np.array([b.sum()])
        elif ftype == 'gaussian':
            b = ss.gaussian(N[0], N[1])
            a = np.array([b.sum()])
        else:
            print '''ftype should be butter, cheby1, boxcar, hamming, gaussian,
        or triangular'''
            [b, a] = [1, 1]


        if convolution:
            filtered = np.convolve(x, b/a.sum(), 'same')
        else:
            filtered = ss.lfilter(b, a, x)

        return filtered


class Population(object):
    '''Population stuff'''
    def __init__(self,n=10, radius=100, z_min=-100, z_max=100,
                 tstart=0, tstop=50):
        self.n = n
        self.radius = radius
        self.z_min = z_min
        self.z_max = z_max
        self.tstart = tstart
        self.tstop = tstop

    def draw_rand_pos(self):
        x = pl.empty(self.n)
        y = pl.empty(self.n)
        z = pl.empty(self.n)
        for i in xrange(self.n):
            x[i] = (pl.rand()-0.5) * self.radius*2
            y[i] = (pl.rand()-0.5) * self.radius*2
            while pl.sqrt(x[i]**2 + y[i]**2) >= self.radius:
                x[i] = (pl.rand()-0.5)*self.radius*2
                y[i] = (pl.rand()-0.5)*self.radius*2
        z = pl.rand(self.n)*(self.z_max - self.z_min) + self.z_min

        r = pl.sqrt(x**2 + y**2 + z**2)

        soma_pos = {
            'xpos' : x,
            'ypos' : y,
            'zpos' : z,
            'r' : r
        }
        return soma_pos

    def draw_rand_sphere_pos(self):
        azimuth = (pl.rand(self.n)-0.5)*2*pl.pi
        zenith = pl.arccos(2*pl.rand(self.n) -1)
        r = pl.rand(self.n)**(2./3.)*self.radius

        x = r*pl.sin(zenith)*pl.cos(azimuth)
        y = r*pl.sin(zenith)*pl.sin(azimuth)
        z = r*pl.cos(zenith)

        soma_pos = {
            'xpos' : x,
            'ypos' : y,
            'zpos' : z,
            'r' : r
        }
        return soma_pos

    def draw_rand_gaussian_pos(self,min_r = pl.array([])):
        '''optional min_r, array or tuple of arrays on the form
        array([[r0,r1,...,rn],[z0,z1,...,zn]])'''

        x = pl.normal(0,self.radius,self.n)
        y = pl.normal(0,self.radius,self.n)
        z = pl.normal(0,self.radius,self.n)

        min_r_z = {}
        if pl.size(min_r) > 0: # != False:
            if type(min_r)==type(()):
                for j in xrange(pl.shape(min_r)[0]):
                    min_r_z[j] = pl.interp(z,min_r[j][0,],min_r[j][1,])
                    if j > 0:
                        [w] = pl.where(min_r_z[j] < min_r_z[j-1])
                        min_r_z[j][w] = min_r_z[j-1][w]
                    minrz = min_r_z[j]

            else:
                minrz = pl.interp(z, min_r[0], min_r[1])

            R_z = pl.sqrt(x**2 + y**2)
            [u] = pl.where(R_z < minrz)

            while len(u) > 0:
                for i in xrange(len(u)):
                    x[u[i]] = pl.normal(0, self.radius,1)
                    y[u[i]] = pl.normal(0, self.radius,1)
                    z[u[i]] = pl.normal(0, self.radius,1)
                    if type(min_r)==type(()):
                        for j in xrange(pl.shape(min_r)[0]):
                            min_r_z[j][u[i]] = pl.interp(z[u[i]],min_r[j][0,],min_r[j][1,])
                            if j > 0:
                                [w] = pl.where(min_r_z[j] < min_r_z[j-1])
                                min_r_z[j][w] = min_r_z[j-1][w]
                            minrz = min_r_z[j]
                    else:
                        minrz[u[i]] = pl.interp(z[u[i]],min_r[0,],min_r[1,])
                R_z = pl.sqrt(x**2 + y**2)
                [u] = pl.where(R_z < minrz)

        soma_pos = {
            'xpos' : x,
            'ypos' : y,
            'zpos' : z,
        }
        return soma_pos

    def get_normal_input_times(self, mu=10, sigma=1):
        ''' generates n normal-distributed prosesses with mean mu and deviation sigma'''
        times = pl.normal(mu,sigma,self.n)
        for i in xrange(self.n):
            while times[i] <= self.tstart or times[i] >= self.tstop:
                times[i] = pl.normal(mu, sigma)
        return times