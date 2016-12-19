#!/usr/bin/env python
'''
Main plotting script. Output in terms of pdf figures will be stored in this
folder while some other data is printed on screen. 

Usage:
    
    python create_figures.py
    

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
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import cPickle
import pickle
import scipy.signal as ss
import scipy.ndimage as sn
import scipy.integrate as si
from NeuroTools.parameters import ParameterSpace, ParameterSet, ParameterRange
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.collections import PolyCollection
import matplotlib.patheffects as path_effects
import neo
import quantities as pq
from elephant.current_source_density import icsd
import LFPy
from cycler import cycler
from hashlib import md5
from glob import glob
from initialize_simulations import get_md5s


import warnings
warnings.filterwarnings('error')

figparams = {
    'figure.subplot.bottom': 0.075,
    'figure.subplot.hspace': 0.2,
    'figure.subplot.left': 0.075,
    'figure.subplot.right': 0.925,
    'figure.subplot.top': 0.925,
    'figure.subplot.wspace': 0.2,
    'font.family' : 'sans-serif',
    'font.size' : 12,
    'axes.titlesize' : 14,
    'legend.frameon' : False,
    'legend.loc' : 'best',
    'text.usetex' : True,
    'text.latex.preamble': ['\usepackage[cm]{sfmath}', '\usepackage{upgreek}']
}
pl.rcParams.update(pl.rcParamsDefault)
pl.rcParams.update(figparams)


flattenlist = lambda lst: sum(lst, [])


def plot_signal_sum(ax, tvec, fname='LFPsum.h5', unit='mV', scaling_factor=1.,
                    ylabels=True, scalebar=True, vlimround=None,
                    T=[0, 5], ylim=[0, 16], colors='k',
                    label='', transient=0, rasterized=False):
    '''
    on axes plot the summed LFP contributions
    
    Arguments
    ---------
    ax : matplotlib.axes.AxesSubplot
    tvec : np.ndarray
        1D array with corresponding time points of fname data
    fname : str/np.ndarray
        path to h5 file or ndim=2 numpy.ndarray 
    unit : str
        scalebar unit
    scaling_factor : float
        scaling factor
    ylabels : bool
        show labels on y-axis
    scalebar : bool
        show scalebar in plot
    vlimround : None/float
        override autoscaling of data and scalebar
    T : list
        [tstart, tstop], which timeinterval
    ylim : list of floats
        see plt.gca().set_ylim
    color : str/colorspec tuple
        color of shown lines
    label : str
        line labels
    
    
    Returns
    -------    
    vlimround : float
        scalebar scaling factor, i.e., to match up plots
    
    '''

    if type(fname) == str and os.path.isfile(fname):
        f = h5py.File(fname)
        data = f['data'].value
        f.close()
    elif type(fname) == np.ndarray and fname.ndim==2:
        data = fname
    else:
        raise Exception, 'type(fname)={} not str or numpy.ndarray'.format(type(fname))

    zvec = np.arange(data.shape[0]) - .5
    vlim = abs(data).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    
    yticklabels=[]
    yticks = []
    
    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(tvec, -data[i] / vlimround + data.shape[0] - z, lw=0.5,
                    color='k',
                    rasterized=rasterized, label=label, clip_on=False)
        else: 
            ax.plot(tvec, -data[i] / vlimround + data.shape[0] - z, lw=0.5,
                    color='k',
                    rasterized=rasterized, clip_on=False)
        if i % 2 == 0:
            yticklabels.append('%i' % (i+1))
        else:
            yticklabels.append('')
        yticks.append(z)
     
    if scalebar:
        ax.plot([tvec[-1]-np.diff(T)*0.95, tvec[-1]-np.diff(T)*0.95],
                [15.1, 16.1], lw=2, color='k', clip_on=False, zorder=10)

    
    return vlimround


def plot_morpho(ax, cell, color='gray', rasterized=False, zorder=0):
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(zip(x, z))

    polycol = PolyCollection(zips,
                             linewidths=(0.25),
                             edgecolors=color,
                             facecolors=color,
                             zorder=zorder,
                             rasterized=rasterized)

    ax.add_collection(polycol)


def uncPickle(fil):
    f = file(fil, 'rb')
    stuff = cPickle.load(f)
    f.close
    return stuff


def get_idx_proximal(cell, r=25.):
    '''find which idx are within radius from midpoint of soma'''

    r2 = pl.array((cell.xmid - cell.somapos[0])**2 +
        (cell.ymid - cell.somapos[1])**2 +
        (cell.zmid - cell.somapos[2])**2)

    return pl.where(r2 <= r**2)


def plot_signal_sum(ax, tvec, fname='LFPsum.h5', unit='mV', scaling_factor=1.,
                    ylabels=True, scalebar=True, vlimround=None,
                    T=[0, 5], ylim=[0, 16], colors='k',
                    label='', transient=0, rasterized=False):
    '''
    on axes plot the summed LFP contributions
    
    Arguments
    ---------
    ax : matplotlib.axes.AxesSubplot
    tvec : np.ndarray
        1D array with corresponding time points of fname data
    fname : str/np.ndarray
        path to h5 file or ndim=2 numpy.ndarray 
    unit : str
        scalebar unit
    scaling_factor : float
        scaling factor
    ylabels : bool
        show labels on y-axis
    scalebar : bool
        show scalebar in plot
    vlimround : None/float
        override autoscaling of data and scalebar
    T : list
        [tstart, tstop], which timeinterval
    ylim : list of floats
        see plt.gca().set_ylim
    color : str/colorspec tuple
        color of shown lines
    label : str
        line labels
    
    
    Returns
    -------    
    vlimround : float
        scalebar scaling factor, i.e., to match up plots
    
    '''

    if type(fname) == str and os.path.isfile(fname):
        f = h5py.File(fname)
        data = f['data'].value
        f.close()
    elif type(fname) == np.ndarray and fname.ndim==2:
        data = fname
    else:
        raise Exception, 'type(fname)={} not str or numpy.ndarray'.format(type(fname))

    zvec = np.arange(data.shape[0]) - .5
    vlim = abs(data).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    
    yticklabels=[]
    yticks = []
    
    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(tvec, -data[i] / vlimround + data.shape[0] - z, lw=0.5,
                    color='k',
                    rasterized=rasterized, label=label, clip_on=False)
        else: 
            ax.plot(tvec, -data[i] / vlimround + data.shape[0] - z, lw=0.5,
                    color='k',
                    rasterized=rasterized, clip_on=False)
        if i % 2 == 0:
            yticklabels.append('%i' % (i+1))
        else:
            yticklabels.append('')
        yticks.append(z)
     
    if scalebar:
        ax.plot([tvec[-1]-np.diff(T)*0.95, tvec[-1]-np.diff(T)*0.95],
                [15.1, 16.1], lw=2, color='k', clip_on=False, zorder=10)

    
    return vlimround


def figure_1(NRS=40, NFS=10, synapses=False, offsets=pl.array([]),
             PS=['PS_simres_RS', 'PS_simres_FS'], index=0):

    id_RS = get_md5s(PS[0])[index]
    id_FS = get_md5s(PS[1])[index]
    c_savedRS = uncPickle(os.path.join('savedata', id_RS, 'c_savedPickle.cpickle'))
    c_savedFS = uncPickle(os.path.join('savedata', id_FS, 'c_savedPickle.cpickle'))
    

    fig = pl.figure(figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.35, bottom=0.1, left=0.05)

    ax = fig.add_subplot(141, frameon=False, xticks=[], yticks=[], clip_on=True)

    #neurons
    #TC
    ax.add_patch(mpatches.Circle(xy=(0, -4), radius=1, edgecolor='k',
                                 facecolor='w', linewidth=2))
    #RS
    ax.add_patch(mpatches.Circle(xy=(-2, 4), radius=0.75, edgecolor='k',
                                 facecolor='w', linewidth=2))
    ax.add_patch(mpatches.Circle(xy=(-2.1, 4), radius=0.75, edgecolor='k',
                                 facecolor='w', zorder=-1, linewidth=1))
    ax.add_patch(mpatches.Circle(xy=(-2.2, 4), radius=0.75, edgecolor='k',
                                 facecolor='w', zorder=-2, linewidth=1))
    ax.add_patch(mpatches.Circle(xy=(-2.3, 4), radius=0.75, edgecolor='k',
                                 facecolor='w', zorder=-3, linewidth=1))
    #FS
    ax.add_patch(mpatches.Circle(xy=(2, 4), radius=0.75, edgecolor='k',
                                 facecolor='w', linewidth=2))
    ax.add_patch(mpatches.Circle(xy=(1.9, 4), radius=0.75, edgecolor='k',
                                 facecolor='w', zorder=-1, linewidth=1))
    ax.add_patch(mpatches.Circle(xy=(1.8, 4), radius=0.75, edgecolor='k',
                                 facecolor='w', zorder=-2, linewidth=1))
    ax.add_patch(mpatches.Circle(xy=(1.7, 4), radius=0.75, edgecolor='k',
                                 facecolor='w', zorder=-3, linewidth=1))

    #electrodes
    ax.add_patch(mpatches.Polygon(xy=pl.array([[0.75, -3],[3.25, 2], [3, 2]]),
                                  edgecolor='k', facecolor='r'))
    ax.add_patch(mpatches.Polygon(xy=pl.array([[-2, 4],[0.5, 9], [0.25, 9]]),
                                  edgecolor='grey', facecolor='grey'))
    ax.add_patch(mpatches.Polygon(xy=pl.array([[2, 4],[4.5, 9], [4.25, 9]]),
                                  edgecolor='grey', facecolor='grey'))

    #MEA
    ax.add_patch(mpatches.Polygon(xy=pl.array([ [-4.15, 0], [-3.85, 0],
                                                [-3.6, 8], [-4.4, 8] ]),
                                  edgecolor='k', facecolor='r'))
    zz = pl.linspace(1, 7, 15)
    for z in zz:
        ax.add_patch(mpatches.Circle(xy=(-4, z), radius=0.15, edgecolor='k',
                                     facecolor='w', linewidth=1))

    #connections
    ax.add_line(mlines.Line2D(xdata=(0, 0, -2, -2), ydata=(-3, 1.5, 2, 3),
                              linewidth=2, color='k'))
    ax.add_line(mlines.Line2D(xdata=(0, 0, 2, 2), ydata=(-3, 1.5, 2, 3),
                              linewidth=2, color='k'))

    ax.add_patch(mpatches.Rectangle(xy=(-2.25, 2.75), width=0.5, height=0.25,
                                    edgecolor='k', facecolor='k'))
    ax.add_patch(mpatches.Rectangle(xy=(1.75, 2.75), width=0.5, height=0.25,
                                    edgecolor='k', facecolor='k'))

    #annotations
    ax.annotate('RS', xy=(-1, 4))
    ax.annotate('TC', xy=(1.25,-4))
    ax.annotate('FS', xy=(3, 4))
    ax.annotate('el.i', xy=(0.25, 9), xytext=(2,10.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    ax.annotate('el.i  ', xy=(4.5, 9), xytext=(2,10.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"))
    ax.annotate('el.e', xy=(3, 2), xytext=(5,7.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"))
    ax.annotate('ME', xy=(-4, 8), xytext=(-3, 10.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    pl.axis('equal')

    ax.text(0.0, 1.0, r'$\textbf{A}$',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)


    ax = fig.add_subplot(142, frameon=True, xticks=[], yticks=[], clip_on=True)
    ax.plot(pl.zeros(16), pl.linspace(-700, 800, 16),
                    color='w', marker='o', linestyle='None')
    for i in [0, 1, 2, 13, 14, 15]:
        ax.text(-40, pl.linspace(-700, 800, 16)[i], 'Pos. %i' % (16-i),
                horizontalalignment='right',
                verticalalignment='center')

    for n in xrange(len(c_savedRS.items())):
        color = [pl.rand()*0.3+0.7, pl.rand()*0.3+0.7, pl.rand()*0.3+0.7]
        if len(offsets > 0):
            for j in xrange(len(offsets.keys())):
                ax.plot(offsets[j]['x_n'], offsets[j]['z_n'], marker='.',
                        color='k', linestyle='None')

        if n < NRS:
            plot_morpho(ax, c_savedRS[n], color,
                        zorder=-abs(c_savedRS[n].somapos[1]))
            if synapses:
                for i in xrange(len(c_savedRS[n].synapses)):
                    c_savedRS[n].synapses[i].update_pos(c_savedRS[n])
                    ax.scatter([c_savedRS[n].synapses[i].x],
                        [c_savedRS[n].synapses[i].z],
                        marker='o', s=10, zorder=0, facecolor='r',
                        edgecolor='gray', linewidth=0.25,
                        )



    for n in xrange(len(c_savedFS.items())):
        color = [pl.rand()*0.3+0.7, pl.rand()*0.3+0.7, pl.rand()*0.3+0.7]
        if len(offsets > 0):
            for j in xrange(len(offsets.keys())):
                ax.plot(offsets[j]['x_n'], offsets[j]['z_n'], marker='.',
                        color='k', linestyle='None')

        if n < NFS:
            plot_morpho(ax, c_savedFS[n], color,
                        zorder=-abs(c_savedRS[n].somapos[1]))
            if synapses:
                for i in xrange(len(c_savedFS[n].synapses)):
                    c_savedFS[n].synapses[i].update_pos(c_savedFS[n])
                    ax.scatter([c_savedFS[n].synapses[i].x],
                        [c_savedFS[n].synapses[i].z],
                        marker='o', s=10, zorder=0, facecolor='r',
                        edgecolor='gray', linewidth=0.25,
                        )


    ax.plot([100, 100],[600, 700], lw=3, color='k')
    ax.text(120, 650, '100 $\upmu$m',
            horizontalalignment='left',
            verticalalignment='center')


    ax.axis(ax.axis('equal'))
    ax.yaxis.set_ticks(pl.arange(-250, 300, 250))
    ax.xaxis.set_ticks(pl.arange(-250, 300, 250))

    adjust_spines(ax,['left', 'bottom'])
    ax.spines['left'].set_bounds( -250, 250)
    ax.set_yticks([-250, 0, 250])
    ax.spines['bottom'].set_bounds( -250, 250)
    ax.set_xticks([-250, 0, 250])
    pl.ylabel(r'$z$ ($\mathrm{\upmu}$m)', ha='center', va='center')
    newaxis = pl.array(ax.axis()) * 1.1
    ax.axis(newaxis)
    ax.text(0.5, -0.1, r'$x$ ($\mathrm{\upmu}$m)',
            horizontalalignment='center',
            verticalalignment='center',transform=ax.transAxes)
    ax.text(0.0, 1.0, r'$\textbf{B}$',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)




    def plot_morpho_idx(ax, cell, idxlist, colors=['k', 'r'], rasterized=False,
                        zorder=0):
        polys = cell.get_idx_polygons()

        for i, idx in enumerate(idxlist):
            zips = []
            if idx.size <= 1:
                try:
                    xz = polys[idx[0]]
                except DeprecationWarning:
                    raise Exception
                zips.append(zip(xz[0], xz[1]))
            else:
                for j in idx:
                    xz = polys[j]
                    zips.append(zip(xz[0], xz[1]))

            polycol = PolyCollection(zips,
                                     linewidths=(0.5),
                                     edgecolors=colors[i],
                                     facecolors=colors[i],
                                     zorder=zorder,
                                     rasterized=rasterized)

            ax.add_collection(polycol)

    ax = fig.add_subplot(122, frameon=True, xticks=[], yticks=[], clip_on=True)
    c = c_savedRS[0]

    #recreate LFPy cell object
    cell = LFPy.Cell(morphology=c.morphology,
                     rm = c.rm, cm=c.cm, Ra=c.Ra,
                     nsegs_method = 'lambda_f',
                     lambda_f = 1000, pt3d=False)

    #adjust nseg for soma
    cell.set_pos(-150, 0, 0)
    cell.set_rotation(x=pl.pi/2)
    synidx = cell.get_idx(['dend', 'apic'])
    otheridx = pl.where(pl.in1d(pl.arange(cell.totnsegs), synidx)==False)[0]
    colors = ['k', 'r']
    plot_morpho_idx(ax, cell, [otheridx, synidx], colors)


    c = c_savedFS[0]
    cell = LFPy.Cell(morphology=c.morphology,
                     rm = c.rm, cm=c.cm, Ra=c.Ra,
                     nsegs_method = 'lambda_f',
                     lambda_f = 1000, pt3d=False)

    cell.set_pos(150, 0, 0)
    cell.set_rotation(x=pl.pi/2)
    synidx = get_idx_proximal(cell, r=50.)[0]
    otheridx = pl.where(pl.in1d(pl.arange(cell.totnsegs), synidx)==False)[0]
    colors = ['k', 'r']
    plot_morpho_idx(ax, cell, [otheridx, synidx], colors)

    pl.axis('equal')

    ax.yaxis.set_ticks(pl.arange(-100, 150, 100))
    ax.xaxis.set_ticks(pl.arange(-100, 150, 100))

    adjust_spines(ax,['left', 'bottom'])
    ax.spines['left'].set_bounds( -100, 100)
    ax.set_yticks([-100, 0, 100])
    ax.spines['bottom'].set_bounds(-100, 100)
    ax.set_xticks([-100, 0, 100])
    ax.text(0.5, -0.1, r'$x$ ($\mathrm{\upmu}$m)',
            horizontalalignment='center',
            verticalalignment='center',transform=ax.transAxes)
    ax.text(-0.1, 0.5, r'$z$ ($\mathrm{\upmu}$m)',
            horizontalalignment='center',
            verticalalignment='center',
            rotation='vertical',transform=ax.transAxes)
    ax.text(0.0, 1.0, r'$\textbf{C}$',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    ax.annotate('RS cell', xy=(-150, pl.axis()[3]), horizontalalignment='center')
    ax.annotate('FS cell', xy=(150, pl.axis()[3]), horizontalalignment='center')


def figure_2(data='lfp_filtered'):
    row1ylim = [pl.array([datas_RS['syn_i'].min(),
                          datas_RS['EPSC'].min()]).min(), 0]

    fig = pl.figure(figsize=(12, 6))
    fig.subplots_adjust(wspace=0.25)

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    mytitles = [
        'Synaptic projection',
        'Population LFP',
        r'Population CSD, $r_\mathrm{CSD}=200 \upmu$m',
        'Synaptic currents',
        'EPSCs',
        'EPSPs'
    ]

    i = 0
    ax = fig.add_subplot(2,3,i+1)
    ax.scatter(datas_RS['som_pos0_x'][:500], datas_RS['som_pos0_z'][:500]-13,
               marker='o', color=(0.75,0.75,0.75), label='soma pos., i', s=5,
               rasterized=False)
    ax.scatter(datas_RS['som_pos_x'], datas_RS['som_pos_z']-13,
               marker='o', color='k', label='soma pos., f', s=5,
               rasterized=False)
    ax.scatter(datas_RS['syn_pos_x'], datas_RS['syn_pos_z']-13,
               marker='|', facecolor='r', edgecolor='gray', linewidth=1,
               label='syn. pos.', s=1.5,
               # marker='o', facecolor='r', edgecolor='gray', linewidth=0.25,
               # label='syn. pos.', s=2.5,
               rasterized=False)
    ax.axis('equal')
    ax.xaxis.set_ticks([])
    ax.xaxis.set_ticks(pl.arange(-250, 300, 250))
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticks(pl.arange(-250, 300, 250))

    adjust_spines(ax,['left', 'bottom'])
    ax.spines['left'].set_bounds( -250, 250 )
    ax.set_yticks([-250,0,250])
    ax.spines['bottom'].set_bounds(-250, 250)
    ax.set_xticks([-250, 0, 250])
    pl.ylabel(r'$z$ ($\upmu$m)', ha='center', va='center')
    pl.xlabel(r'$x$ ($\upmu$m)', {'verticalalignment' : 'top'})
    ax.text(0.5, 1.55, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1.7, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)

    divider = make_axes_locatable(ax)
    axHistx = divider.append_axes("right", 0.5, pad=0)
    axHisty = divider.append_axes("top", 0.5, pad=0)

    binwidth = 10
    binsy = pl.arange(ax.axis()[2], ax.axis()[3] + binwidth, binwidth)
    binsx = pl.arange(ax.axis()[0], ax.axis()[1] + binwidth, binwidth)
    hist1 = axHistx.hist(datas_RS['syn_pos_z']-13, bins=binsy,
                         orientation='horizontal', histtype='stepfilled',
                         color='r',)
    hist2 = axHisty.hist(datas_RS['syn_pos_x'], bins=binsx,
                         histtype='stepfilled', color='r',)

    nmax = hist1[0].max()
    axHistx.axis([0, nmax, ax.axis()[2], ax.axis()[3]])
    axHistx.yaxis.set_ticks([])
    adjust_spines(axHistx, ['bottom'])
    axHistx.xaxis.set_ticks([nmax])

    nmax = hist2[0].max()
    axHisty.xaxis.set_ticks([])
    adjust_spines(axHisty, ['right'])
    axHisty.yaxis.set_ticks([nmax])
    axHisty.yaxis.set_ticks_position('right')

    i += 1

    ax = fig.add_subplot(2,3,i+1)
    clim = abs(datas_RS[data]).max()
    im = ax.imshow(datas_RS[data], cmap=pl.get_cmap('PRGn', 51),
              origin='bottom', clim=(-clim, clim), interpolation='nearest',
              rasterized=False,
              extent=extent)

    vlimround = plot_signal_sum(ax, datas_RS['tvec'], datas_RS[data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)

    # find and mark global minima
    min = np.where(datas_RS[data] == datas_RS[data].min())
    ax.plot(datas_RS['tvec'][min[1]], 16-min[0]+.5, 'wo')
    print('global RS LFP minima: position {}, time {} s.'.format(16-min[0][0],
                                                                 datas_RS['tvec'][min[1]][0]))
    
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    pl.setp(ax, xticklabels=[])
    pl.ylabel('Position', ha='left', va='center')


    bbox = pl.array(ax.get_position())
    cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                             0.01, bbox[1][1]-bbox[0][1]-0.1])
    cbar = pl.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
    cbar.ax.set_yticklabels([-1,0,1])


    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1

    
    ax = fig.add_subplot(2,3,i+1)
    clim = abs(datas_RS['CSD_filtered']*1E6).max()
    im = ax.imshow(datas_RS['CSD_filtered']*1E6, cmap=pl.get_cmap('bwr_r', 51),
              origin='bottom', clim=(-clim, clim), interpolation='nearest',
              rasterized=False,
              extent=extent)
    vlimround = plot_signal_sum(ax, datas_RS['tvec'], datas_RS['CSD_filtered']*1E6,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    # find and mark global minima
    min = np.where(datas_RS['CSD_filtered'] == datas_RS['CSD_filtered'].min())
    ax.plot(datas_RS['tvec'][min[1]], 16-min[0]+.5, 'wo')
    print('global RS CSD minima: position {}, time {} s.'.format(16-min[0][0],
                                                                 datas_RS['tvec'][min[1]][0]))

    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    pl.setp(ax, yticklabels=[])
    pl.setp(ax, xticklabels=[])
    
    bbox = pl.array(ax.get_position())
    cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                             0.01, bbox[1][1]-bbox[0][1]-0.1])
    cbar = pl.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
    cbar.ax.set_yticklabels([-1,0,1])
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1


    ax = fig.add_subplot(2,3,i+1)
    for j in xrange(datas_RS['syn_i'].shape[0]):
        ax.plot(tvec, datas_RS['syn_i'][j, ],
                color=(pl.rand()*0.3+0.7, pl.rand()*0.3+0.7, pl.rand()*0.3+0.7),
                clip_on=False,)
    ax.plot(tvec, datas_RS['mean_syn_i'], color='k', linewidth=3, clip_on=False,)
    ax.plot(tvec, datas_RS['LS_syn_i'], color='r', ls='--', linewidth=3,
            clip_on=False,)
    pl.axis('tight')
    pl.xlabel(r'$t$ (ms)')
    pl.ylabel(r'$i_\mathrm{syn}$ (nA)', ha='center', va='center')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.2, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1

    ax = fig.add_subplot(2,3,i+1)


    for j in xrange(datas_RS['EPSC'].shape[0]):
        ax.plot(tvec, datas_RS['EPSC'][j, ],
                color=(pl.rand()*0.3+0.7, pl.rand()*0.3+0.7, pl.rand()*0.3+0.7),
                clip_on=False,)
    ax.plot(tvec, datas_RS['mean_EPSC'], color='k', linewidth=3, clip_on=False,)
    ax.plot(tvec, datas_RS['LS_EPSC'], color='r', ls='--', linewidth=3,
            clip_on=False,)

    # add median and upper/lower quartile of EPSC
    ax.plot(tvec, np.median(datas_RS['EPSC'], axis=0), 'b', lw=1.5)
    ax.plot(tvec, np.percentile(datas_RS['EPSC'], 25, axis=0), 'b:', lw=1.5)
    ax.plot(tvec, np.percentile(datas_RS['EPSC'], 75, axis=0), 'b:', lw=1.5)
    
    ax.axis([tvec[0], tvec[-1], row1ylim[0], row1ylim[1]])
    pl.axis('tight')
    pl.xlabel(r'$t$ (ms)')
    pl.ylabel(r'$i_\mathrm{EPSC}$ (nA)', ha='center', va='center')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1



    ax = fig.add_subplot(2,3,i+1)

    for j in xrange(datas_RS['somav'].shape[0]):
        ax.plot(tvec, datas_RS['somav'][j, ],
                color=(pl.rand()*0.3+0.7, pl.rand()*0.3+0.7, pl.rand()*0.3+0.7),
                clip_on=False,)
    # mean EPSP
    ax.plot(tvec, datas_RS['mean_EPSP'], color='k', linewidth=3, clip_on=False,)
    # fit to mean EPSP
    ax.plot(tvec, datas_RS['LS_EPSP'], color='r', ls='--', linewidth=3,
            clip_on=False,)

    # add median and upper/lower quartile of EPSP
    ax.plot(tvec, np.median(datas_RS['somav'], axis=0), 'b', lw=1.5)
    ax.plot(tvec, np.percentile(datas_RS['somav'], 25, axis=0), 'b:', lw=1.5)
    ax.plot(tvec, np.percentile(datas_RS['somav'], 75, axis=0), 'b:', lw=1.5)
    
    ax.axis(ax.axis('tight'))
    pl.ylabel(r'$V_\mathrm{EPSP}$ (mV)', ha='center', va='center')
    pl.xlabel(r'$t$ (ms)')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1


def figure_3(data='lfp_filtered'):

    fig = pl.figure(figsize=(12, 6))
    fig.subplots_adjust(wspace=0.25)


    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mytitles = [
        'Synaptic projection',
        'Population LFP',
        r'Population CSD, $r_\mathrm{CSD}=200 \upmu$m',
        'Synaptic currents',
        'EPSCs',
        'EPSPs'
    ]

    i = 0
    ax = fig.add_subplot(2,3,i+1)
    ax.scatter(datas_FS['som_pos0_x'][:500], datas_FS['som_pos0_z'][:500],
               marker='o', color=(0.75,0.75,0.75), label='soma pos., i.', s=5,
               rasterized=False)
    ax.scatter(datas_FS['som_pos_x'], datas_FS['som_pos_z'],
               marker='o', color='k', label='soma pos., f.', s=5,
               rasterized=False)
    ax.scatter(datas_FS['syn_pos_x'], datas_FS['syn_pos_z'],
               marker='|', facecolor='r', edgecolor='gray', linewidth=1,
               label='syn. pos.', s=1.5,
               #     marker='o', facecolor='r', edgecolor='gray', linewidth=0.25,
               # label='syn. pos.', s=2.5,
               rasterized=False)
    ax.axis('equal')
    ax.xaxis.set_ticks([])
    ax.xaxis.set_ticks(pl.arange(-250, 300, 250))
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticks(pl.arange(-250, 300, 250))

    adjust_spines(ax,['left', 'bottom'])
    ax.spines['left'].set_bounds( -250, 250 )
    ax.set_yticks([-250,0,250])
    ax.spines['bottom'].set_bounds(-250, 250)
    ax.set_xticks([-250, 0, 250])
    pl.ylabel(r'$z$ ($\upmu$m)', ha='center', va='center')
    pl.xlabel(r'$x$ ($\upmu$m)', {'verticalalignment' : 'top'})
    ax.text(0.5, 1.75, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1.7, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)


    divider = make_axes_locatable(ax)
    axHistx = divider.append_axes("right", 0.5, pad=0)
    axHisty = divider.append_axes("top", 0.5, pad=0)

    binwidth = 10
    binsy = pl.arange(ax.axis()[2], ax.axis()[3] + binwidth, binwidth)
    binsx = pl.arange(ax.axis()[0], ax.axis()[1] + binwidth, binwidth)
    hist1 = axHistx.hist(datas_FS['syn_pos_z'], bins=binsy,
                         orientation='horizontal',
                         histtype='stepfilled', color='r',)
    hist2 = axHisty.hist(datas_FS['syn_pos_x'], bins=binsx,
                         histtype='stepfilled', color='r',)

    nmax = hist1[0].max()
    axHistx.axis([0, nmax, ax.axis()[2], ax.axis()[3]])
    axHistx.yaxis.set_ticks([])
    adjust_spines(axHistx, ['bottom'])
    axHistx.xaxis.set_ticks([nmax])

    nmax = hist2[0].max()
    axHisty.xaxis.set_ticks([])
    adjust_spines(axHisty, ['right'])
    axHisty.yaxis.set_ticks([nmax])
    axHisty.yaxis.set_ticks_position('right')

    i += 1

    ax = fig.add_subplot(2,3,i+1)
    clim = abs(datas_FS[data]).max()
    im = ax.imshow(datas_FS[data], cmap=pl.get_cmap('PRGn', 51),
              origin='bottom', clim=(-clim, clim), interpolation='nearest',
              rasterized=False,
              extent=extent)

    vlimround = plot_signal_sum(ax, datas_FS['tvec'], datas_FS[data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)

    # find and mark global minima
    min = np.where(datas_FS[data] == datas_FS[data].min())
    ax.plot(datas_FS['tvec'][min[1]], 16-min[0]+.5, 'wo')
    print('global FS LFP minima: position {}, time {} s.'.format(16-min[0][0],
                                                                 datas_FS['tvec'][min[1]][0]))

    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    pl.setp(ax, xticklabels=[])
    pl.ylabel('Position', ha='center', va='center')

    bbox = pl.array(ax.get_position())
    cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                             0.01, bbox[1][1]-bbox[0][1]-0.1])
    cbar = pl.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
    cbar.ax.set_yticklabels([-1,0,1])
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1

    ax = fig.add_subplot(2,3,i+1)
    clim = abs(datas_FS['CSD_filtered']*1E6).max()
    im = ax.imshow(datas_FS['CSD_filtered']*1E6, cmap=pl.get_cmap('bwr_r', 51),
              origin='bottom', clim=(-clim, clim), interpolation='nearest',
              rasterized=False,
              extent=extent)

    vlimround = plot_signal_sum(ax, datas_FS['tvec'], datas_FS['CSD_filtered']*1E6,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    # find and mark global minima
    min = np.where(datas_FS['CSD_filtered'] == datas_FS['CSD_filtered'].min())
    ax.plot(datas_FS['tvec'][min[1]], 16-min[0]+.5, 'wo')
    print('global FS LFP minima: position {}, time {} s.'.format(16-min[0][0],
                                                                 datas_FS['tvec'][min[1]][0]))

    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    pl.setp(ax, yticklabels=[])
    pl.setp(ax, xticklabels=[])

    bbox = pl.array(ax.get_position())
    cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                             0.01, bbox[1][1]-bbox[0][1]-0.1])
    cbar = pl.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
    cbar.ax.set_yticklabels([-1,0,1])
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1

    ax = fig.add_subplot(2,3,i+1)
    for j in xrange(datas_FS['syn_i'].shape[0]):
        ax.plot(tvec, datas_FS['syn_i'][j, ],
                color=(pl.rand()*0.3+0.7, pl.rand()*0.3+0.7, pl.rand()*0.3+0.7),
                clip_on=False,)
    ax.plot(tvec, datas_FS['mean_syn_i'], color='k', linewidth=3, clip_on=False,)
    ax.plot(tvec, datas_FS['LS_syn_i'], color='r', ls='--', linewidth=3,
            clip_on=False,)
    pl.axis('tight')
    pl.xlabel(r'$t$ (ms)')
    pl.ylabel(r'$i_\mathrm{syn}$ (nA)', ha='center', va='center')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1

    ax = fig.add_subplot(2,3,i+1)
    for j in xrange(datas_FS['EPSC'].shape[0]):
        ax.plot(tvec, datas_FS['EPSC'][j, ],
                color=(pl.rand()*0.3+0.7, pl.rand()*0.3+0.7, pl.rand()*0.3+0.7),
                clip_on=False,)
    ax.plot(tvec, datas_FS['mean_EPSC'], color='k', linewidth=3, clip_on=False,)
    ax.plot(tvec, datas_FS['LS_EPSC'], color='r', ls='--', linewidth=3,
            clip_on=False,)
    # add median and upper/lower quartile of EPSC
    ax.plot(tvec, np.median(datas_FS['EPSC'], axis=0), 'b', lw=1.5)
    ax.plot(tvec, np.percentile(datas_FS['EPSC'], 25, axis=0), 'b:', lw=1.5)
    ax.plot(tvec, np.percentile(datas_FS['EPSC'], 75, axis=0), 'b:', lw=1.5)

    pl.axis('tight')
    pl.xlabel(r'$t$ (ms)')
    pl.ylabel(r'$i_\mathrm{EPSC}$ (nA)', ha='center', va='center')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1

    ax = fig.add_subplot(2,3,i+1)
    for j in xrange(datas_FS['somav'].shape[0]):
        ax.plot(tvec, datas_FS['somav'][j, ],
                color=(pl.rand()*0.3+0.7, pl.rand()*0.3+0.7, pl.rand()*0.3+0.7),
                clip_on=False,)
    ax.plot(tvec, datas_FS['mean_EPSP'], color='k', linewidth=3, clip_on=False,)
    ax.plot(tvec, datas_FS['LS_EPSP'], color='r', ls='--', linewidth=3,
            clip_on=False,)
    # add median and upper/lower quartile of EPSP
    ax.plot(tvec, np.median(datas_FS['somav'], axis=0), 'b', lw=1.5)
    ax.plot(tvec, np.percentile(datas_FS['somav'], 25, axis=0), 'b:', lw=1.5)
    ax.plot(tvec, np.percentile(datas_FS['somav'], 75, axis=0), 'b:', lw=1.5)

    pl.xlabel(r'$t$ (ms)')
    pl.ylabel(r'$V_\mathrm{EPSP}$ (mV)', ha='center', va='center')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1


def figure_9(dset='simres_R200', sigma=0.3, data='lfp_filtered'):
    nrows = 3

    fig = pl.figure(figsize=(12, nrows*3))
    fig.subplots_adjust(hspace=0.3, wspace=0.45)

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mytitles = [
        '$r_\mathrm{syn}=200$',
        'LFP',
        'CSD, $r_{\Omega_k}=200 \upmu$m',
        'high res. CSD, $r_{\Omega_k}=200 \upmu$m',
        r'$LS$ (norm) and $cc$',
        'tCSD',
        '$\delta$-iCSD',
        'spline-iCSD',
    ]


    # Input dictionaries for each method
    #test delta-iCSD for various diameters
    z_data = pl.linspace(100E-6, 1600E-6, 16) * pq.m
    delta_input = {
        'coord_electrode' : z_data,
        'sigma' : sigma * pq.S / pq.m ,        # extracellular conductivity
        'sigma_top' : sigma * pq.S / pq.m ,    # conductivity on top of cortex
        'f_type' : 'gaussian',  # gaussian filter
        'f_order' : (3, 1),     # 3-point filter, sigma = 1.
    }
    #various diameters tested
    diams = pl.array([100., 400, 1000, 2E9])*1E-6*pq.m

    # spline iCSD parameters
    spline_input = {
        'coord_electrode' : z_data,
        'diam' : 200E-6 * pq.m,
        'sigma' : sigma * pq.S / pq.m ,
        'sigma_top' : sigma * pq.S / pq.m ,
        'num_steps' : 76,      # Spatial CSD upsampling to N steps
        'tol' : 1E-12,
        'f_type' : 'gaussian',
        'f_order' : (19, 5),
    }
    
    # standard CSD parameters
    std_input = {
        'coord_electrode' : z_data,
        'sigma' : sigma * pq.S / pq.m,
        'vaknin_el' : True,
        'f_type' : 'gaussian',
        'f_order' : (3, 1),
    }

    i = 0
    ax = fig.add_subplot(nrows, 4, i+1)
    ax.scatter(datas_d[dset]['som_pos0_x'][:500], datas_d[dset]['som_pos0_z'][:500],
               marker='o', color=(0.75,0.75,0.75), label='no syn.', s=5,
               rasterized=False)
    ax.scatter(datas_d[dset]['som_pos_x'], datas_d[dset]['som_pos_z'],
               marker='o', color='k', label='w. syn.', s=5,
               rasterized=False)
    ax.scatter(datas_d[dset]['syn_pos_x'], datas_d[dset]['syn_pos_z'],
               marker='|', facecolor='r', edgecolor='gray', linewidth=1,
               label='syn. pos.', s=1.5,
               # marker='o', facecolor='r', edgecolor='gray', linewidth=0.25,
               # label='syn. pos.', s=2.5,
               rasterized=False)

    ax.scatter(datas_d[dset]['som_pos0_x'][:500], datas_d[dset]['som_pos0_y'][:500] - 800,
               marker='o', color=(0.75,0.75,0.75), label='no syn.', s=5,
               rasterized=False)
    ax.scatter(datas_d[dset]['som_pos_x'], datas_d[dset]['som_pos_y'] - 800,
               marker='o', color='k', label='w. syn.', s=5,
               rasterized=False)
    ax.scatter(datas_d[dset]['syn_pos_x'], datas_d[dset]['syn_pos_y'] - 800,
               marker='|', facecolor='r', edgecolor='gray', linewidth=1,
               label='syn. pos.', s=1.5,
               # marker='o', facecolor='r', edgecolor='gray', linewidth=0.25,
               # label='syn. pos.', s=2.5,
               rasterized=False)

    ax.axis('equal')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticks(pl.arange(-250, 300, 250))
    if i == 0:
        adjust_spines(ax,['left'])
        ax.spines['left'].set_bounds(-250, 250)
        ax.set_yticks([-250, 0, 250])
        ax.text(0.5, 1.02, mytitles[i],
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax.transAxes)
        ax.text(-0.4, 1.0, r'$\textbf{%s}$' % alphabet[i] ,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='black',
            transform=ax.transAxes)
        ax.text(-1300, 0, '$z$ ($\upmu$m)', rotation='vertical',
                horizontalalignment='center',
            verticalalignment='center')
    else:
        adjust_spines(ax,[])

    divider = make_axes_locatable(ax)
    axHistx = divider.append_axes("right", 0.2, pad=0)

    binwidth = 10
    bins = pl.arange(-1400, 400 + binwidth, binwidth)
    hist1 = axHistx.hist(datas_d[dset]['syn_pos_z'], bins=bins,
                         orientation='horizontal', histtype='stepfilled', color='r',)
    hist2 = axHistx.hist(datas_d[dset]['syn_pos_y']-800, bins=bins,
                         orientation='horizontal', histtype='stepfilled', color='r',)

    nmax = pl.array([hist1[0], hist2[0]]).max()
    axHistx.axis([0, nmax, -1400, 400])

    axHistx.yaxis.set_ticks([])
    adjust_spines(axHistx, ['bottom'])
    axHistx.xaxis.set_ticks([nmax])

    ax.set_ylim([-1400, 400])


    i += 1


    ax = fig.add_subplot(nrows, 4, i+1)
    clim = abs(datas_d[dset][data]).max()
    im = ax.imshow(datas_d[dset][data],
              cmap=pl.get_cmap('PRGn', 51),
              clim=(-clim, clim), interpolation='nearest', rasterized=False,
              extent=extent, origin='bottom')
    vlimround = plot_signal_sum(ax, datas_d[dset]['tvec'], datas_d[dset][data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    pl.setp(ax, xticklabels=[])
    pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.02, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.1, 1.0, r'$\textbf{%s}$' % alphabet[i] ,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)

    bbox = pl.array(ax.get_position())
    cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                             0.01, bbox[1][1]-bbox[0][1]-0.1])
    cbar = pl.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
    cbar.ax.set_yticklabels([-1,0,1])

    i += 1


    ax = fig.add_subplot(nrows, 4, i+1)
    clim = abs(datas_d[dset]['CSD_filtered']*1E6).max()
    im = ax.imshow(datas_d[dset]['CSD_filtered']*1E6, cmap=pl.get_cmap('bwr_r', 51),
              clim=(-clim, clim), interpolation='nearest', rasterized=False,
              extent=extent, origin='bottom')
    vlimround = plot_signal_sum(ax, datas_d[dset]['tvec'],
                                datas_d[dset]['CSD_filtered']*1E6,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    text = ax.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    pl.setp(ax, xticklabels=[])
    pl.setp(ax, yticklabels=[])
    ax.text(0.5, 1.02, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.1, 1.0, r'$\textbf{%s}$' % alphabet[i] ,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1

    ax = fig.add_subplot(nrows, 4, i+1)
    clim = abs(datas_d[dset]['CSD76ptF']*1E6).max()
    im = ax.imshow(datas_d[dset]['CSD76ptF']*1E6, cmap=pl.get_cmap('bwr_r', 51),
              clim=(-clim, clim), interpolation='nearest', rasterized=False,
              extent=extent, origin='bottom')
    vlimround = plot_signal_sum(ax, datas_d[dset]['tvec'],
                                datas_d[dset]['CSD76ptF'][::5, :]*1E6,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)

    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    text = ax.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    pl.setp(ax, xticklabels=[])
    pl.setp(ax, yticklabels=[])
    ax.text(0.5, 1.02, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.1, 1.0, r'$\textbf{%s}$' % alphabet[i] ,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)

    bbox = pl.array(ax.get_position())
    cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                             0.01, bbox[1][1]-bbox[0][1]-0.1])
    cbar = pl.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
    cbar.ax.set_yticklabels([-1,0,1])

    i += 1


    ax = fig.add_subplot(nrows, 4, i+1)
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", 1, pad=0.1)
    # ax2 = ax.twinx()
    ax.loglog(datas_d[dset]['my_diams_delta']/2.*1E6,
              datas_d[dset]['my_errors_delta']/ datas_d[dset]['my_errors_delta'].min(),
              # np.sqrt(datas_d[dset]['my_errors_delta'])/ np.sqrt(datas_d[dset]['my_errors_delta']).min(),
              'k', label=r'$\delta-$iCSD')
    # ax.loglog(dset['my_diams_step'], np.sqrt(dset['my_errors_step']), label='step')
    ax.loglog(datas_d[dset]['my_diams_spline']/2.*1E6,
              datas_d[dset]['my_errors_spline'] / datas_d[dset]['my_errors_spline'].min(),
              # np.sqrt(datas_d[dset]['my_errors_spline']) / np.sqrt(datas_d[dset]['my_errors_spline']).min(),
              'k--', label='spline-iCSD')
    
    
    # correlations
    ax2.semilogx(datas_d[dset]['my_diams_delta']/2.*1E6,
              datas_d[dset]['my_corrcoefs_delta'],
              'k', label=r'$CC_{\delta\mathrm{-iCSD}}$')
    ax2.semilogx(datas_d[dset]['my_diams_spline']/2.*1E6,
              datas_d[dset]['my_corrcoefs_spline'],
              'k--', label=r'$CC_\mathrm{spline-iCSD}$')
    
    cc = datas_d[dset]['my_corrcoefs_delta']
    print('max correlation coefficient CSD vs delta-iCSD: {} for radius {} um'.format(cc.max(), datas_d[dset]['my_diams_delta'][cc==cc.max()]/2.*1E6))
    cc = datas_d[dset]['my_corrcoefs_spline']
    print('max correlation coefficient CSD vs spline-iCSD: {} for radius {} um'.format(cc.max(), datas_d[dset]['my_diams_delta'][cc==cc.max()]/2.*1E6))
    
    ax.axis(ax.axis('tight'))
    ax2.axis(ax2.axis('tight'))
    ax.grid('on')
    ax2.grid('on', axis='x')
    ax.legend(loc='best', fontsize='small')
    ax.set_xticklabels([])
    ax.set_ylabel(r'$LS$ (norm)', labelpad=0)
    ax2.set_ylabel(r'$cc$')
    ax2.set_xlabel(r'$r_\mathrm{CSD}$ ($\upmu$m)', labelpad=0)
    ax.text(0.5, 1.02, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.2, 1.0, r'$\textbf{%s}$' % alphabet[i] ,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    
    i += 1


    ax = fig.add_subplot(nrows, 4, i+1)
    std_csd = icsd.StandardCSD(lfp=datas_d[dset]['el_sd']*pq.mV,
                                    **std_input)
    csd = std_csd.filter_csd(std_csd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**2) #unit conversion
    csd /= (pl.diff(z_data)[0] * (1E3*pq.mm/pq.m))
    clims = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation='nearest', rasterized=False,
                   cmap=pl.get_cmap('bwr_r', 51), origin='bottom',
                   vmin=-clims, vmax=clims,
                   extent=extent)
    vlimround = plot_signal_sum(ax, datas_d[dset]['tvec'], csd.magnitude,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clims)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clims).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    pl.setp(ax, xticklabels=[])
    pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.02, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.15, 1.0, r'$\textbf{%s}$' % alphabet[i] ,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1



    ax = fig.add_subplot(nrows, 4, i+1)
    delta_icsd = icsd.DeltaiCSD(lfp=datas_d[dset]['el_sd']*pq.mV,
                                     diam = 200*1E-6*pq.m,
                                     **delta_input)
    csd = delta_icsd.filter_csd(delta_icsd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**2) #unit conversion
    csd /= (pl.diff(z_data)[0] * (1E3*pq.mm/pq.m))
    clims = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation='nearest', rasterized=False,
                   cmap=pl.get_cmap('bwr_r', 51), origin='bottom',
              vmin=-clims, vmax=clims,
              extent=extent)
    vlimround = plot_signal_sum(ax, datas_d[dset]['tvec'], csd.magnitude,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clims)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clims).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    pl.setp(ax, xticklabels=[])
    pl.setp(ax, yticklabels=[])
    ax.text(0.5, 1.02, mytitles[i]+r', $r_\mathrm{CSD}$=%.0f $\upmu$m' % (100),
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.1, 1.0, r'$\textbf{%s}$' % alphabet[i] ,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1



    ax = fig.add_subplot(nrows, 4, i+1)
    spline_icsd = icsd.SplineiCSD(lfp=datas_d[dset]['el_sd']*pq.mV,
                                       **spline_input)
    csd = spline_icsd.filter_csd(spline_icsd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**3) #unit conversion
    clims = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation='nearest', rasterized=False,
                   cmap=pl.get_cmap('bwr_r', 51), origin='bottom',
              vmin=-clims, vmax=clims,
              extent=extent)    
    vlimround = plot_signal_sum(ax, datas_d[dset]['tvec'], csd.magnitude[::5, :],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clims)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clims).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    pl.setp(ax, xticklabels=[])
    pl.setp(ax, yticklabels=[])
    ax.text(0.5, 1.02, mytitles[i]+r', $r_\mathrm{CSD}$=%.0f $\upmu$m' % (spline_input['diam']/2*1E6),
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.1, 1.0, r'$\textbf{%s}$' % alphabet[i] ,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1



    #loop over test diameters
    for j, diam in enumerate(diams):
        print diam

        delta_icsd = icsd.DeltaiCSD(lfp=datas_d[dset]['el_sd']*pq.mV,
                                         diam = diam,
                                         **delta_input)

        ax = fig.add_subplot(nrows, 4, i+1)
        csd = delta_icsd.filter_csd(delta_icsd.get_csd())
        csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**2) #unit conversion
        csd /= (pl.diff(z_data)[0] * (1E3*pq.mm/pq.m))
        clims = abs(csd.magnitude).max()
        im = ax.imshow(csd,
                       interpolation='nearest', rasterized=False,
                       cmap=pl.get_cmap('bwr_r', 51), origin='bottom',
                  vmin=-clims, vmax=clims,
                  extent=extent)
        vlimround = plot_signal_sum(ax, datas_d[dset]['tvec'], csd.magnitude,
                                    T=(tvec[0], tvec[-1]),
                                    scaling_factor=1., scalebar=True,
                                    vlimround=clims)
        ax.set_yticks(pl.arange(2, 17, 2)+0.5)
        ax.set_yticklabels(pl.arange(2, 17, 2))

        ax.axis('tight')
        text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clims).split('e')]), color='k')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
        pl.setp(ax, xlabel=r'$t$ (ms)')
        if j == 0:
            pl.ylabel('Position', ha='center', va='center')
        else:
            pl.setp(ax, yticklabels=[])


        if diam <= 2E-3:
            mytitle = r'$\delta$-iCSD, ' + r'$r_\mathrm{CSD}$=%g $\upmu$m' % (diam/2*1E6)
        else:
            mytitle = r'$\delta$-iCSD, ' + r'$r_\mathrm{{CSD}}={}\cdot 10^{{{:.0f}}}~\upmu$m'.format(*[float(flt) for flt in ('%.1e' % (diam/2*1E6)).split('e')])

        ax.text(0.5, 1.02, mytitle,
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax.transAxes)
        ax.text(-0.1, 1.0, r'$\textbf{%s}$' % alphabet[i] ,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='black',
            transform=ax.transAxes)
        i += 1


def figure_5(filtering=False, filtering_CSD=True, globalscaling=False, rasterized=False):
    f_d = ['simres_R050', 'simres_R100', 'simres_R200', 'simres_R300', 'simres_R400']
    radiis = pl.array([50, 100, 200, 300, 400])
    n_rows = 3
    n_cols = 5

    mytitles = [
        '$r_\mathrm{syn}=50 \upmu$m',
        '$r_\mathrm{syn}=100 \upmu$m',
        '$r_\mathrm{syn}=200 \upmu$m',
        '$r_\mathrm{syn}=300 \upmu$m',
        '$r_\mathrm{syn}=400 \upmu$m',
        'LFP',
        'LFP',
        'LFP',
        'LFP',
        'LFP',
        'CSD, $r_\mathrm{CSD}=50 \upmu$m',
        'CSD, $r_\mathrm{CSD}=100 \upmu$m',
        'CSD, $r_\mathrm{CSD}=200 \upmu$m',
        'CSD, $r_\mathrm{CSD}=300 \upmu$m',
        'CSD, $r_\mathrm{CSD}=400 \upmu$m',
    ]


    fig =  pl.figure(figsize=(12,9))
    for i, k in enumerate(f_d):
        ax = pl.subplot(n_rows, n_cols, i+1)
        ax.scatter(datas_d[k]['som_pos0_x'][:500], datas_d[k]['som_pos0_z'][:500],
                   marker='o', color=(0.75,0.75,0.75), label='no syn.', s=5,
               rasterized=rasterized)
        ax.scatter(datas_d[k]['som_pos_x'], datas_d[k]['som_pos_z'],
                   marker='o', color='k', label='w. syn.', s=5,
               rasterized=rasterized)
        ax.scatter(datas_d[k]['syn_pos_x'], datas_d[k]['syn_pos_z'],
                   marker='|', facecolor='r', edgecolor='gray', linewidth=1,
                   label='syn. pos.', s=1.5,
                   # marker='o', facecolor='r', edgecolor='gray', linewidth=0.25,
                   # label='syn. pos.', s=2.5,
               rasterized=rasterized)

        ax.scatter(datas_d[k]['som_pos0_x'][:500], datas_d[k]['som_pos0_y'][:500] - 800,
                   marker='o', color=(0.75,0.75,0.75), label='no syn.', s=5,
               rasterized=rasterized)
        ax.scatter(datas_d[k]['som_pos_x'], datas_d[k]['som_pos_y'] - 800,
                   marker='o', color='k', label='w. syn.', s=5,
               rasterized=rasterized)
        ax.scatter(datas_d[k]['syn_pos_x'], datas_d[k]['syn_pos_y'] - 800,
                   marker='|', facecolor='r', edgecolor='gray', linewidth=1,
                   label='syn. pos.', s=1.5,
                   # marker='o', facecolor='r', edgecolor='gray', linewidth=0.25,
                   # label='syn. pos.', s=2.5,
               rasterized=rasterized)

        ax.axis('equal')
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticks(pl.arange(-250, 300, 500))
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticks(pl.arange(-250, 300, 250))
        if i == 0:
            adjust_spines(ax,['left']) #'bottom'
            ax.spines['left'].set_bounds(-250, 250)
            ax.set_yticks([-250, 0, 250])
            ax.text(-0.25, 1.0, r'$\textbf{A}$',
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize=16, fontweight='black',
                transform=ax.transAxes)
            ax.text(-1100, 0, r'$z$ ($\upmu$m)', rotation='vertical',
                    horizontalalignment='center',
                verticalalignment='center')

        else:
            adjust_spines(ax,[])

        divider = make_axes_locatable(ax)
        axHistx = divider.append_axes("right", 0.2, pad=0)

        binwidth = 10
        bins = pl.arange(-1400, 400 + binwidth, binwidth)
        hist1 = axHistx.hist(datas_d[k]['syn_pos_z'], bins=bins,
                             orientation='horizontal', histtype='stepfilled',
                             color='r',)
        hist2 = axHistx.hist(datas_d[k]['syn_pos_y']-800, bins=bins,
                             orientation='horizontal', histtype='stepfilled',
                             color='r',)

        nmax = pl.array([hist1[0], hist2[0]]).max()
        axHistx.axis([0, nmax, -1400, 400])

        ax.set_ylim([-1400, 400])

        axHistx.yaxis.set_ticks([])
        adjust_spines(axHistx, ['bottom'])
        axHistx.xaxis.set_ticks([nmax])

        ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)


    #spatial filtering coefficients
    b = ss.gaussian(3,1)
    a = pl.array([b.sum()])

    if globalscaling:
        clims = 0
        for i, k in enumerate(f_d):
            if filtering:
                fdata = sn.filters.convolve1d(datas_d[k]['el_sd'], b/a.sum(),
                                  axis=0)
                if abs(fdata).max() > clims:
                    clims = abs(fdata).max()
            else:
                if abs(datas_d[k]['el_sd']).max() > clims:
                    clims = abs(datas_d[k]['el_sd']).max()


    for i, k in enumerate(f_d):
        ax = fig.add_subplot(n_rows, n_cols, n_cols+i+1)
        if filtering:
            fdata = sn.filters.convolve1d(datas_d[k]['el_sd'], b/a.sum(),
                              axis=0)

            if not globalscaling:
                clims = abs(fdata).max()

            im = ax.imshow(fdata,
                      interpolation='nearest', rasterized=False, origin='bottom',
                      vmin=-clims, vmax=clims, cmap=pl.get_cmap('PRGn', 51),
                      extent=extent)
            vlimround = plot_signal_sum(ax, datas_d[k]['tvec'], fdata,
                                        T=(tvec[0], tvec[-1]),
                                        scaling_factor=1., scalebar=True,
                                        vlimround=clims)
        else:
            if not globalscaling:
                clims = abs(datas_d[k]['el_sd']).max()
            im = ax.imshow(datas_d[k]['el_sd'],
                      interpolation='nearest', rasterized=False, origin='bottom',
                      vmin=-clims, vmax=clims, cmap=pl.get_cmap('PRGn', 51),
                      extent=extent)
            vlimround = plot_signal_sum(ax, datas_d[k]['tvec'], datas_d[k]['el_sd'],
                                        T=(tvec[0], tvec[-1]),
                                        scaling_factor=1., scalebar=True,
                                        vlimround=clims)
        ax.set_yticks(pl.arange(2, 17, 2)+0.5)
        ax.set_yticklabels(pl.arange(2, 17, 2))
        ax.axis('tight')
        if not globalscaling:
            text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clims).split('e')]), color='k')
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
        pl.setp(ax, xticklabels=[])
        ax.set_title(mytitles[n_cols+i])
        if i == 0:
            pl.ylabel('Position', ha='center', va='center')
            ax.text(-0.25, 1.0, r'$\textbf{B}$',
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize=16, fontweight='black',
                transform=ax.transAxes)
        if i > 0:
            pl.setp(ax, yticklabels=[])

    bbox = pl.array(ax.get_position())
    cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                             0.01, bbox[1][1]-bbox[0][1]-0.1])
    cbar = pl.colorbar(im, cax=cax, ticks=[-clims, 0, clims])
    if not globalscaling:
        cbar.ax.set_yticklabels([-1,0,1])
    else:
        cbar.set_ticklabels([r'$-{}\cdot 10^{{{:.0f}}}$'.format(*[float(flt) for flt in ('%.1e' % clims).split('e')]),
                             0,
                             r'${}\cdot 10^{{{:.0f}}}$'.format(*[float(flt) for flt in ('%.1e' % clims).split('e')]),])
        cbar.set_label(r'(mV)', labelpad=0)

    if globalscaling:
        clims = 0
        for i, k in enumerate(f_d):
            if filtering_CSD:
                fdata = sn.filters.convolve1d(datas_d[k]['CSD']*1E6, b/a.sum(),
                                  axis=0)
                if abs(fdata).max() > clims:
                    clims = abs(fdata).max()
            else:
                if abs(datas_d[k]['el_sd']).max() > clims:
                    clims = abs(datas_d[k]['el_sd']).max()

    for i, k in enumerate(f_d):
        ax = fig.add_subplot(n_rows, n_cols, 2*n_cols+i+1)
        if filtering_CSD:
            fdata = sn.filters.convolve1d(datas_d[k]['CSD']*1E6, b/a.sum(),
                              axis=0)
            if not globalscaling:
                clims = abs(fdata).max()
            im = ax.imshow(fdata,
                      interpolation='nearest', rasterized=False, origin='bottom',
                      vmin=-clims, vmax=clims, cmap=pl.get_cmap('bwr_r', 51),
                      extent=extent)
            vlimround = plot_signal_sum(ax, datas_d[k]['tvec'], fdata,
                                        T=(tvec[0], tvec[-1]),
                                        scaling_factor=1., scalebar=True,
                                        vlimround=clims)
        else:
            if not globalscaling:
                clims = abs(datas_d[k]['CSD']*1E6).max()
            im = ax.imshow(datas_d[k]['CSD']*1E6,
                      interpolation='nearest', rasterized=False, origin='bottom',
                      vmin=-clims, vmax=clims, cmap=pl.get_cmap('bwr_r', 51),
                      extent=extent)
            vlimround = plot_signal_sum(ax, datas_d[k]['tvec'],
                                        datas_d[k]['CSD']*1E6,
                                        T=(tvec[0], tvec[-1]),
                                        scaling_factor=1., scalebar=True,
                                        vlimround=clims)
        ax.set_yticks(pl.arange(2, 17, 2)+0.5)
        ax.set_yticklabels(pl.arange(2, 17, 2))
        ax.axis('tight')
        ax.set_title(r'CSD, $r_{\Omega_k}=%i \upmu$m' % radiis[i], color='k')
        if not globalscaling:
            text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clims).split('e')]), color='k')
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
        pl.xlabel(r'$t$ (ms)')
        if i == 0:
            pl.ylabel('Position', ha='center', va='center')
            ax.text(-0.25, 1.0, r'$\textbf{C}$',
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize=16, fontweight='black',
                transform=ax.transAxes)
        if i > 0:
            pl.setp(ax, yticklabels=[])


    bbox = pl.array(ax.get_position())
    cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                             0.01, bbox[1][1]-bbox[0][1]-0.1])
    cbar = pl.colorbar(im, cax=cax, ticks=[-clims, 0, clims])
    if not globalscaling:
        cbar.ax.set_yticklabels([-1,0,1])
    else:
        cbar.set_ticklabels([r'$-{}\cdot 10^{{{:.0f}}}$'.format(*[float(flt) for flt in ('%.1e' % clims).split('e')]),
                             0,
                             r'${}\cdot 10^{{{:.0f}}}$'.format(*[float(flt) for flt in ('%.1e' % clims).split('e')]),])
        cbar.set_label(r'($\upmu$Amm$^{{-3}}$)', labelpad=0)

def figure_4(exp_data='smoothed', model_data='lfp_filtered',
             globalclim=False, clim=1E-2):

    PS = ParameterSpace({
        'factor' : ParameterRange([1]),
        'spatial' : ParameterRange([0, -2]),
        'temporal' : ParameterRange([0])
    })


    ncols=4
    nrows=3
    fig = pl.figure(figsize=(12, 9))
    j = 1

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    mytitles = [
        'Exp. LFP, TC1',
        'Exp. LFP, TC2',
        'Exp. LFP, TC3',
        'Exp. LFP, TC4',
        'Model LFP, RS',
        'Model LFP, FS',
        'Model LFP, RS+FS',
        'Model LFP, RS+FS, shifted',
        'Model LFP, FS ($-50\%$)',
        'Model LFP, RS+FS ($-50\%$)',
        'Model LFP, RS+FS ($-50\%$), shifted',
    ]

    i = 0
    ax = fig.add_subplot(nrows, ncols, j)
    if not globalclim:
        clim = abs(data_02_1BN1[exp_data]).max()
    ax.imshow(data_02_1BN1[exp_data], interpolation='nearest', rasterized=False,
              cmap=pl.get_cmap('PRGn', 51), origin='bottom',
              vmin = -clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_02_1BN1['tvec'], data_02_1BN1[exp_data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))
    
    ax.annotate("", xy=(1, 9), xytext=(0, 6),
        arrowprops=dict(arrowstyle="->"))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.125, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1




    ax = fig.add_subplot(nrows, ncols, j)
    if not globalclim:
        clim = abs(data_02_1BN2[exp_data]).max()
    im = ax.imshow(data_02_1BN2[exp_data], interpolation='nearest',
                   rasterized=False, cmap=pl.get_cmap('PRGn', 51), origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_02_1BN2['tvec'], data_02_1BN2[exp_data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.125, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    ax = fig.add_subplot(nrows, ncols, j)
    if not globalclim:
        clim = abs(data_08_1E[exp_data]).max()
    im = ax.imshow(data_08_1E[exp_data], interpolation='nearest',
                   rasterized=False, cmap=pl.get_cmap('PRGn', 51), origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_08_1E['tvec'], data_08_1E[exp_data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.125, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    ax = fig.add_subplot(nrows, ncols, j)
    if not globalclim:
        clim = abs(data_08_2A1[exp_data]).max()
    im = ax.imshow(data_08_2A1[exp_data], interpolation='nearest',
                   rasterized=False, cmap=pl.get_cmap('PRGn', 51),
                   origin='bottom', vmin=-clim, vmax=clim, extent=extent)
    vlimround = plot_signal_sum(ax, data_08_2A1['tvec'], data_08_2A1[exp_data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.125, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1

    bbox = pl.array(ax.get_position())
    cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                             0.01, bbox[1][1]-bbox[0][1]-0.1])
    cbar = pl.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
    cbar.ax.set_yticklabels([-1,0,1])


    ax = fig.add_subplot(nrows, ncols, j)
    if not globalclim:
        clim = abs(datas_RS[model_data]).max()
    ax.imshow(datas_RS[model_data], interpolation='nearest', rasterized=False,
              cmap=pl.get_cmap('PRGn', 51), origin='bottom',
              vmin = -clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, datas_RS['tvec'], datas_RS[model_data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.125, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1



    ax = fig.add_subplot(nrows, ncols, j)
    if not globalclim:
        clim = abs(datas_FS[model_data]).max()
    ax.imshow(datas_FS[model_data], interpolation='nearest', rasterized=False,
              cmap=pl.get_cmap('PRGn', 51), origin='bottom',
              vmin = -clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, datas_FS['tvec'], datas_FS[model_data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.125, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    # compute and print some correlation coefficients between experimental
    # and model LFP datas
    def corrcoef_exp_model(exp, exp_t, mod, mod_t):
        '''
        compute correlation coefficients of experiment and model data
        '''
        return np.corrcoef(exp.flatten(), mod[:, :exp_t.size].flatten())[1,0]
    
    print 'CC(TC1, RS): {:.3f}'.format(corrcoef_exp_model(data_02_1BN1[exp_data], data_02_1BN1['tvec'], datas_RS[model_data], datas_RS['tvec']))
    print 'CC(TC2, FS): {:.3f}'.format(corrcoef_exp_model(data_02_1BN1[exp_data], data_02_1BN1['tvec'], datas_FS[model_data], datas_RS['tvec']))
    print 'CC(TC1, RS): {:.3f}'.format(corrcoef_exp_model(data_02_1BN2[exp_data], data_02_1BN2['tvec'], datas_RS[model_data], datas_RS['tvec']))
    print 'CC(TC2, FS): {:.3f}'.format(corrcoef_exp_model(data_02_1BN2[exp_data], data_02_1BN2['tvec'], datas_FS[model_data], datas_RS['tvec']))


    

    k = 0
    LFP = {}
    for P in PS.iter_inner():
        LFP_shift = translate_data(P['spatial'], P['temporal'], datas_FS[model_data])
        LFP[k] = datas_RS[model_data] + P['factor'] * LFP_shift
        if not globalclim:    
            clim = abs(LFP[k]).max()
        ax = fig.add_subplot(nrows, ncols, j)
        ax.imshow(LFP[k], interpolation='nearest', rasterized=False,
                  cmap=pl.get_cmap('PRGn', 51), origin='bottom',
                  vmin = -clim, vmax=clim,
                  extent=extent)
        vlimround = plot_signal_sum(ax, datas_RS['tvec'], LFP[k],
                                    T=(tvec[0], tvec[-1]),
                                    scaling_factor=1., scalebar=True,
                                    vlimround=clim)
        ax.set_yticks(pl.arange(2, 17, 2)+0.5)
        
        ax.axis('tight')
        
        if pl.mod(j-1, ncols) != 0:
            pl.setp(ax, yticklabels=[])
            pl.setp(ax, xticklabels=[])
        else:
            pl.ylabel('Position', ha='center', va='center')
        ax.text(0.5, 1.025,
                mytitles[i],
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes)
        text = ax.text(0.5, 0.975,
                r'$z_\mathrm{shift}=$%i$\upmu$m' % int(P['spatial']*100),
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes,
                )
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
        ax.text(-0.125, 1, r'$\textbf{%s}$' % alphabet[i],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='black',
            transform=ax.transAxes)
        i += 1

        text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])

        if k == 0:
            print 'CC(TC1, RS + FS): {:.3f}'.format(corrcoef_exp_model(data_02_1BN1[exp_data], data_02_1BN1['tvec'], LFP[k], datas_RS['tvec']))
            print 'CC(TC2, RS + FS): {:.3f}'.format(corrcoef_exp_model(data_02_1BN2[exp_data], data_02_1BN2['tvec'], LFP[k], datas_RS['tvec']))
        elif k == 1:
            print 'CC(TC1, RS + FS (shifted)): {:.3f}'.format(corrcoef_exp_model(data_02_1BN1[exp_data], data_02_1BN1['tvec'], LFP[k], datas_RS['tvec']))
            print 'CC(TC2, RS + FS (shifted)): {:.3f}'.format(corrcoef_exp_model(data_02_1BN2[exp_data], data_02_1BN2['tvec'], LFP[k], datas_RS['tvec']))
            
        j += 1
        k += 1



    j += 1

    ax = fig.add_subplot(nrows, ncols, j)
    if not globalclim:
        clim = abs(datas_FS[model_data]*0.5).max()
    ax.imshow(datas_FS[model_data]*0.5, interpolation='nearest', rasterized=False,
              cmap=pl.get_cmap('PRGn', 51), origin='bottom',
              vmin = -clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, datas_FS['tvec'], datas_FS[model_data]*0.5,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    pl.xlabel(r'$t$ (ms)')
    pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.125, 1, r'$\textbf{%s}$' % alphabet[i],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    k = 0
    LFP = {}
    for P in PS.iter_inner():
        LFP_shift = translate_data(P['spatial'], P['temporal'],
                                   datas_FS[model_data]*0.5)
        LFP[k] = datas_RS[model_data] + P['factor'] * LFP_shift
        if not globalclim:
            clim = abs(LFP[k]).max()
        ax = fig.add_subplot(nrows, ncols, j)
        ax.imshow(LFP[k], interpolation='nearest', rasterized=False,
                  cmap=pl.get_cmap('PRGn', 51), origin='bottom',
                  vmin = -clim, vmax=clim,
                  extent=extent)
        vlimround = plot_signal_sum(ax, datas_RS['tvec'], LFP[k],
                                    T=(tvec[0], tvec[-1]),
                                    scaling_factor=1., scalebar=True,
                                    vlimround=clim)
        ax.set_yticks(pl.arange(2, 17, 2)+0.5)
        ax.set_yticklabels(pl.arange(2, 17, 2))

        ax.axis('tight')
        pl.xlabel(r'$t$ (ms)')

        if pl.mod(j-1, ncols) != 0:
            pl.setp(ax, yticklabels=[])
        else:
            pl.ylabel('Position', ha='center', va='center')
        ax.text(0.5, 1.025,
                mytitles[i],
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes)
        text = ax.text(0.5, 0.975,
                r'$z_\mathrm{shift}=$%i$\upmu$m' % int(P['spatial']*100),
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes,
                )
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
        ax.text(-0.125, 1, r'$\textbf{%s}$' % alphabet[i],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16, fontweight='black',
            transform=ax.transAxes)
        i += 1

        text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])

        if k == 0:
            print 'CC(TC1, RS + 50pc FS): {:.3f}'.format(corrcoef_exp_model(data_02_1BN1[exp_data], data_02_1BN1['tvec'], LFP[k], datas_RS['tvec']))
            print 'CC(TC2, RS + 50pc FS): {:.3f}'.format(corrcoef_exp_model(data_02_1BN2[exp_data], data_02_1BN2['tvec'], LFP[k], datas_RS['tvec']))
        elif k == 1:
            print 'CC(TC1, RS + 50pc FS (shifted)): {:.3f}'.format(corrcoef_exp_model(data_02_1BN1[exp_data], data_02_1BN1['tvec'], LFP[k], datas_RS['tvec']))
            print 'CC(TC2, RS + 50pc FS (shifted)): {:.3f}'.format(corrcoef_exp_model(data_02_1BN2[exp_data], data_02_1BN2['tvec'], LFP[k], datas_RS['tvec']))


        j += 1
        k += 1



def translate_data(spatial_shift, temp_shift, data):
    '''translate in 2D using delay filter'''

    #Spatial translation
    if spatial_shift > 0:
        b = pl.zeros(spatial_shift + 1)
        b[-1] = 1
        a = pl.array([1])
        data_shift = ss.lfilter(b, a, data, axis=0)
    elif spatial_shift == 0:
        b = pl.array([1])
        a = pl.array([1])
        data_shift = ss.lfilter(b, a, data, axis=0)
    elif spatial_shift < 0:
        b = pl.zeros(-spatial_shift + 1)
        b[-1] = 1
        a = pl.array([1])
        data_shift = ss.lfilter(b, a, pl.squeeze(list(reversed(data))), axis=0)
        data_shift = pl.squeeze(list(reversed(data_shift)))

    #temporal translation
    if temp_shift > 0:
        b = pl.zeros(temp_shift + 1)
        b[-1] = 1
        a = pl.array([1])
        data_shift = ss.lfilter(b, a, data_shift, axis=-1)
    elif temp_shift == 0:
        b = pl.array([1])
        a = pl.array([1])
        data_shift = ss.lfilter(b, a, data_shift, axis=-1)
    elif temp_shift < 0:
        b = pl.zeros(-temp_shift + 1)
        b[-1] = 1
        a = pl.array([1])
        data_shiftT = ss.lfilter(b, a, pl.squeeze(list(reversed(data_shift.T))),
                                 axis=0)
        data_shift = pl.squeeze(list(reversed(data_shiftT))).T

    return data_shift


def figure_6(dset=pl.array(['simres_Spherical', 'simres_Cylindrical',
                            'simres_Gaussian']), filtering=False):
    n_rows = 4
    n_cols = 6

    titles = ['spherical', 'cylindrical', 'Gaussian']

    radiis = pl.array([0, 100, 200, 300, 400, 500])

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    fig = pl.figure(figsize=(12,n_rows*3))
    fig.subplots_adjust(wspace=0.25, hspace=0.25)

    i = 0
    for j in dset:
        ax = pl.subplot(n_rows, dset.size, i+1)
        ax.scatter(datas_d[j]['som_pos0_x'][:500], datas_d[j]['som_pos0_z'][:500],
                   marker='o', color=(0.75,0.75,0.75), label='no syn.', s=5,
               rasterized=False)
        ax.scatter(datas_d[j]['som_pos_x'], datas_d[j]['som_pos_z'],
                   marker='o', color='k', label='w. syn.', s=5,
               rasterized=False)
        ax.scatter(datas_d[j]['syn_pos_x'], datas_d[j]['syn_pos_z'],
                   marker='|', facecolor='r', edgecolor='gray', linewidth=1,
                   label='syn. pos.', s=1.5,
                   # marker='o', facecolor='r', edgecolor='gray', linewidth=0.25,
                   # label='syn. pos.', s=2.5,
               rasterized=False)

        ax.scatter(datas_d[j]['som_pos0_x'][:500], datas_d[j]['som_pos0_y'][:500]-800,
                   marker='o', color=(0.75,0.75,0.75), label='no syn.', s=5,
               rasterized=False)
        ax.scatter(datas_d[j]['som_pos_x'], datas_d[j]['som_pos_y']-800,
                   marker='o', color='k', label='w. syn.', s=5,
               rasterized=False)
        ax.scatter(datas_d[j]['syn_pos_x'], datas_d[j]['syn_pos_y']-800,
                   marker='|', facecolor='r', edgecolor='gray', linewidth=1,
                   label='syn. pos.', s=1.5,
                   # marker='o', facecolor='r', edgecolor='gray', linewidth=0.25,
                   # label='syn. pos.', s=2.5,
               rasterized=False)

        ax.axis('equal')
        ax.axis((pl.array(ax.axis())+pl.array([100, 100, 0, 0])))
        ax.xaxis.set_ticks([])
        ax.text(-0.30, 1.0, r'$\textbf{%s}$' % alphabet[i] ,
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=16, fontweight='black',
                transform=ax.transAxes)
        if i == 0:
            adjust_spines(ax,['left'])
            ax.spines['left'].set_bounds(-250, 250)
            ax.set_yticks([-250,0,250])
            ax.text(-1000, 0, r'$z$ ($\upmu$m)', rotation='vertical',
                    va='center', ha='center')
        else:
            adjust_spines(ax,[])
        pl.title(titles[i])

        divider = make_axes_locatable(ax)
        axHistx = divider.append_axes("right", 1, pad=0)

        binwidth = 10
        bins = pl.arange(-1400, 400 + binwidth, binwidth)
        hist1 = axHistx.hist(datas_d[j]['syn_pos_z'], bins=bins,
                             orientation='horizontal', histtype='stepfilled',
                             color='r',)
        hist2 = axHistx.hist(datas_d[j]['syn_pos_y']-800, bins=bins,
                             orientation='horizontal', histtype='stepfilled',
                             color='r',)

        nmax = pl.array([hist1[0], hist2[0]]).max()
        axHistx.axis([0, nmax, -1400, 400])

        axHistx.yaxis.set_ticks([])
        adjust_spines(axHistx, ['bottom'])
        axHistx.xaxis.set_ticks([nmax])

        ax.set_ylim([-1400, 400])

        i += 1

    i = 6


    #spatial filtering coefficients
    b = ss.gaussian(3,1)
    a = pl.array([b.sum()])



    for k in dset:
        for j in xrange(n_cols):
            if j == 0:
                ax = pl.subplot(n_rows, n_cols, i+1)
                if filtering:
                    fdata = sn.filters.convolve1d(datas_d[k]['el_sd'], b/a.sum(),
                                                  axis=0)
                    clim = abs(fdata).max()
                    if clim == 0:
                        clim=1E-10
                    im = ax.imshow(fdata,
                              interpolation='nearest', rasterized=False,
                              origin='bottom',
                              vmin=-clim, vmax=clim, cmap=pl.get_cmap('PRGn', 51),
                              extent=extent)
                    vlimround = plot_signal_sum(ax, datas_d[k]['tvec'], fdata,
                                                T=(tvec[0], tvec[-1]),
                                                scaling_factor=1., scalebar=True,
                                                vlimround=clim)
                else:
                    clim = abs(datas_d[k]['el_sd']).max()
                    if clim == 0:
                        clim=1E-10
                    im = ax.imshow(datas_d[k]['el_sd'],
                              interpolation='nearest', rasterized=False,
                              origin='bottom',
                              vmin=-clim, vmax=clim, cmap=pl.get_cmap('PRGn', 51),
                              extent=extent)
                    vlimround = plot_signal_sum(ax, datas_d[k]['tvec'],
                                                datas_d[k]['el_sd'],
                                                T=(tvec[0], tvec[-1]),
                                                scaling_factor=1., scalebar=True,
                                                vlimround=clim)
                    ax.contour(tvec, np.arange(1,17)[::-1], datas_d[k]['el_sd'],
                               [-2E-5, 2E-5],
                               colors=['gray', 'gray'], linewidths=2,
                               linestyles=['dashed', 'solid'])
                ax.set_yticks(pl.arange(2, 17, 2)+0.5)
                ax.set_yticklabels(pl.arange(2, 17, 2))
                ax.axis('tight')
                text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
                text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
                if i == 6:
                    pl.ylabel('Position', ha='center', va='center')
                    ax.text(-0.25, 1.0, r'$\textbf{D}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)
                if i == 12:
                    pl.ylabel('Position', ha='center', va='center')
                    ax.text(-0.25, 1.0, r'$\textbf{E}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)
                    pl.setp(ax, xticklabels=[])
                if i == 18:
                    pl.ylabel('Position', ha='center', va='center')
                    ax.text(-0.25, 1.0, r'$\textbf{F}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)
                if i >= 18:
                    pl.xlabel(r'$t$ (ms)')
                if i < 7:
                    pl.setp(ax, xticklabels=[])
                i += 1
            else:
                ax = pl.subplot(n_rows, n_cols, i+1)
                if filtering:
                    fdata = sn.filters.convolve1d(datas_d[k]['el_%i' % radiis[j]], b/a.sum(), axis=0)
                    clim = abs(fdata).max()
                    if clim == 0:
                        clim=1E-10
                    im = ax.imshow(fdata,
                              interpolation='nearest', rasterized=False,
                              origin='bottom',
                              vmin=-clim, vmax=clim, cmap=pl.get_cmap('PRGn', 51),
                              extent=extent)
                    vlimround = plot_signal_sum(ax, datas_d[k]['tvec'], fdata,
                                                T=(tvec[0], tvec[-1]),
                                                scaling_factor=1., scalebar=True,
                                                vlimround=clim)
                else:
                    clim = abs(datas_d[k]['el_%i' % radiis[j]]).max()
                    if clim == 0:
                        clim=1E-10
                    im = ax.imshow(datas_d[k]['el_%i' % radiis[j]],
                              interpolation='nearest', rasterized=False,
                              origin='bottom',
                              vmin=-clim, vmax=clim, cmap=pl.get_cmap('PRGn', 51),
                              extent=extent)
                    vlimround = plot_signal_sum(ax, datas_d[k]['tvec'],
                                                datas_d[k]['el_%i' % radiis[j]],
                                                T=(tvec[0], tvec[-1]),
                                                scaling_factor=1., scalebar=True,
                                                vlimround=clim)
                    ax.contour(tvec, np.arange(1,17)[::-1], datas_d[k]['el_%i' % radiis[j]],
                               [-2E-5, 2E-5],
                               colors=['gray', 'gray'], linewidths=2,
                               linestyles=['dashed', 'solid'])
                ax.set_yticks(pl.arange(2, 17, 2)+0.5)
                ax.set_yticklabels(pl.arange(2, 17, 2))
                ax.axis('tight')
                # if i == 22 or i == 23:
                #     text = pl.text(-0.5, 2.5, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
                # else:
                text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
                text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
                pl.setp(ax, yticklabels=[])
                if i < 18:
                    pl.setp(ax, xticklabels=[])
                if i >= 18:
                    pl.xlabel(r'$t$ (ms)')

                i += 1
            if i <= 12:
                if j > 0:
                    pl.title(r'$x=%i~\mu$m' % radiis[j])
                else:
                    pl.title(r'$x=%i$' % radiis[j])

        if k == 'simres_Spherical':
            bbox = pl.array(ax.get_position())
            cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                                     0.01, bbox[1][1]-bbox[0][1]-0.1])
            cbar = pl.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
            cbar.ax.set_yticklabels([-1,0,1])


def figure_7(dset=pl.array(['simres_Spherical', 'simres_Cylindrical',
                            'simres_Gaussian']),
             filtering=False, sigma=0.3, tstep=1.5):
    fig = pl.figure(figsize=(12, 6))
    fig.subplots_adjust(wspace=0.4, hspace=0.20)

    # find column index corresponding to tstep
    [col] = pl.where(datas_d['simres_Spherical']['tvec'] == tstep)[0]

    colors = ['c', 'm', 'y']
    linestyles = [(0, (2,1)), '-', (0, (3,1))]

    # counter
    i = 0

    ax = pl.subplot(2, 3, i+1)
    ax.set_prop_cycle(cycler('color', colors)+cycler('linestyle', linestyles))
    
    #spatial filtering coefficients
    b = ss.gaussian(3,1)
    a = pl.array([b.sum()])

    for j in dset:
        if filtering:
            fdata = sn.filters.convolve1d(datas_d[j]['el_r'], b/a.sum(), axis=0)
            ax.plot(datas_d[j]['el_r_x'], fdata[:, col], lw=2)
        else:
            ax.plot(datas_d[j]['el_r_x'], datas_d[j]['el_r'][:, col], lw=2)
    ax.axis(ax.axis('tight'))
    ax.set_xlim(0, 500)
    ax.plot([200, 200],[pl.axis()[2], pl.axis()[3]], ':', color='k')
    ax.plot([pl.axis()[0], pl.axis()[1]],[0, 0], '-.', color='k')
    pl.legend(['spherical', 'cylindrical', 'Gaussian', '$x=200\upmu$m', '$\phi=0$'],
        fontsize='small')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # pl.xlabel(r'$x$ ($\upmu$m)')
    pl.ylabel(r'$\phi(x)$ (mV)', labelpad=0)
    ax.text(-0.25, 1.0, r'$\textbf{A}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)


    i += 1

    #spatial filtering coefficients
    b = ss.gaussian(3,1)
    a = pl.array([b.sum()])

    ax = pl.subplot(2, 3, i+1)
    ax.set_prop_cycle(cycler('color', colors)+cycler('linestyle', linestyles))
    for j in dset:
        if filtering:
            fdata = sn.filters.convolve1d(datas_d[j]['el_r'], b/a.sum(), axis=0)
            ax.plot(datas_d[j]['el_r_x'], abs(fdata[:, col]), lw=2)
        else:
            ax.plot(datas_d[j]['el_r_x'], abs(datas_d[j]['el_r'][:, col]), lw=2)
    ax.axis(ax.axis('tight'))
    ax.plot([200, 200],[0, pl.axis()[3]], ':', color='k')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none')
    ax.set_xlim(0, 500)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # pl.xlabel(r'$x$ ($\upmu$m)')
    pl.ylabel(r'$| \phi(x) |$ (mV)', labelpad=0)
    ax.text(-0.25, 1.0, r'$\textbf{B}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)


    i += 1

    ax = pl.subplot(2, 3, i+1)
    ax.set_prop_cycle(cycler('color', colors)+cycler('linestyle', linestyles))
    for j in dset:
        if filtering:
            fdata = sn.filters.convolve1d(datas_d[j]['el_r'], b/a.sum(), axis=0)
            ax.loglog(datas_d[j]['el_r_x'], abs(fdata[:, col]), lw=2)
        else:
            ax.loglog(datas_d[j]['el_r_x'], abs(datas_d[j]['el_r'][:, col]),
                      lw=2, label='_nolegend_')

    ax.loglog([200, 200],[pl.axis()[2], pl.axis()[3]], ':', color='k',
        label='$x=200\upmu$m')

    ax.loglog([1E3, 4E2], [1E-4, 6.25E-4], '-', lw=2, color=[0,0,0],
        label='$\propto r^{-2}$')
    ax.loglog([1E3, 4E2], [1E-4, 15.625E-4], '-', lw=2, color=[0.33,0.33,0.33],
        label='$\propto r^{-3}$')
    ax.loglog([1E3, 4E2], [1E-4, 39.0625E-4], '-', lw=2, color=[0.66,0.66,0.66],
        label='$\propto r^{-4}$')

    ax.axis(ax.axis('tight'))
    ax.legend(loc='best', fontsize='small')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # pl.xlabel(r'$x$ ($\upmu$m)', verticalalignment='bottom')
    pl.ylabel(r'$| \phi(x) |$ (mV)', labelpad=0)

    ax.text(-0.25, 1.0, r'$\textbf{C}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)

    i += 1


    # plot LFP with lateral offset for different input sizes
    radiis = pl.array([50, 100, 200, 300, 400])
    f_d = ['simres_R{0:03d}'.format(r) for r in radiis]
    mytitles = ['$r_\mathrm{{syn}}={} \upmu$m'.format(r) for r in radiis]   
    colors = [
        (0.75,0.5,0.25),
        (0.75,0.25,0.5),
        (0.75,0,0.75),
        (0.5,0.25,1),
        (0.25,0.5,1)
        ]
        
    linestyles = [(0, (1,4)), (0, (1,2)), '-', (0, (2,1)), (0, (4,1))]


    ax = pl.subplot(2, 3, i+1)
    ax.set_prop_cycle(cycler('color', colors)+cycler('linestyle', linestyles))
    #spatial filtering coefficients
    b = ss.gaussian(3,1)
    a = pl.array([b.sum()])

    for f, title in zip(f_d, mytitles):
        if filtering:
            fdata = sn.filters.convolve1d(datas_d[f]['el_r'], b/a.sum(), axis=0)
            ax.plot(datas_d[f]['el_r_x'], fdata[:, col], lw=2, label=title)
        else:
            ax.plot(datas_d[f]['el_r_x'], datas_d[f]['el_r'][:, col], lw=2,
                    label=title)
    ax.legend(loc='best', fontsize='small')
    ax.axis(ax.axis('tight'))
    ax.set_xlim(0, 1000)
    ax.plot([ax.axis()[0], ax.axis()[1]],[0, 0], '-.', color='k')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    pl.xlabel(r'$x$ ($\upmu$m)')
    pl.ylabel(r'$\phi(x)$ (mV)', labelpad=0)
    ax.text(-0.25, 1.0, r'$\textbf{D}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)
    i += 1


    ax = pl.subplot(2, 3, i+1)
    ax.set_prop_cycle(cycler('color', colors)+cycler('linestyle', linestyles))
    #spatial filtering coefficients
    b = ss.gaussian(3,1)
    a = pl.array([b.sum()])

    for f, title in zip(f_d, mytitles):
        if filtering:
            fdata = sn.filters.convolve1d(datas_d[f]['el_r'], b/a.sum(), axis=0)
            ax.plot(datas_d[f]['el_r_x'], abs(fdata[:, col]), lw=2, label=title)
        else:
            ax.plot(datas_d[f]['el_r_x'], abs(datas_d[f]['el_r'][:, col]), lw=2,
                    label=title)
    ax.axis(ax.axis('tight'))
    ax.set_xlim(0, 1000)
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    pl.xlabel(r'$x$ ($\upmu$m)')
    pl.ylabel(r'$| \phi(x) |$ (mV)', labelpad=0)
    ax.text(-0.25, 1.0, r'$\textbf{E}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)
    i += 1
    
    ax = pl.subplot(2, 3, i+1)
    ax.set_prop_cycle(cycler('color', colors)+cycler('linestyle', linestyles))
    #spatial filtering coefficients
    b = ss.gaussian(3,1)
    a = pl.array([b.sum()])

    for f, title in zip(f_d, mytitles):
        if filtering:
            fdata = sn.filters.convolve1d(datas_d[f]['el_r'], b/a.sum(), axis=0)
            ax.loglog(datas_d[f]['el_r_x'], abs(fdata[:, col]), lw=2,
                      label='_nolegend_')
        else:
            ax.loglog(datas_d[f]['el_r_x'], abs(datas_d[f]['el_r'][:, col]),
                      lw=2, label='_nolegend_')

    ax.loglog([1E3, 4E2], [1E-4, 6.25E-4], '-', lw=2, color=[0,0,0],
        label='$\propto r^{-2}$')
    ax.loglog([1E3, 4E2], [1E-4, 15.625E-4], '-', lw=2, color=[0.33,0.33,0.33],
        label='$\propto r^{-3}$')
    ax.loglog([1E3, 4E2], [1E-4, 39.0625E-4], '-', lw=2, color=[0.66,0.66,0.66],
        label='$\propto r^{-4}$')


    ax.axis(ax.axis('tight'))
    # ax.set_xlim(0, 500)
    ax.legend(loc='best', fontsize='small')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    pl.xlabel(r'$x$ ($\upmu$m)')
    pl.ylabel(r'$| \phi(x) |$ (mV)', labelpad=0)
    ax.text(-0.25, 1.0, r'$\textbf{F}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)
    i += 1


def figure_8(dset=pl.array(['simres_Spherical', 'simres_Cylindrical',
                            'simres_Gaussian']),
             filtering=False, sigma=0.3, tstep=1.5):
    fig = pl.figure(figsize=(12, 3.5))
    fig.subplots_adjust(wspace=0.4, hspace=0.30, bottom=0.15)

    # find column index corresponding to tstep
    [col] = pl.where(datas_d['simres_Spherical']['tvec'] == tstep)[0]

    colors = ['c', 'm', 'y']
    linestyles = [(0, (2,1)), '-', (0, (3,1))]

    # counter
    i = 0


    ax = pl.subplot(1, 3, i+1)
    ax.set_prop_cycle(cycler('color', colors)+cycler('linestyle', linestyles))
    for j in dset:
        radiuser = datas_d[j]['CSDr_r']
        csder = datas_d[j]['CSDr'][:, col]
        volumet = 4 * pl.pi / 3 * (radiuser[1:]**3 - radiuser[:-1]**3)
        strommen = pl.r_[0, csder*volumet]
        ax.plot(radiuser, strommen, label=j, lw=2)


    v = pl.axis('tight')
    ax.plot([200, 200],[pl.axis()[2], pl.axis()[3]], ':', color='k')
    ax.plot([pl.axis()[0], pl.axis()[1]],[0, 0], '-.', color='k')
    pl.axis(v)
    ax.set_xlim(0, 500)
    pl.legend(['spherical', 'cylindrical', 'Gaussian', '$r=200\upmu$m', '$\phi=0$'],
        fontsize='small')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.ticklabel_format(style='sci')
    pl.xlabel(r'$r$ ($\upmu$m)')
    pl.ylabel(r'$\Delta i_\mathrm{net}(r)$ (nA)', labelpad=0)
    ax.text(-0.25, 1.0, r'$\textbf{A}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)

    i += 1



    ax = pl.subplot(1, 3, i+1)
    ax.set_prop_cycle(cycler('color', colors)+cycler('linestyle', linestyles))
    for j in dset:
        radiuser = datas_d[j]['CSDr_r']
        csder = datas_d[j]['CSDr'][:, col]
        volumet = 4 * pl.pi / 3 * (radiuser[1:]**3 - radiuser[:-1]**3)
        strommen = csder * volumet

        # plot potensial som funksjon av CSD regnet ut via the Poisson
        # equation in spherical coordinates. Vi regner ut dette via
        # scipy.integrate.odeint.

        # redefine some variable names
        c = pl.r_[0, csder] # CSD @ r=0 is 0
        r = radiuser
        hr = 1.
        dr = pl.diff(r)[0]
        R = r[-1]
        I = strommen

        def dE_dr(E, r0):
            '''
            Compute the derivative of E.

            It follows from Poisson's equation in spherical coordinates that
            dE/dr = c(r)/sigma - 2E/r

            E : electric field
            r0 : compute the derivative at this radius
            '''
            # need to choose a value in c
            if r0 <= r[-1]:
                ci = c[(r >= r0)][0]
            else:
                ci = 0

            # dE_dr = 0 for r0 = 0:
            if r0 == 0:
                dE_dr = 0.
            else:
                dE_dr = pl.array(ci / sigma - 2*E / r0)
            return dE_dr

        # initial condition E = 0 for r=0:
        E0 = 0

        # compute the radial component of the electric field
        E = si.odeint(dE_dr, E0, r, hmax=hr).flatten()

        # compute the potential phi as function or r
        phi = -E.cumsum()*dr

        # assuming that all c(r>=R) = 0, we can compute directly the potential at R, and
        # subtract from our estimate of the potential. Computed directly from
        # the sum of currents inside of R.
        phi_R = I.sum() / (4*pl.pi*sigma*R)
        phi -= phi[-1]
        phi += phi_R

        # plot potensial som funksjon av stroem i sfaeriske skall med
        # konstant stroemtetthet
        ax.plot(r, phi, label=j, lw=2)

    ax.axis(ax.axis('tight'))
    ax.plot([200, 200],[pl.axis()[2], pl.axis()[3]], ':', color='k')
    ax.plot([pl.axis()[0], pl.axis()[1]],[0, 0], '-.', color='k')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.ticklabel_format(style='sci')
    pl.ylabel(r'$\phi(r)$  (mV)', labelpad=0)
    pl.xlabel(r'$r$ ($\upmu$m)')
    ax.axis(ax.axis('tight'))
    ax.text(-0.25, 1.0, r'$\textbf{B}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)


    i += 1


    ax = pl.subplot(1, 3, i+1)
    ax.set_prop_cycle(cycler('color', colors)+cycler('linestyle', linestyles))
    axJ = ax.twinx() # axes for radial current density J


    for j in dset:
        radiuser = datas_d[j]['CSDr_r']
        csder = datas_d[j]['CSDr'][:, col]
        volumet = 4 * pl.pi / 3 * (radiuser[1:]**3 - radiuser[:-1]**3)
        strommen = csder * volumet

        # plot potensial som funksjon av CSD regnet ut via the Poisson
        # equation in spherical coordinates. Vi regner ut dette via
        # scipy.integrate.odeint.

        # redefine some variable names
        c = pl.r_[0, csder] # CSD @ r=0 is 0
        r = radiuser
        hr = 1.
        dr = pl.diff(r)[0]
        R = r[-1]
        I = strommen

        def dE_dr(E, r0):
            '''
            Compute the derivative of E.

            It follows from Poisson's equation in spherical coordinates that
            dE/dr = c(r)/sigma - 2E/r

            E : electric field
            r0 : compute the derivative at this radius
            '''
            # need to choose a value in c
            if r0 <= r[-1]:
                ci = c[(r >= r0)][0]
            else:
                ci = 0

            # dE_dr = 0 for r0 = 0:
            if r0 == 0:
                dE_dr = 0.
            else:
                dE_dr = pl.array(ci / sigma - 2*E / r0)
            return dE_dr

        # initial condition E = 0 for r=0:
        E0 = 0

        # compute the radial component of the electric field
        E = si.odeint(dE_dr, E0, r, hmax=hr).flatten()

        # plot radial component of electric field
        ax.plot(r, E, label=j, lw=2)

        # plot radial current density J = sigma*E
        axJ.plot(r, 1E3*sigma*E, label=j, lw=0) # unit muA/mm2

    # beautify J axes
    axJ.axis(axJ.axis('tight'))
    for loc, spine in axJ.spines.iteritems():
        if loc in ['left','top', 'bottom']:
            spine.set_color('none')
    axJ.yaxis.set_ticks_position('right')
    axJ.ticklabel_format(style='sci')
    axJ.set_ylabel(r'$J(r)$ ($\upmu$Amm$^{-2}$)', labelpad=2)
    axJ.axis(axJ.axis('tight'))

    # beautify E axes
    ax.axis(ax.axis('tight'))
    ax.plot([200, 200],[ax.axis()[2], ax.axis()[3]], ':', color='k')
    ax.plot([ax.axis()[0], ax.axis()[1]],[0, 0], '-.', color='k')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.ticklabel_format(style='sci')
    ax.set_ylabel(r'$E(r)$ (mV$\upmu$m$^{-1}$)', labelpad=0)
    ax.set_xlabel(r'$r$ ($\upmu$m)')
    ax.axis(ax.axis('tight'))
    ax.text(-0.25, 1.0, r'$\textbf{C}$',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, fontweight='black',
                        transform=ax.transAxes)



def figure_10(plotStepiCSD=False, sigma=0.3,
            interpolation='nearest',
            f_order=[(3, 1), (19, 5)],
            csd_cmap=pl.get_cmap('bwr_r', 51),
            data='smoothed'):


    if plotStepiCSD:
        ncols = 5
        alphabet = 'AAAAAABBBBBCCCCCDDDDDEEEEEE'
        mytitles = [
            'Exp. LFP, TC1',
            'tCSD',
            '$\delta$-iCSD',
            'step-iCSD',
            'spline-iCSD',
            'Exp. LFP, TC2',
            '',
            '',
            '',
            '',
            'Exp. LFP, TC3',
            '',
            '',
            '',
            '',
            'Exp. LFP, TC4',
            '',
            '',
            '',
            '',
                ]
    else:
        ncols = 4
        alphabet = 'AAAABBBBCCCCDDDDEEEEE'
        mytitles = [
            'Exp. LFP, TC1',
            'tCSD',
            '$\delta$-iCSD, $r_\mathrm{CSD}=100 \upmu$m',
            'spline-iCSD, $r_\mathrm{CSD}=100 \upmu$m',
            'Exp. LFP, TC2',
            '',
            '',
            '',
            'Exp. LFP, TC3',
            '',
            '',
            '',
            'Exp. LFP, TC4',
            '',
            '',
            '',
                ]
    nrows = 4
    fig = pl.figure(figsize=(12, 12))
    fig.subplots_adjust()
    j = 1



    # Input dictionaries for each method
    z_data = pl.linspace(100E-6, 1600E-6, 16) * pq.m
    delta_input = {
        'coord_electrode' : z_data,
        'diam' : 200E-6 * pq.m,          # source diameter
        'sigma' : sigma * pq.S / pq.m ,        # extracellular conductivity
        'sigma_top' : sigma * pq.S / pq.m ,    # conductivity on top of cortex
        'f_type' : 'gaussian',  # gaussian filter
        'f_order' : f_order[0],     # 3-point filter, sigma = 1.
    }
    step_input = {
        'coord_electrode' : z_data,
        'diam' : 200E-6 * pq.m,
        'h' : 100E-6 * pq.m,                # source thickness
        'sigma' : sigma * pq.S / pq.m ,
        'sigma_top' : sigma * pq.S / pq.m ,
        'tol' : 1E-12,          # Tolerance in numerical integration
        'f_type' : 'gaussian',
        'f_order' : f_order[0],
    }
    spline_input = {
        'coord_electrode' : z_data,
        'diam' : 200E-6 * pq.m,
        'sigma' : sigma * pq.S / pq.m ,
        'sigma_top' : sigma * pq.S / pq.m ,
        'num_steps' : 76,      # Spatial CSD upsampling to N steps
        'tol' : 1E-12,
        'f_type' : 'gaussian',
        'f_order' : f_order[1],
    }
    std_input = {
        'coord_electrode' : z_data,
        'sigma' : sigma * pq.S / pq.m,
        'vaknin_el' : True,
        'f_type' : 'gaussian',
        'f_order' : f_order[0],
    }

    i = 0



    ax = fig.add_subplot(nrows, ncols, j)
    clim = abs(data_02_1BN1[data]).max()
    im = ax.imshow(data_02_1BN1[data], interpolation=interpolation,
                   rasterized=False, cmap=pl.get_cmap('PRGn', 51),
                   origin='bottom', vmin = -clim, vmax=clim, extent=extent)
    vlimround = plot_signal_sum(ax, data_02_1BN1['tvec'], data_02_1BN1[data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.1, 1, r'$\textbf{%s}$' % alphabet[i] ,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    
    j += 1

    bbox = pl.array(ax.get_position())
    cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                             0.01, bbox[1][1]-bbox[0][1]-0.1])
    cbar = pl.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
    cbar.ax.set_yticklabels([-1,0,1])


    #Calling the different subclasses, with respective inputs.
    std_csd = icsd.StandardCSD(lfp=data_02_1BN1['raw_data']*pq.mV,
                                    **std_input)
    delta_icsd = icsd.DeltaiCSD(lfp=data_02_1BN1['raw_data']*pq.mV,
                                     **delta_input)
    step_icsd = icsd.StepiCSD(lfp=data_02_1BN1['raw_data']*pq.mV,
                                   **step_input)
    spline_icsd = icsd.SplineiCSD(lfp=data_02_1BN1['raw_data']*pq.mV,
                                       **spline_input)


    ax = fig.add_subplot(nrows, ncols, j)
    csd = std_csd.filter_csd(std_csd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**2) #unit conversion
    csd /= (pl.diff(z_data)[0] * (1E3*pq.mm/pq.m))
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
                    vmin=-clim, vmax=clim,
                    extent=extent)
    vlimround = plot_signal_sum(ax, data_02_1BN1['tvec'], csd.magnitude,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    ax = fig.add_subplot(nrows, ncols, j)
    csd = delta_icsd.filter_csd(delta_icsd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**2) #unit conversion
    csd /= (pl.diff(z_data)[0] * (1E3*pq.mm/pq.m))
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_02_1BN1['tvec'], csd.magnitude,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1
    
    if plotStepiCSD:
        ax = fig.add_subplot(nrows, ncols, j)
        csd = step_icsd.filter_csd(step_icsd.get_csd())
        csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**3) #unit conversion
        clim = abs(csd.magnitude).max()
        im = ax.imshow(csd,
                       interpolation=interpolation, rasterized=False,
                       cmap=csd_cmap, origin='bottom',
                  vmin=-clim, vmax=clim,
                  extent=extent)
        vlimround = plot_signal_sum(ax, data_02_1BN1['tvec'], csd.magnitude,
                                    T=(tvec[0], tvec[-1]),
                                    scaling_factor=1., scalebar=True,
                                    vlimround=clim)
        ax.set_yticks(pl.arange(2, 17, 2)+0.5)
        ax.set_yticklabels(pl.arange(2, 17, 2))

        ax.axis('tight')
        if j / (ncols*(nrows-1) + 1) != 1:
            pl.setp(ax, xticklabels=[])
        else:
            pl.xlabel(r'$t$ (ms)')

        if pl.mod(j-1, ncols) != 0:
            pl.setp(ax, yticklabels=[])
        else:
            pl.ylabel('Position', ha='center', va='center')
        ax.text(0.5, 1.025, mytitles[i],
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax.transAxes)
        i += 1
        text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
        j += 1


    ax = fig.add_subplot(nrows, ncols, j)
    csd = spline_icsd.filter_csd(spline_icsd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**3) #unit conversion
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_02_1BN1['tvec'], csd.magnitude[::5, :],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1

    bbox = pl.array(ax.get_position())
    cax = pl.gcf().add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                             0.01, bbox[1][1]-bbox[0][1]-0.1])
    cbar = pl.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
    cbar.ax.set_yticklabels([-1,0,1])




    ax = fig.add_subplot(nrows, ncols, j)
    clim = abs(data_02_1BN2[data]).max()
    im = ax.imshow(data_02_1BN2[data], interpolation=interpolation,
                   rasterized=False, cmap=pl.get_cmap('PRGn', 51),
                   origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_02_1BN2['tvec'], data_02_1BN2[data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.1, 1, r'$\textbf{%s}$' % alphabet[i] ,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    #Calling the different subclasses, with respective inputs.
    std_csd = icsd.StandardCSD(lfp=data_02_1BN2['raw_data']*pq.mV,
                                    **std_input)
    delta_icsd = icsd.DeltaiCSD(lfp=data_02_1BN2['raw_data']*pq.mV,
                                     **delta_input)
    step_icsd = icsd.StepiCSD(lfp=data_02_1BN2['raw_data']*pq.mV,
                                   **step_input)
    spline_icsd = icsd.SplineiCSD(lfp=data_02_1BN2['raw_data']*pq.mV,
                                       **spline_input)


    ax = fig.add_subplot(nrows, ncols, j)
    csd = std_csd.filter_csd(std_csd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**2) #unit conversion
    csd /= (pl.diff(z_data)[0] * (1E3*pq.mm/pq.m))
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_02_1BN2['tvec'], csd.magnitude,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    ax = fig.add_subplot(nrows, ncols, j)
    csd = delta_icsd.filter_csd(delta_icsd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**2) #unit conversion
    csd /= (pl.diff(z_data)[0] * (1E3*pq.mm/pq.m))
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_02_1BN2['tvec'], csd.magnitude,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    if plotStepiCSD:
        ax = fig.add_subplot(nrows, ncols, j)
        csd = step_icsd.filter_csd(step_icsd.get_csd())
        csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**3) #unit conversion
        clim = abs(csd.magnitude).max()
        im = ax.imshow(csd,
                       interpolation=interpolation, rasterized=False,
                       cmap=csd_cmap, origin='bottom',
                  vmin=-clim, vmax=clim,
                  extent=extent)
        vlimround = plot_signal_sum(ax, data_02_1BN2['tvec'], csd.magnitude,
                                    T=(tvec[0], tvec[-1]),
                                    scaling_factor=1., scalebar=True,
                                    vlimround=clim)
        ax.set_yticks(pl.arange(2, 17, 2)+0.5)
        ax.set_yticklabels(pl.arange(2, 17, 2))

        ax.axis('tight')
        if j / (ncols*(nrows-1) + 1) != 1:
            pl.setp(ax, xticklabels=[])
        else:
            pl.xlabel(r'$t$ (ms)')

        if pl.mod(j-1, ncols) != 0:
            pl.setp(ax, yticklabels=[])
        else:
            pl.ylabel('Position', ha='center', va='center')
        ax.text(0.5, 1.025, mytitles[i],
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax.transAxes)
        i += 1
        text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
        j += 1


    ax = fig.add_subplot(nrows, ncols, j)
    csd = spline_icsd.filter_csd(spline_icsd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**3) #unit conversion
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_02_1BN2['tvec'], csd.magnitude[::5, :],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1




    ax = fig.add_subplot(nrows, ncols, j)
    clim = abs(data_08_1E[data]).max()
    im = ax.imshow(data_08_1E[data], interpolation=interpolation,
                   rasterized=False, cmap=pl.get_cmap('PRGn', 51), origin='bottom',
                   vmin=-clim, vmax=clim, extent=extent)
    vlimround = plot_signal_sum(ax, data_08_1E['tvec'], data_08_1E[data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.1, 1, r'$\textbf{%s}$' % alphabet[i] ,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    #Calling the different subclasses, with respective inputs.
    std_csd = icsd.StandardCSD(lfp=data_08_1E['raw_data']*pq.mV, **std_input)
    delta_icsd = icsd.DeltaiCSD(lfp=data_08_1E['raw_data']*pq.mV, **delta_input)
    step_icsd = icsd.StepiCSD(lfp=data_08_1E['raw_data']*pq.mV, **step_input)
    spline_icsd = icsd.SplineiCSD(lfp=data_08_1E['raw_data']*pq.mV, **spline_input)


    ax = fig.add_subplot(nrows, ncols, j)
    csd = std_csd.filter_csd(std_csd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**2) #unit conversion
    csd /= (pl.diff(z_data)[0] * (1E3*pq.mm/pq.m))
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_08_1E['tvec'], csd.magnitude,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    ax = fig.add_subplot(nrows, ncols, j)
    csd = delta_icsd.filter_csd(delta_icsd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**2) #unit conversion
    csd /= (pl.diff(z_data)[0] * (1E3*pq.mm/pq.m))
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_08_1E['tvec'], csd.magnitude,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1

    if plotStepiCSD:
        ax = fig.add_subplot(nrows, ncols, j)
        csd = step_icsd.filter_csd(step_icsd.get_csd())
        csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**3) #unit conversion
        clim = abs(csd.magnitude).max()
        im = ax.imshow(csd,
                       interpolation=interpolation, rasterized=False,
                       cmap=csd_cmap, origin='bottom',
                  vmin=-clim, vmax=clim,
                  extent=extent)
        vlimround = plot_signal_sum(ax, data_08_1E['tvec'], csd.magnitude,
                                    T=(tvec[0], tvec[-1]),
                                    scaling_factor=1., scalebar=True,
                                    vlimround=clim)
        ax.set_yticks(pl.arange(2, 17, 2)+0.5)
        ax.set_yticklabels(pl.arange(2, 17, 2))

        ax.axis('tight')
        if j / (ncols*(nrows-1) + 1) != 1:
            pl.setp(ax, xticklabels=[])
        else:
            pl.xlabel(r'$t$ (ms)')

        if pl.mod(j-1, ncols) != 0:
            pl.setp(ax, yticklabels=[])
        else:
            pl.ylabel('Position', ha='center', va='center')
        ax.text(0.5, 1.025, mytitles[i],
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax.transAxes)
        i += 1
        text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
        j += 1


    ax = fig.add_subplot(nrows, ncols, j)
    csd = spline_icsd.filter_csd(spline_icsd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**3) #unit conversion
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_08_1E['tvec'], csd.magnitude[::5, :],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1



    ax = fig.add_subplot(nrows, ncols, j)
    clim = abs(data_08_2A1[data]).max()
    im = ax.imshow(data_08_2A1[data], interpolation=interpolation,
                   rasterized=False,
                   cmap=pl.get_cmap('PRGn', 51), origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_08_2A1['tvec'], data_08_2A1[data],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    else:
        pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    ax.text(-0.1, 1, r'$\textbf{%s}$' % alphabet[i] ,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16, fontweight='black',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    #Calling the different subclasses, with respective inputs.
    std_csd = icsd.StandardCSD(lfp=data_08_2A1['raw_data']*pq.mV, **std_input)
    delta_icsd = icsd.DeltaiCSD(lfp=data_08_2A1['raw_data']*pq.mV, **delta_input)
    step_icsd = icsd.StepiCSD(lfp=data_08_2A1['raw_data']*pq.mV, **step_input)
    spline_icsd = icsd.SplineiCSD(lfp=data_08_2A1['raw_data']*pq.mV, **spline_input)


    ax = fig.add_subplot(nrows, ncols, j)
    csd = std_csd.filter_csd(std_csd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**2) #unit conversion
    csd /= (pl.diff(z_data)[0] * (1E3*pq.mm/pq.m))
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_08_2A1['tvec'], csd.magnitude,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    pl.xlabel(r'$t$ (ms)')

    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1


    ax = fig.add_subplot(nrows, ncols, j)
    csd = delta_icsd.filter_csd(delta_icsd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**2) #unit conversion
    csd /= (pl.diff(z_data)[0] * (1E3*pq.mm/pq.m))
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_08_2A1['tvec'], csd.magnitude,
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])
    pl.xlabel(r'$t$ (ms)')

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1

    if plotStepiCSD:
        ax = fig.add_subplot(nrows, ncols, j)
        csd = step_icsd.filter_csd(step_icsd.get_csd())
        csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**3) #unit conversion
        clim = abs(csd.magnitude).max()
        im = ax.imshow(csd,
                       interpolation=interpolation, rasterized=False,
                       cmap=csd_cmap, origin='bottom',
                  vmin=-clim, vmax=clim,
                  extent=extent)
        vlimround = plot_signal_sum(ax, data_08_2A1['tvec'], csd.magnitude,
                                    T=(tvec[0], tvec[-1]),
                                    scaling_factor=1., scalebar=True, vlimround=clim)
        ax.set_yticks(pl.arange(2, 17, 2)+0.5)
        ax.set_yticklabels(pl.arange(2, 17, 2))

        ax.axis('tight')
        if j / (ncols*(nrows-1) + 1) != 1:
            pl.setp(ax, xticklabels=[])

        if pl.mod(j-1, ncols) != 0:
            pl.setp(ax, yticklabels=[])
        else:
            pl.ylabel('Position', ha='center', va='center')
        pl.xlabel(r'$t$ (ms)')
        ax.text(0.5, 1.025, mytitles[i],
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax.transAxes)
        i += 1
        text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
        j += 1


    ax = fig.add_subplot(nrows, ncols, j)
    csd = spline_icsd.filter_csd(spline_icsd.get_csd())
    csd *= ((1E6*pq.uA / pq.A) / (1E3*pq.mm/pq.m)**3) #unit conversion
    clim = abs(csd.magnitude).max()
    im = ax.imshow(csd,
                   interpolation=interpolation, rasterized=False,
                   cmap=csd_cmap, origin='bottom',
              vmin=-clim, vmax=clim,
              extent=extent)
    vlimround = plot_signal_sum(ax, data_08_2A1['tvec'], csd.magnitude[::5, :],
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
    ax.set_yticks(pl.arange(2, 17, 2)+0.5)
    ax.set_yticklabels(pl.arange(2, 17, 2))

    ax.axis('tight')
    if j / (ncols*(nrows-1) + 1) != 1:
        pl.setp(ax, xticklabels=[])

    if pl.mod(j-1, ncols) != 0:
        pl.setp(ax, yticklabels=[])
    else:
        pl.ylabel('Position', ha='center', va='center')
    pl.xlabel(r'$t$ (ms)')
    ax.text(0.5, 1.025, mytitles[i],
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
    i += 1
    text = pl.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~\upmu$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
    j += 1



def adjust_spines(ax,spines):
    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            spine.set_position(('outward',0)) # outward by 0 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])



def figure_data(f_RS='PS_simres_RS', f_FS='PS_simres_FS', index=0):

    id_RS = get_md5s(f_RS)[index]
    id_FS = get_md5s(f_FS)[index]
    
    dataset_RS = h5py.File(os.path.join('savedata', id_RS, 'simres.h5'), 'r')
    tvec = pl.array(dataset_RS['tvec'])
    dataset_FS = h5py.File(os.path.join('savedata', id_FS, 'simres.h5'), 'r')


    datakeys = [
        'tvec',
        'el_sd',
        'el_100', 'el_200', 'el_300', 'el_400', 'el_500',
        'el_r',
        'el_r_x',
        'lfp_filtered',
        'CSD',
        'CSD_filtered',
        'CSDr',
        'CSDr_r',
        'diam_best_delta',
        'diam_best_step',
        'diam_best_spline',
        'my_diams_delta',
        'my_diams_step',
        'my_diams_spline',
        'my_errors_delta',
        'my_errors_step',
        'my_errors_spline',
        'my_corrcoefs_delta',
        'my_corrcoefs_step',
        'my_corrcoefs_spline',
        'somav',
        'mean_EPSP', 'SD_EPSP',
        'EPSC', 'mean_EPSC', 'SD_EPSC',
        'syn_i','syn_n', 'mean_syn_i', 'SD_syn_i',
        'syn_pos_x', 'syn_pos_y', 'syn_pos_z',
        'som_pos_x', 'som_pos_y', 'som_pos_z',
        'som_pos0_x', 'som_pos0_y', 'som_pos0_z',
        'v_EPSP', 'v_EPSC', 'v_syn_i',
        'LS_EPSP', 'LS_EPSC', 'LS_syn_i',
        'R2_EPSP', 'R2_EPSC', 'R2_syn_i',
    ]
    datas_RS = {}
    datas_FS = {}
    for k in datakeys:
        print k
        if k == 'diam_best':
            datas_RS[k] = pickle.loads(dataset_RS[k])
            datas_FS[k] = pickle.loads(dataset_FS[k])
        else:
            datas_RS[k] = pl.array(dataset_RS[k])
            datas_FS[k] = pl.array(dataset_FS[k])

    dataset_RS.close()
    dataset_FS.close()

    return tvec, datas_RS, datas_FS

def load_swadlow_stoelzel_data():
    f = open('data/Swadlows2002_1BN1.c', 'rb')
    data_02_1BN1 = cPickle.load(f)
    f.close()
    f = open('data/Swadlows2002_1BN2.c', 'rb')
    data_02_1BN2 = cPickle.load(f)
    f.close()
    f = open('data/Stoelzel2008_1E.c', 'rb')
    data_08_1E = cPickle.load(f)
    f.close()
    f = open('data/Stoelzel2008_2A1.c', 'rb')
    data_08_2A1 = cPickle.load(f)
    f.close()
    return data_02_1BN1, data_02_1BN2, data_08_1E, data_08_2A1

def load_datasets(filelist = []):
    datakeys = [
        'tvec',
        'el_sd',
        'el_100', 'el_200', 'el_300', 'el_400', 'el_500',
        'el_r',
        'el_r_x',
        'lfp_filtered',
        'CSD',
        'CSD_filtered',
        # 'CSD_offset',
        'CSD_diam',
        'CSDr',
        'CSDr_r',
        'diam_best_delta',
        'diam_best_step',
        'diam_best_spline',
        'my_diams_delta',
        'my_diams_step',
        'my_diams_spline',
        'my_errors_delta',
        'my_errors_step',
        'my_errors_spline',
        'my_corrcoefs_delta',
        'my_corrcoefs_step',
        'my_corrcoefs_spline',
        'icsd_delta',
        'icsd_step',
        'icsd_spline',
        'csd_std',
        'CSD76ptF',
        'CSD76pt',
        'syn_pos_x', 'syn_pos_y', 'syn_pos_z',
        'som_pos_x', 'som_pos_y', 'som_pos_z',
        'som_pos0_x', 'som_pos0_y', 'som_pos0_z',
    ]

    datas = {}
    for f in filelist:
        dataset = h5py.File(os.path.join('savedata', f, 'simres.h5'), 'r')
        datas[f] = {}
        for k in datakeys:
            try:
                if k == 'diam_best':
                    datas[f][k] = pickle.loads(dataset[k])
                else:
                    datas[f][k] = pl.array(dataset[k])
            except:
                print "couldn't find %s" % k
        dataset.close()
    return datas

def printRStable(datas, string):
    v_EPSP = pl.r_[datas['v_EPSP'], float(datas['R2_EPSP'])]
    v_EPSC = pl.r_[datas['v_EPSC'], float(datas['R2_EPSC'])]
    v_syni = pl.r_[datas['v_syn_i'], float(datas['R2_syn_i'])]
    print '\n\n'
    print('\\begin{table}[H]')
    print('\caption{Results of LS fit of two-exponential function to averaged %s cell post-synaptic responses.}' % string)
    print('\\begin{center}')
    print('\\begin{tabular*}{\\textwidth}{@{\extracolsep{\\fill}} c c c c c}')
    print('\hline')
    print('Parameter & $\\bar{i}_\mathrm{syn}$ & $\\bar{i}_\mathrm{EPSC}$ & $\\bar{V}_\mathrm{EPSP}$ & unit \\\\')
    print('\hline')
    print('$\\tau_\mathrm{delay}$	& %.3f	& %.3f	& %.3f	& ms \\\\' % (v_syni[0], v_EPSC[0], v_EPSP[0]))
    print('$\\tau_\mathrm{rise}$	& %.3f	& %.3f	& %.3f	& ms \\\\' % (v_syni[1], v_EPSC[1], v_EPSP[1]))
    print('$\\tau_\mathrm{decay}$	& %.3f	& %.3f	& %.3f	& ms \\\\' % (v_syni[2], v_EPSC[2], v_EPSP[2]))
    print('Amplitude 			& %.1f pA	& %.1f pA & %.3f mV &  \\\\' % (v_syni[4]*1E3, v_EPSC[4]*1E3, v_EPSP[4]))
    print('R$^2$ 			& %.3f 	& %.3f  & %.3f  & - \\\\' % (v_syni[5], v_EPSC[5], v_EPSP[5]))
    print('\hline')
    print('\end{tabular*}')
    print('\end{center}')
    print('\label{table:results_%s}' % string)
    print('\end{table}')
    print '\n\n'



def print_nsom_nsyn(name, dset):
    print 'dataset %s, soma count: %i, syn count: %i' % (name, \
                                dset['som_pos_x'].size, dset['syn_pos_x'].size)


def figure_compare_seeds(key, f_keys):
    PS = ParameterSpace(os.path.join('parameters', key+'.pspace'))
    for ncols, _ in enumerate(PS.iter_inner()):
        ncols = ncols + 1
    fig, axes = plt.subplots(2, ncols, figsize=(ncols*2, 8))
    fig.subplots_adjust(left=0.05, right=0.95)
    fig.suptitle(key.replace('_', '\_'))

    # container for minima as list of tuples (position, time, value)
    LFP_min = []
    CSD_min = []

    for i, PSet in enumerate(PS.iter_inner()):
        P = PSet.items()
        P.sort()
        m = md5()
        m.update(pickle.dumps(P))
        id = m.hexdigest()
        print 'column {} id: {}'.format(i+1, id)
        fpath = os.path.join('savedata', id)
        try:
            assert fpath in glob(os.path.join('savedata', '*'))
        except AssertionError as ae:
            raise ae('{} not found'.format(fpath))
        
        f = h5py.File(os.path.join(fpath, 'simres.h5'))
        
        for j, data in enumerate(f_keys):
            
            tvec = f['tvec'].value
            extent = (tvec[0], tvec[-1], 17, 1)
            
            if axes.ndim == 2:
                ax = axes[j, i]
            else:
                ax = axes[j]
            
            clim = abs(f[data].value).max()
            im = ax.imshow(f[data].value, cmap=plt.get_cmap('PRGn', 51) if j == 0 else plt.get_cmap('bwr_r', 51),
                           origin='bottom', clim=(-clim, clim), interpolation='nearest',
                           rasterized=False,
                           extent=extent)
        
            plot_signal_sum(ax, tvec, f[data].value,
                            T=(tvec[0], tvec[-1]),
                            scaling_factor=1., scalebar=True, vlimround=clim)
        
            # find and mark global minima
            min = np.where(f[data].value == f[data].value.min())
            ax.plot(tvec[min[1]], 16 - min[0]+.5, 'wo')
            print('global {} {} minima: position {}, time {} s.'.format(key,
                                                                         'LFP' if j == 0 else 'CSD',
                                                                         16-min[0][0],
                                                                         tvec[min[1]][0]))
            if j == 0:
                LFP_min.append((16-min[0][0], tvec[min[1]][0], f[data].value.min()))
            else:
                CSD_min.append((16-min[0][0], tvec[min[1]][0], f[data].value.min()))

            
            
            ax.set_yticks(np.arange(2, 17, 2)+0.5)
            ax.set_yticklabels(np.arange(2, 17, 2))
        
            ax.axis('tight')
            if j == 0:
                text = ax.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
                plt.setp(ax, xticklabels=[])
            else:
                text = ax.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
            
            if i == 0:
                ax.set_ylabel('Position', ha='left', va='center')
            else:
                ax.set_yticklabels([])
        
            
            if i == ncols-1:
                bbox = np.array(ax.get_position())
                cax = fig.add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                                         0.01, bbox[1][1]-bbox[0][1]-0.1])
                cbar = plt.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
                cbar.ax.set_yticklabels([-1,0,1])

        
        f.close()
    
    LFP_min = np.array(LFP_min)
    out = [[LFP_min[:, h].mean(), LFP_min[:, h].std()] for h in range(3)]
    print 'LFP stats: position mean {:.3f} std {:.3f}, time mean {:.3f} std {:.3f} s, value mean {:.3e} std {:.3e} mV'.format(*flattenlist(out))        
    CSD_min = np.array(CSD_min)
    out = [[CSD_min[:, h].mean(), CSD_min[:, h].std()] for h in range(3)]
    print 'CSD stats: position mean {:.3f} std {:.3f}, time mean {:.3f} std {:.3f} s, value mean {:.3e} std {:.3e} Amm-3'.format(*flattenlist(out))        
    
    return fig


def illustrate_trial_variability_mean_std(PS_keys, f_keys, titles):

    nrows = len(f_keys)
    ncols = len(PS_keys) * 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, 8))

    for h, key in enumerate(PS_keys):
        PS = ParameterSpace(os.path.join('parameters', key+'.pspace'))
        
        # LFP and CSD data container
        DATAs = [[], []]
    
        for i, PSet in enumerate(PS.iter_inner()):
            P = PSet.items()
            P.sort()
            m = md5()
            m.update(pickle.dumps(P))
            id = m.hexdigest()
            print 'column {} id: {}'.format(i+1, id)
            fpath = os.path.join('savedata', id)
            try:
                assert fpath in glob(os.path.join('savedata', '*'))
            except AssertionError as ae:
                raise ae('{} not found'.format(fpath))
            
            f = h5py.File(os.path.join(fpath, 'simres.h5'))
            if i == 0:
                tvec = f['tvec'].value
                extent = (tvec[0], tvec[-1], 17, 1)
            for j, data in enumerate(f_keys):
                DATAs[j] += [f[data].value]
            f.close()
        
        DATAs = np.array(DATAs)
        
        for j, data in enumerate(f_keys):
            for k, OP in enumerate([np.mean, np.std]):
                ax = axes[j, h*2+k]
                clim = abs(OP(DATAs[j], axis=0)).max()
                im = ax.imshow(OP(DATAs[j], axis=0), cmap=plt.get_cmap('PRGn', 51) if j == 0 else plt.get_cmap('bwr_r', 51),
                               origin='bottom', clim=(-clim, clim), interpolation='nearest',
                               rasterized=False,
                               extent=extent)
            
                plot_signal_sum(ax, tvec, OP(DATAs[j], axis=0),
                                T=(tvec[0], tvec[-1]),
                                scaling_factor=1., scalebar=True, vlimround=clim)
            
                # find and mark global minimima, maxima
                if k == 0:
                    min = np.where(OP(DATAs[j], axis=0) == OP(DATAs[j], axis=0).min())
                    ax.plot(tvec[min[1]], 16 - min[0]+.5, 'wo')
                else:
                    max = np.where(OP(DATAs[j], axis=0) == OP(DATAs[j], axis=0).max())
                    ax.plot(tvec[max[1]], 16 - max[0]+.5, 'ko')
                    
                
                ax.set_yticks(np.arange(2, 17, 2)+0.5)
                ax.set_yticklabels(np.arange(2, 17, 2))
            
                ax.axis('tight')
                if j == 0:
                    text = ax.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$mV'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
                    plt.setp(ax, xticklabels=[])
                else:
                    text = ax.text(-0.5, 16, r'$\pm {}\cdot 10^{{{:.0f}}}~$Amm$^{{-3}}$'.format(*[float(flt) for flt in ('%.1e' % clim).split('e')]), color='k')
                text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                           path_effects.Normal()])
                
                if h*2+k == 0:
                    ax.set_ylabel('Position', ha='left', va='center')
                else:
                    ax.set_yticklabels([])
            
                if j == 0:
                    ax.set_title(titles[h*2+k])
                
                if h*2+k == ncols-1:
                    bbox = np.array(ax.get_position())
                    cax = fig.add_axes([bbox[1][0]+0.005, bbox[0][1]+0.05,
                                             0.01, bbox[1][1]-bbox[0][1]-0.1])
                    cbar = plt.colorbar(im, cax=cax, ticks=[-clim, 0, clim])
                    cbar.ax.set_yticklabels([-1,0,1])
  
    return fig


def illustrate_PS_populations(key, N=20, synapses=True):
    PS = ParameterSpace(os.path.join('parameters', key+'.pspace'))
    for ncols, _ in enumerate(PS.iter_inner()):
        ncols = ncols + 1
    fig, axes = plt.subplots(1, ncols, figsize=(ncols*2, 4), sharey=True)
    
    for i, PSet in enumerate(PS.iter_inner()):
        P = PSet.items()
        P.sort()
        m = md5()
        m.update(pickle.dumps(P))
        id = m.hexdigest()
        print 'column {} id: {}'.format(i+1, id)
        fpath = os.path.join('savedata', id)
        try:
            assert fpath in glob(os.path.join('savedata', '*'))
        except AssertionError as ae:
            raise ae('{} not found'.format(fpath))
    
    
        c_saved = uncPickle(os.path.join('savedata', id, 'c_savedPickle.cpickle'))
        ax = axes[i]
    
        ax.plot(pl.zeros(16), pl.linspace(-700, 800, 16),
                        color='w', marker='o', linestyle='None')
        for i in [0, 1, 2, 13, 14, 15]:
            ax.text(-40, pl.linspace(-700, 800, 16)[i], 'Pos. %i' % (16-i),
                    horizontalalignment='right',
                    verticalalignment='center')
    
        for n in xrange(len(c_saved.items())):
            color = [pl.rand()*0.3+0.3, pl.rand()*0.3+0.3, pl.rand()*0.3+0.3]
    
            if n < N:
                plot_morpho(ax, c_saved[n], color,
                            zorder=-abs(c_saved[n].somapos[1]))
                if synapses:
                    for i in xrange(len(c_saved[n].synapses)):
                        c_saved[n].synapses[i].update_pos(c_saved[n])
                        ax.scatter([c_saved[n].synapses[i].x],
                            [c_saved[n].synapses[i].z],
                            marker='o', s=10, zorder=0, facecolor='r',
                            edgecolor='gray', linewidth=0.25,
                            )    
    
        ax.plot([100, 100],[-600, -700], lw=3, color='k')
        ax.text(120, -650, '100 $\upmu$m',
                horizontalalignment='left',
                verticalalignment='center')
    
    
        ax.axis(ax.axis('equal'))
        ax.yaxis.set_ticks(pl.arange(-250, 300, 250))
        ax.xaxis.set_ticks(pl.arange(-250, 300, 250))
    
        adjust_spines(ax,['left', 'bottom'])
        ax.spines['left'].set_bounds( -250, 250)
        ax.set_yticks([-250, 0, 250])
        ax.spines['bottom'].set_bounds( -250, 250)
        ax.set_xticks([-250, 0, 250])
        pl.ylabel(r'$z$ ($\mathrm{\upmu}$m)', ha='center', va='center')
        newaxis = pl.array(ax.axis()) * 1.1
        ax.axis(newaxis)
        ax.text(0.5, -0.1, r'$x$ ($\mathrm{\upmu}$m)',
                horizontalalignment='center',
                verticalalignment='center',transform=ax.transAxes)

    return fig



if __name__ == '__main__':

    data_02_1BN1, data_02_1BN2, data_08_1E, data_08_2A1 = load_swadlow_stoelzel_data()
    
    tvec, datas_RS, datas_FS = figure_data('PS_simres_RS', 'PS_simres_FS')
    print_nsom_nsyn('simres_RS_cell', datas_RS)
    print_nsom_nsyn('simres_FS_cell', datas_FS)

    extent = (tvec[0], tvec[-1], 17, 1)

    printRStable(datas_RS, 'RS')
    printRStable(datas_FS, 'FS')

    # set up dictionary with various other simulation cases
    datas_d = {}
    
    # cylinder, variable diameter
    datas = load_datasets(get_md5s('PS_simres_RXXX'))
    for key, value in datas.items():
        PSet = ParameterSet(os.path.join('parameters', key+'.pset'))
        datas_d['simres_R{:03d}'.format(PSet['sigma'][1])] = value
    
    # gaussian
    datas = load_datasets(get_md5s('PS_simres_Gaussian'))
    for key, value in datas.items():
        PSet = ParameterSet(os.path.join('parameters', key+'.pset'))
        datas_d['simres_Gaussian'] = value

    # cylindrical
    datas = load_datasets(get_md5s('PS_simres_Cylindrical'))
    for key, value in datas.items():
        PSet = ParameterSet(os.path.join('parameters', key+'.pset'))
        datas_d['simres_Cylindrical'] = value
        
    # spherical
    datas = load_datasets(get_md5s('PS_simres_Spherical'))
    for key, value in datas.items():
        PSet = ParameterSet(os.path.join('parameters', key+'.pset'))
        datas_d['simres_Spherical'] = value
        
    for dset in datas_d.keys():
        print_nsom_nsyn(dset, datas_d[dset])

    # put figure files in the subfolder "figures"
    if not os.path.isdir('figures'):
        os.mkdir('figures')

    # ### Figures in Hagen et al. (submitted) ###################################
    fileformats = ['.pdf', '.eps']
    bbox_inches = 'tight'
    pad_inches = 0.05
    dpi = 200 # not presently effective, because no rasterization
    
    figure_1(synapses=True, NRS=40, NFS=10)
    for fileformat in fileformats:
        pl.savefig(os.path.join('figures', 'figure_1' + fileformat),
                   dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    figure_2(data='el_sd')
    for fileformat in fileformats:
        pl.savefig(os.path.join('figures', 'figure_2' + fileformat),
                   dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    figure_3(data='el_sd')
    for fileformat in fileformats:
        pl.savefig(os.path.join('figures', 'figure_3' + fileformat),
                   dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    figure_4(exp_data='raw_data', model_data='el_sd', globalclim=False)
    for fileformat in fileformats:
        pl.savefig(os.path.join('figures', 'figure_4' + fileformat),
                   dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    figure_5(filtering=False, globalscaling=False, rasterized=False)
    for fileformat in fileformats:
        pl.savefig(os.path.join('figures', 'figure_5' + fileformat),
                   dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        
    figure_6(filtering=False)
    for fileformat in fileformats:
        pl.savefig(os.path.join('figures', 'figure_6' + fileformat),
                   dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    figure_7(dset=pl.array(['simres_Spherical', 'simres_Cylindrical',
                            'simres_Gaussian']),
            filtering=False)
    for fileformat in fileformats:
        pl.savefig(os.path.join('figures', 'figure_7' + fileformat),
                   dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    figure_8(filtering=False)
    for fileformat in fileformats:
        pl.savefig(os.path.join('figures', 'figure_8' + fileformat),
                   dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    figure_9(dset = 'simres_R200', data='el_sd')
    for fileformat in fileformats:
        pl.savefig(os.path.join('figures', 'figure_9' + fileformat),
                   dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    figure_10(data='raw_data')        
    for fileformat in fileformats:
        pl.savefig(os.path.join('figures', 'figure_10' + fileformat),
                   dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    
    ##### Auxiliary figures and testing plots ##################################
    
    plt.close('all') # kill all open figures
    
    PS_keys = ['PS_simres_RS_seed', 'PS_simres_FS_seed', 'PS_simres_FS_seed_50']
    f_keys = ['el_sd', 'CSD_filtered']    
    for key in PS_keys:
        fig = figure_compare_seeds(key, f_keys)
        fig.savefig(os.path.join('figures', 'seedtest_{}'.format(key) + fileformat), dpi=dpi, bbox_inches=bbox_inches)
    
    
    PS_keys = ['PS_simres_RS_seed', 'PS_simres_FS_seed', 'PS_simres_FS_seed_50']
    f_keys = ['el_sd', 'CSD_filtered']
    titles = ['RS avg.', 'RS std.',
              'FS avg.', 'FS std.',
              r'FS$_{50\%}$ avg.', r'FS$_{50\%}$ std.']
    fig = illustrate_trial_variability_mean_std(PS_keys, f_keys, titles)
    fig.savefig(os.path.join('figures', 'seedtest_mean_std'.format(key) + fileformat), dpi=dpi, bbox_inches=bbox_inches)
    
    
    figure_1(synapses=True, NRS=40, NFS=10, PS=['PS_simres_P4', 'PS_simres_FS'])
    pl.savefig(os.path.join('figures', 'figure_1x' + fileformat),
               dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    
    # compare LFP and CSD of spiny stellates and pyramidal cells  ##############
    PS_keys = ['PS_simres_RS', 'PS_simres_P4']
    f_keys = ['el_sd', 'CSD_filtered']  
    for key in PS_keys:
        fig = figure_compare_seeds(key, f_keys)
        fig.savefig(os.path.join('figures', 'SS_vs_PC_{}'.format(key) + fileformat),
                    dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    
    # plot some neuron populations
    keys = ['PS_simres_RS', 'PS_simres_P4']
    for key in keys:
        fig = illustrate_PS_populations(key, N=10, synapses=True)
        fig.savefig(os.path.join('figures', 'SS_vs_PC_populations_{}'.format(key) + fileformat),
                    dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    # compare lateral LFP from different pyramidal cell populations ############    
    # find column index corresponding to tstep
    tstep=1.5
    [col] = pl.where(datas_d['simres_Spherical']['tvec'] == tstep)[0]
    
    fig, axes = plt.subplots(2,1)
    fig.subplots_adjust(left=0.15)
    for ps_id in get_md5s('PS_simres_RS'):
        f = h5py.File(os.path.join('savedata', ps_id, 'simres.h5'))
        axes[0].plot(f['el_r_x'], f['el_r'][:, col], lw=2, alpha=0.5)
        axes[1].loglog(f['el_r_x'], abs(f['el_r'][:, col]), lw=2, alpha=0.5)
    for ps_id in get_md5s('PS_simres_P4'):
        f = h5py.File(os.path.join('savedata', ps_id, 'simres.h5'))
        axes[0].plot(f['el_r_x'], f['el_r'][:, col], lw=2, ls=':')
        axes[1].loglog(f['el_r_x'], abs(f['el_r'][:, col]), lw=2, ls=':')
    
    
    axes[1].loglog([1E3, 4E2], [1E-5, 6.25E-5], '-', lw=2, color=[0,0,0],
        label='$\propto r^{-2}$')
    axes[1].loglog([1E3, 4E2], [1E-5, 15.625E-5], '-', lw=2, color=[0.33,0.33,0.33],
        label='$\propto r^{-3}$')
    axes[1].loglog([1E3, 4E2], [1E-5, 39.0625E-5], '-', lw=2, color=[0.66,0.66,0.66],
        label='$\propto r^{-4}$')
    axes[1].grid('on')
    axes[1].set_xlabel(r'$x (\upmu\mathrm{m})$', labelpad=0)
    
    
    ylabels = [r'$\phi(x)$', r'$|\phi(x)|$']
    for ax, ylabel in zip(axes, ylabels):
        ax.axis(ax.axis('tight'))
        ax.set_ylabel(ylabel)
    fig.savefig(os.path.join('figures', 'lateral_LFP_SS_vs_PC' + fileformat),
                dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    
    pl.show()

