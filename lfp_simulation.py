#!/usr/bin/env python
'''
Main LFP simulation script. Use the script initialize_simulations.py to
generate parameter set files and batch submit jobs to the cluster queue. 

Usage:

    python lfp_simulation.py <parameterset_file>


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
import LFPy
import h5py
import sys
import openopt as opt
import cPickle
import neo
import quantities as pq
from elephant.current_source_density import icsd
import neuron
import tools
from glob import glob
from time import time
from NeuroTools.parameters import ParameterSet
from multiprocessing import Process, Queue, freeze_support, cpu_count, Pool


figparams = {
    'text.usetex' : False,
    'font.family': 'serif',
}
pl.rcParams.update(pl.rcParamsDefault)
pl.rcParams.update(figparams)

################################################################################
#
# FUNCTION DEFINITIONS
#
################################################################################

def set_nsegs_lambda_f(frequency, nsegsoma):
    '''my own method with multicomp soma'''
    for sec in neuron.h.allsec():
        sec.nseg = int((sec.L / (0.1 * neuron.h.lambda_f(frequency)) + .9)
            / 2 )*2 + 1
    for sec in neuron.h.soma:
        sec.nseg = nsegsoma

def random_rot_angles(x=True, y=True, z=True):
    rotation = {}
    if x:
        rotation['x'] = pl.random()*2*pl.pi
    if y:
        rotation['y'] = pl.random()*2*pl.pi
    if z:
        rotation['z'] = pl.random()*2*pl.pi

    return rotation

def risetime(t, y, limit=[0.1, 0.9]):
    y = (y-y.min())/(y-y.min()).max()

    i0 = pl.where(y > limit[0])
    i1 = pl.where(y > limit[1])

    t0 = t[i0][0]
    t1 = t[i1][0]

    t_rise = t1 - t0
    return t_rise

def R2(data, fit):
    SS_tot = sum((data - pl.mean(data))**2)
    SS_err = sum((data - fit)**2)
    R2 = 1 - SS_err/SS_tot
    return R2

def _normalize(y):
    '''normalize to the interval [0, 1]'''
    y_norm = (y-y.min())/(y.max()-y.min())
    return y_norm

def fp(v, t):
    y = pl.zeros(pl.shape(t)) + v[3]
    [i] = pl.where((t>v[0]))
    num = abs(pl.exp(-(t[i]-v[0])/v[2]) - pl.exp(-(t[i]-v[0])/v[1]))
    if i.size > 0:
        denom = abs(pl.exp(-(t[i]-v[0])/v[2]) - pl.exp(-(t[i]-v[0])/v[1])).max()
        y[i] = v[3] + v[4]*num/denom
    return y

error = lambda v,  t,  y: (fp(v, t)-y)

def v_view(v):
    s = 'tau: %.4f\ntau1: %.4f\ntau2: %.4f\ny0: %.4f\na: %.4f' % \
          (v[0], v[1], v[2], v[3], v[4])
    return s

difference = lambda v, v1, v2: (v*v1 - v2)

def fit_alpha(v0, tvec, signal):
    '''return best fit coefficient for fit to alpha-function, where
    v0 = [delay, tau_rise, tau_decay, amplitude]'''
    A = [   [-1,  0,  0,  0,  0],
            [ 0, -1,  0,  0,  0],
            [ 0,  1, -1,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            ]
    b = [1.E-6, 1.E-6, 1.E-6, 0, 0]
    ub = [4., pl.inf, pl.inf, 100, pl.inf]
    lb = [1., -pl.inf, -pl.inf, -100, -pl.inf]


    def costfun(v, tvec, signal):
        return pl.sqrt((fp(v, tvec)-signal)**2).sum()
    
    objfnc = lambda v: costfun(v, tvec, signal)


    p = opt.NLP(f = objfnc, x0=v0, A=A, b=b, ub=ub, lb=lb)
    
    p.ftol = 1.E-9
    p.maxIter = 10000
    p.maxLineSearch = 5000
    r = p.solve('ralg')

    vf = r.xf

    return vf


def input_init(lfp_data,
               z_data,
               diam=500E-6*pq.m,
               cond=0.3*pq.S/pq.m,
               cond_top=0.3*pq.S/pq.m):
    '''Input dictionaries for each method'''
    input = {
        'delta' : {
            'lfp' : lfp_data,
            'coord_electrode' : z_data,
            'diam' : diam,        # source diameter
            'sigma' : cond,           # extracellular conductivity
            'sigma_top' : cond_top,       # conductivity on top of cortex
            'f_type' : 'gaussian',  # gaussian filter
            'f_order' : (3, 1),     # 3-point filter, sigma = 1.
        },
        'step' : {
            'lfp' : lfp_data,
            'coord_electrode' : z_data,
            'h' : 100E-6*pq.m,
            'diam' : diam,
            'sigma' : cond,
            'sigma_top' : cond_top,
            'tol' : 1E-12,          # Tolerance in numerical integration
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        },
        'spline' : {
            'lfp' : lfp_data,
            'coord_electrode' : z_data,
            'diam' : diam,
            'sigma' : cond,
            'sigma_top' : cond_top,
            'num_steps' : 76,      # Spatial CSD upsampling to N steps
            'tol' : 1E-12,
            'f_type' : 'gaussian',
            'f_order' : (19, 5),
        },
        'std' : {
            'lfp' : lfp_data,
            'coord_electrode' : z_data,
            'vaknin_el' : True,
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        },
    }
    return input

def cell_thread(task_queue, done_queue):
    '''thread function for simulation of single neuron process'''
    for n in iter(task_queue.get, 'STOP'):
        print 'Cell number %s out of %d.' % (n, pop_params['n']-1)
        cell = LFPy.Cell(**cellparams)

        cell.set_pos(**soma_pos[withsyn_i[n]])
        cell.set_rotation(**rotation[withsyn_i[n]])
        cell.color = 'g'

        #synapses
        e_synparams = e_synparams_init.copy()
        for idx in allidx_e[n]:
            e_synparams = e_synparams_init.copy()
            e_synparams.update({
                'idx'   : int(idx),
                'color' : 'r'
            })
            e_spiketimes = [pl.array([spiketime])]

            s_e = LFPy.Synapse(cell, **e_synparams)
            s_e.set_spike_times(e_spiketimes[0])

        cell.simulate(**simparams)

        cell.strip_hoc_objects()

        done_queue.put([e_synparams, cell])

def __csd_error(v, m):
    '''using squared difference summed'''
    if m == 'delta':
        icsd_input[m].update({'diam' : v*pq.m})
        _icsd = icsd.DeltaiCSD(**icsd_input[m])
        error = ((pl.array(_icsd.filter_csd(_icsd.get_csd()) / (100E-6*pq.m)).reshape(CSD_filtered.size)*1E-9 -
                  CSD_filtered.reshape(CSD_filtered.size))**2).sum()
    elif m == 'step':
        icsd_input[m].update({'diam' : v*pq.m})
        _icsd = icsd.StepiCSD(**icsd_input[m])
        error = ((pl.array(_icsd.filter_csd(_icsd.get_csd())).reshape(CSD_filtered.size)*1E-9 -
                  CSD_filtered.reshape(CSD_filtered.size))**2).sum()
    elif m == 'spline':
        icsd_input[m].update({'diam' : v*pq.m})
        _icsd = icsd.SplineiCSD(**icsd_input[m])
        error = ((pl.array(_icsd.filter_csd(_icsd.get_csd())).reshape(CSD76ptF.size)*1E-9 -
                  CSD76ptF.reshape(CSD76ptF.size))**2).sum()
    else:
        raise Exception, 'm = %s should be either [delta, step, spline]' % m

    return error

def __csd_error_thr(task_queue, diams, m, done_queue):
    for n in iter(task_queue.get, 'STOP'):
        error = __csd_error(diams[n], m)
        done_queue.put([n, error])

def minimize_icsd_error_brute(m, exps=pl.log10(pl.linspace(1E-5, 1E-3, 100))):
    '''
    Return summed squared difference between iCSD estimates and ground truth CSD
    for a series of assumed source diameters
    '''
    if m != 'delta' and m != 'step' and m != 'spline':
        raise ValueError, 'Must have iCSD method'
    diams = 10**exps
    errors = pl.zeros(diams.size)
    if __name__ == '__main__':
        freeze_support()
        task_queue = Queue()
        done_queue = Queue()

        TASKS = pl.arange(diams.size)

        for task in TASKS:
            task_queue.put(int(task))
        for i in xrange(NUMBER_OF_PROCESSES):
            Process(target=__csd_error_thr,
                    args=(task_queue, diams, m, done_queue)).start()
        for n in xrange(TASKS.size):
            nn, error = done_queue.get()
            errors[nn] = error
        for i in xrange(NUMBER_OF_PROCESSES):
            task_queue.put('STOP')

        task_queue.close()
        done_queue.close()

        return errors, diams, diams[pl.where(errors == errors.min())]



def __csd_correlation(v, m):
    '''compute correlation coefficient between CSD estimate and CSD
    for a given source diameter'''
    if m == 'delta':
        icsd_input[m].update({'diam' : v*pq.m})
        _icsd = icsd.DeltaiCSD(**icsd_input[m])
        corrcoef = pl.corrcoef(CSD_filtered.flatten(),
                        pl.array(_icsd.filter_csd(_icsd.get_csd()) / pq.m).flatten())
    elif m == 'step':
        icsd_input[m].update({'diam' : v*pq.m})
        _icsd = icsd.StepiCSD(**icsd_input[m])
        corrcoef = pl.corrcoef(CSD_filtered.flatten(),
                        pl.array(_icsd.filter_csd(_icsd.get_csd()) / pq.m).flatten())
    elif m == 'spline':
        icsd_input[m].update({'diam' : v*pq.m})
        _icsd = icsd.SplineiCSD(**icsd_input[m])
        corrcoef = pl.corrcoef(CSD76ptF.flatten(),
                        pl.array(_icsd.filter_csd(_icsd.get_csd()) / pq.m).flatten())
    else:
        raise Exception, 'm = %s should be either [delta, step, spline]' % m

    return corrcoef[0, -1]

def __csd_correlation_thr(task_queue, diams, m, done_queue):
    for n in iter(task_queue.get, 'STOP'):
        corrcoef = __csd_correlation(diams[n], m)
        done_queue.put([n, corrcoef])

def maximize_icsd_correlation_brute(m, exps=pl.log10(pl.linspace(1E-5, 1E-3, 100))):
    '''
    Return correlation coefficient between iCSD estimates and ground truth CSD
    for a series of assumed source diameters
    '''
    if m != 'delta' and m != 'step' and m != 'spline':
        raise ValueError, 'Must have iCSD method'
    diams = 10**exps
    corrcoefs = pl.zeros(diams.size)
    if __name__ == '__main__':
        freeze_support()
        task_queue = Queue()
        done_queue = Queue()

        TASKS = pl.arange(diams.size)

        for task in TASKS:
            task_queue.put(int(task))
        for i in xrange(NUMBER_OF_PROCESSES):
            Process(target=__csd_correlation_thr,
                    args=(task_queue, diams, m, done_queue)).start()
        for n in xrange(TASKS.size):
            nn, corrcoef = done_queue.get()
            corrcoefs[nn] = corrcoef
        for i in xrange(NUMBER_OF_PROCESSES):
            task_queue.put('STOP')

        task_queue.close()
        done_queue.close()

        return corrcoefs, diams, diams[pl.where(corrcoef == corrcoef.max())]

# def minimize_icsd_error(m):
#     '''Try to minimize the error in icsd estimates by adjusting diam'''
#     A = [[1.]]
#     b = [50E-6]
#     lb = [50E-6]
#     ub = [1000E-6]
#     v0 = icsd_diam
# 
#     csd_error = lambda v: __csd_error(v, m)
# 
#     rr = {}
#     pp = {}
# 
#     solvers = ['ralg', 'scipy_slsqp', 'scipy_cobyla', 'pswarm']
#     for solver in solvers:
#         if solver == 'pswarm':
#             p = opt.GLP(f = csd_error, A=A, b=b, lb=lb,
#                         ub=ub, plot=1, show=False)
#         else:
#             p = opt.NLP(f = csd_error, x0=v0, A=A, b=b,
#                         lb=lb, ub=ub, plot=1, show=False)
# 
#         r = p.solve(solver)
# 
#         pp[solver] = p
#         rr[solver] = r
# 
#     return pp, rr

def CSD_thread(task_queue, icsd_diam, done_queue):
    for n in iter(task_queue.get, 'STOP'):
        print 'Cell number %s out of %d.' % (n, pop_params['n'])
        radii = pl.arange(50, 550, 50)
        CSD_diam = pl.zeros((radii.size, 16,
                             cells[n].tvec.size))
        for i in xrange(radii.size):
            [CSD_diam[i, :, :], z_data] = tools.true_lam_csd(cells[n],
                             dr=radii[i], z=electrodeparams['z'][:16])

        if disttype == 'hard_cyl':
            dr = sigma[1]
        else:
            dr = icsd_diam/2*1E6 #m -> mum

        [CSD, z_data] = tools.true_lam_csd(cells[n],
                                        dr=dr, z=electrodeparams['z'][:16])
        [CSD76pt, z_data76pt] = tools.true_lam_csd(cells[n],
                                        dr=dr, z=pl.linspace(-700, 800, 76))
        #cheat with the time axis;
        tvec = pl.arange(-1, cellparams['timeres_python']*len(cells[n].tvec)-1,
                         cellparams['timeres_python'])
        done_queue.put([CSD_diam, CSD, CSD76pt, tvec, z_data])

def CSD_donut_thr(task_queue, done_queue):
    for n in iter(task_queue.get, 'STOP'):
        print 'Cell %i' % n
        z, r, CSD_donut = tools.donut_csd(cells[n], rmax=250, zlim=250)
        done_queue.put(CSD_donut)

def CSDr_thr(task_queue, done_queue):
    for n in iter(task_queue.get, 'STOP'):
        print 'Cell %i' % n
        CSDr, CSDr_r = tools.tru_sphershell_csd(cells[n], r = pl.arange(0, 520, 20))
        done_queue.put([CSDr_r, CSDr])


def cell_thread_PSC(task_queue, done_queue):
    '''thread function for simulation of single neuron process'''
    for n in iter(task_queue.get, 'STOP'):
        print 'Cell number %s out of %d.' % (n, pop_params['n']-1)
        cell = LFPy.Cell(**cellparams)

        e_synparams = e_synparams_init.copy()
        for idx in allidx_e[n]:
            e_synparams = e_synparams_init.copy()
            e_synparams.update({
                'idx'   : int(idx),
                'color' : (0.8, 0.8, 0.8)
            })
            e_spiketimes = [pl.array([spiketime])]

            s_e = LFPy.Synapse(cell, **e_synparams)
            s_e.set_spike_times(e_spiketimes[0])

        e_synparams = e_synparams_init.copy()

        LFPy.StimIntElectrode(cell, **pointprocparams)

        cell.simulate(**simparams2)
        cell.tvec = pl.arange(-1, cellparams['timeres_python']*len(cell.tvec)-1,
                           cellparams['timeres_python'])

        cell.strip_hoc_objects()

        done_queue.put(cell)


def get_idx_proximal(cell, r=25.):
    '''find which idx are within radius from midpoint of soma'''
    
    r2 = (cell.xmid - cell.somapos[0])**2 + \
        (cell.ymid - cell.somapos[1])**2 + \
        (cell.zmid - cell.somapos[2])**2
    
    return pl.where(r2 <= r**2)[0]


def c0_thread(task_queue, randomseed, done_queue):
    for n in iter(task_queue.get, 'STOP'):
        print '\nCell number %s out of %d.' % (n, pop_params['n']-1)

        pl.seed(randomseed - n)

        cell = LFPy.Cell(**cellparams)

        soma_pos = {
            'xpos' : pop_soma_pos0['xpos'][n],
            'ypos' : pop_soma_pos0['ypos'][n],
            'zpos' : pop_soma_pos0['zpos'][n]
        }
        cell.set_pos(**soma_pos)

        cell.color = 'g'
        
        # make list of cells with morphology rotation file
        L4_pc = [fname.split('.rot')[0]+'.hoc' for fname in glob(os.path.join('morphologies', '*.rot'))]


        if cellparams['morphology'] not in L4_pc:
            rotation = random_rot_angles()
        else:
            rotation = random_rot_angles(x=False, y=False, z=True)
        cell.set_rotation(**rotation)


        #Manage probabilities of synapse positions
        if section_syn == 'somaproximal':
            idxs = get_idx_proximal(cell, r=50.)
        else:
            idxs = cell.get_idx(section=section_syn)
            
        P = cell.get_rand_prob_area_norm_from_idx(idx=idxs)
        
        idx = []

        if disttype == 'hard_cyl':
            rad_xy = pl.sqrt(cell.xmid[idxs]**2 + cell.ymid[idxs]**2)
            indices = pl.where( pl.array(rad_xy < sigma[1], dtype=int) * \
                pl.array(cell.zmid[idxs] <= sigma[0]-my, dtype=int) * \
                pl.array(cell.zmid[idxs] > -sigma[0]-my, dtype=int) == 1)
            W = pl.zeros(idxs.size)
            W[indices] = 1
            for i in xrange(idxs.size):
                if W[i] == 1: # synapse allowed
                    for j in xrange(pl.poisson(mean_n_syn)):
                        base_prob = P[i]/P.sum()
                        if pl.rand() < base_prob:
                            idx.append(i)
            allidx_e0 = idxs[idx]
        elif disttype == 'hard_sphere':
            rad_xyz = pl.sqrt(cell.xmid[idxs]**2 + cell.ymid[idxs]**2 +
                              (cell.zmid[idxs]-my)**2)
            indices = pl.where( pl.array(rad_xyz < sigma, dtype=int) == 1)
            W = pl.zeros(idxs.size)
            W[indices] = 1
            for i in xrange(idxs.size):
                if W[i] == 1:
                    for j in xrange(pl.poisson(mean_n_syn)):
                        base_prob = P[i]/P.sum()
                        if pl.rand() < base_prob:
                            idx.append(i)
            allidx_e0 = idxs[idx]
        elif disttype == 'anisotrop':
            W = pl.exp(-cell.xmid[idxs]**2/(2*sigma[0]**2)) * \
                pl.exp(-cell.ymid[idxs]**2/(2*sigma[1]**2)) * \
                pl.exp(-(cell.zmid[idxs]-my)**2/(2*sigma[2]**2))
            PW = P*W
            for i in xrange(idxs.size):
                for j in xrange(pl.poisson(mean_n_syn)):
                    base_prob = PW[i] / P.sum()
                    if pl.rand() < base_prob:
                        idx.append(i)
            allidx_e0 = idxs[idx]

        cell.strip_hoc_objects()

        done_queue.put([cell, rotation, soma_pos, allidx_e, allidx_e0])





################################################################################
# MAIN PROGRAM
#
# Dealing with the parameterspaces
################################################################################

pl.close('all')

tstart = time()
simtime = 6.


#create variable names
pop_geom = None
weight = None
custom_code = None
Ra = None
morphology = None
tau2 = None
cm = None
mean_n_syn = None
spiketime = None
disttype = None
pop_params_n = None
tau1 = None
rm = None
icsd_diam = None
active = None
uuid = None
e = None
sigma = None
v_init = None
section_syn = None
simulated = None
my = None
electrode_r = None
randomseed = None


# Grab command line input
parametersetfile = sys.argv[1]
if not os.path.isfile(parametersetfile):
    raise Exception, 'provide parameterset filename on command line'
PSet = ParameterSet(parametersetfile)

#create some variables from parameter set
for i in PSet.iterkeys():
    vars()[i] = PSet[i]

psetid = PSet['uuid']

print('Current simulation are using ParameterSet:')
print PSet.pretty()

#create folder to save data if it doesnt exist
datafolder = os.path.join('savedata', psetid)
if not os.path.isdir(datafolder):
    os.system('mkdir %s' % datafolder)
    print 'created folder %s!' % datafolder


# set global seed
pl.seed(seed=randomseed)


################################################################################
# Simulation setup
################################################################################
cellparams = {
    'morphology' : morphology,
    'timeres_NEURON' : 0.025,
    'timeres_python' : 0.025,
    'custom_code' : custom_code,
    'rm' : rm,
    'cm' : cm,
    'Ra' : Ra,
    'e_pas' : v_init,
    'v_init' : v_init,
    'tstartms' : -1,
    'tstopms' : simtime,
    'nsegs_method' : None,
    'custom_fun' : [set_nsegs_lambda_f],
    'custom_fun_args'  : [{'frequency' : 1000, 'nsegsoma' : 11}]
}


simparams = {
    'rec_imem' : True,
    'rec_vmem' : True,
    'rec_isyn' : True,
    'rec_vmemsyn' : True,
}
simparams2 = {
    'rec_istim' : True
}
pop_params = {
    'n' : pop_params_n,
    'radius' : pop_geom[0],
    'tstart' : 0,
    'tstop' : simtime,
    'z_min' : pop_geom[1],
    'z_max' : pop_geom[2],
}

#LFP from bottom to top from zero-500 mum x-offset 
N = pl.empty((96, 3))
for i in range(N.shape[0]):
    N[i,] = [0, 1, 0]
x = pl.linspace(0, 500, 6)
z = pl.linspace(-700, 800, 16)
X, Z = pl.meshgrid(x, z)
electrodeparams = {
    'x' : X.T.reshape(-1),
    'y' : pl.zeros(96),
    'z' : Z.T.reshape(-1),
    'sigma' : 0.3,
    'color' : 'g',
    'marker' : 'o',
    'N' : N,
    'r' : electrode_r,
    'n' : 100,
    'verbose' : True,
    'method' : 'linesource'
}

#along x-axis
el_r_par = electrodeparams.copy()
N = pl.empty((21, 3))
for i in range(N.shape[0]):
    N[i,] = [0, 1, 0]
el_r_par.update({
    'x' : 10**pl.linspace(0, 3, 21),
    'y' : pl.zeros(21),
    'z' : pl.zeros(21),
    'N' : N,
    })

pointprocparams = {
    'idx' : 5,
    'pptype' : 'SEClamp',
    'amp1' : v_init,
    'dur1' : 100.,
    'amp2' : v_init,
    'dur2' : 100.,
    'amp3' : v_init,
    'dur3' : 100.,
    'rs' : 1E-3,
    'record_current' : True,
    'color' : 'p',
    'marker' : '*'
}
e_synparams_init = {
    'syntype':'Exp2Syn',
    'tau1' : tau1,
    'tau2' : tau2,
    'weight': weight,
    'e' : e,
    'record_current' : True,
    'record_potential' : True,
}

#time constants of EPSP's and EPSC's,  input to plot_beta(v, t)
if morphology == 'C120398A-I4.CNG_sansAxon.hoc':
    v_EPSP_measured = pl.array([1.45,  0.37142164,  5.49141455,  v_init,  4.1])
    v_EPSC_measured = pl.array([1.45,  0.0892241,  0.53488082,  0.,  -229E-3])
elif morphology == 'C120398A-P1.CNG_sansAxon.hoc':
    v_EPSP_measured = pl.array([1.50,  1.32059472,  11.79607538,  v_init,  2.4])
    v_EPSC_measured = pl.array([1.45,  0.44408851,  2.50966917,  0.,  -56E-3])
else:
    v_EPSP_measured = pl.array([1.50,  1.32059472,  11.79607538,  v_init,  2.4])
    v_EPSC_measured = pl.array([1.45,  0.44408851,  2.50966917,  0.,  -56E-3])

#Deal with population,  and the number of synapses on each neuron!
lp = tools.Population(**pop_params)
pop_soma_pos0 = lp.draw_rand_pos()
pop_soma_pos = {}
syn_n = []
cells = {}
rotation = {}
allidx_e0 = {}
allidx_e = {}
soma_pos = {}

if __name__ == '__main__':
    freeze_support()
    NUMBER_OF_PROCESSES = cpu_count() #leaving one core to handle threads
    task_queue = Queue()
    done_queue = Queue()

    TASKS = pl.arange(0, pop_params['n'])

    for task in TASKS:
        task_queue.put(int(task))
    for i in xrange(NUMBER_OF_PROCESSES):
        Process(target=c0_thread,
                args=(task_queue, randomseed, done_queue)).start()
    for n in xrange(TASKS.size):
        [cells[n], rotation[n], soma_pos[n],
                allidx_e[n], allidx_e0[n]] = done_queue.get()
    for i in xrange(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')

    task_queue.close()
    done_queue.close()

for n in xrange(pop_params['n']):
    syn_n.append(allidx_e0[n].size)
    if allidx_e0[n].size == 0:
        del cells[n]
        del allidx_e0[n]
        del soma_pos[n]
        del rotation[n]

#Rebuild the population to only contain neurons with input!
[nosyn_i] = pl.where(pl.array(syn_n) == 0)
[withsyn_i] = pl.where(pl.array(syn_n) != 0)

idx_params0 = {}
idx_params = {}

som_pos_x = []
som_pos_y = []
som_pos_z = []

syn_pos_x = []
syn_pos_y = []
syn_pos_z = []

for n in allidx_e0:
    for i in allidx_e0[n]:
        syn_pos_x.append(cells[n].xmid[i])
        syn_pos_y.append(cells[n].ymid[i])
        syn_pos_z.append(cells[n].zmid[i])

for n in withsyn_i:
    som_pos_x.append(soma_pos[n]['xpos'])
    som_pos_y.append(soma_pos[n]['ypos'])
    som_pos_z.append(soma_pos[n]['zpos'])


for n in xrange(pop_params['n']):
    idx_params0[n] = {
        'section' : section_syn,
        'r' : 1000,
        'n' : syn_n[n]
    }

j = 0
for i in withsyn_i:
    idx_params[j] = idx_params0[i]
    allidx_e[j] = allidx_e0[i]
    j += 1

# Creating some dictionaries for storing
cells = {}
e_synparams = {}
i_synparams = {}
synonset = {}

pop_params.update({'n' : withsyn_i.size})

TASKS = pl.array(pl.linspace(0, pop_params['n']-1, pop_params['n']))

#Initializing proper NEURON simulations
if __name__ == '__main__':
    NUMBER_OF_PROCESSES = cpu_count()
    freeze_support()
    task_queue = Queue()
    done_queue = Queue()

    for task in TASKS:
        task_queue.put(int(task))
    for i in xrange(NUMBER_OF_PROCESSES):
        Process(target=cell_thread,
                args=(task_queue, done_queue)).start()
    for n in xrange(TASKS.size):
        [e_synparams[n], cells[n]] = done_queue.get()
    for i in xrange(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')

    task_queue.close()
    done_queue.close()



if __name__ == '__main__':
    NUMBER_OF_PROCESSES = cpu_count()
    freeze_support()
    task_queue = Queue()
    done_queue = Queue()

    for task in TASKS:
        task_queue.put(int(task))
    for i in xrange(NUMBER_OF_PROCESSES):
        Process(target=CSD_thread,
                args=(task_queue, icsd_diam, done_queue)).start()
    for n in xrange(TASKS.size):
        [cells[n].CSD_diam, cells[n].CSD,
         cells[n].CSD76pt, cells[n].tvec, z_data] = done_queue.get()
    for i in xrange(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')

    task_queue.close()
    done_queue.close()


#calc tru_sphershell_csd
if __name__ == '__main__':
    NUMBER_OF_PROCESSES = cpu_count()
    freeze_support()
    task_queue = Queue()
    done_queue = Queue()

    for task in TASKS:
        task_queue.put(int(task))
    for i in xrange(NUMBER_OF_PROCESSES):
        Process(target=CSDr_thr,
                args=(task_queue, done_queue)).start()
    for n in xrange(TASKS.size):
        [CSDr_r, cells[n].CSDr] = done_queue.get()
    for i in xrange(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')

    task_queue.close()
    done_queue.close()

CSD = cells[0].CSD
CSD_diam = cells[0].CSD_diam
CSD76pt = cells[0].CSD76pt
CSDr = cells[0].CSDr

tvec = cells[0].tvec
totnsynapses = 0
for n in xrange(1, pop_params['n']):
    CSD = CSD + cells[n].CSD
    CSD_diam = CSD_diam + cells[n].CSD_diam
    CSD76pt = CSD76pt + cells[n].CSD76pt
    CSDr = CSDr + cells[n].CSDr
    totnsynapses = totnsynapses + idx_params[n]['n']
print 'total number of synapses is %d.' % totnsynapses


#compute the local field potentials from the two electrode objects
def compute_electrode_LFP(n):
    print 'Electrode %i' % n
    electrode = LFPy.RecExtElectrode(cells[n], **electrodeparams)
    electrode.calc_lfp()
    return electrode.LFP


if __name__ == "__main__":
    print('Pool compute_electrode_LFP:\n')
    pool = Pool(processes=cpu_count())
    electrodeLFP = pool.map(compute_electrode_LFP, range(len(cells.keys())))
    electrodeLFP = pl.array(electrodeLFP).sum(axis=0)



def compute_el_r_LFP(n):
    print 'Electrode %i' % n
    electrode = LFPy.RecExtElectrode(cells[n], **el_r_par)
    electrode.calc_lfp()
    return electrode.LFP


if __name__ == "__main__":
    print('Pool compute_el_r_LFP:\n')
    pool = Pool(processes=cpu_count())
    el_r_LFP = pool.map(compute_el_r_LFP, range(len(cells.keys())))
    el_r_LFP = pl.array(el_r_LFP).sum(axis=0)


signal_params = {
        'dt' : cellparams['timeres_python']
}

#Signal filtering and analysis
signal = tools.Signal(**signal_params)

# Filtering along spatial dimension of LFP and lam. CSD,  lfp+ax
a, b = electrodeLFP.shape
lfp_filtered = pl.zeros((16, b))
CSD_filtered = pl.zeros(CSD.shape)
CSD76ptF = pl.zeros(CSD76pt.shape)
filter_params = {
    'ftype' : 'gaussian',
    'order' : (3, 1)
}
filter_params76pt = {
    'ftype' : 'gaussian',
    'order' : (19, 5)
}

for j in xrange(b):
    lfp_filtered[:, j] = signal.filter_signal(electrodeLFP[:16, j],
                                               **filter_params)
    CSD_filtered[:, j] = signal.filter_signal(CSD[:, j], **filter_params)
    CSD76ptF[:, j] = signal.filter_signal(CSD76pt[:, j], **filter_params76pt)


dataset = {
    'data' : lfp_filtered,
    'setname' : 'z',
    'tvec' : tvec,
    'dt' : cellparams['timeres_python'],
    'labels' : False,
}
analyze_params = {
    'do_fft' : False,
    'norm_output' : False,
    'do_csd' : False,
}
plot_params = {
    'plottype' : 'image',
    'axis' : 'tight',
    'legend' : False,
    'new_fig' : True
}

icsd_input = input_init(lfp_data=electrodeLFP[:16, :]*pq.mV,
                        z_data=pl.linspace(100, 1600, 16)*1E-6*pq.m,
                        diam=icsd_diam*pq.m)
icsd_output = {}
diam_best = {}

my_errors = {}
my_diams = {}
for m in icsd_input:
    if m == 'delta':
        _icsd = icsd.DeltaiCSD(**icsd_input[m])
        icsd_output.update({'icsd_delta' : _icsd.filter_csd(_icsd.get_csd()) / (100E-6*pq.m) * 1E-9})

        my_errors['delta'], my_diams['delta'], diam_best['delta'] = \
            minimize_icsd_error_brute(m)
        icsd_input[m].update({'diam' : diam_best['delta']*pq.m})

        print 'best diameter delta: %.5e' % diam_best['delta']
        _icsd = icsd.DeltaiCSD(**icsd_input[m])
        icsd_output.update({'icsd_delta' : _icsd.filter_csd(_icsd.get_csd()) / (100E-6*pq.m) * 1E-9})        

    elif m == 'step':
        _icsd = icsd.StepiCSD(**icsd_input[m])
        icsd_output.update({'icsd_step' : _icsd.filter_csd(_icsd.get_csd()) * 1E-9})

        my_errors['step'], my_diams['step'], diam_best['step'] = \
            minimize_icsd_error_brute(m)
        icsd_input[m].update({'diam' : diam_best['step']*pq.m})

        print 'best diameter step: %.5e' % diam_best['step']

        _icsd = icsd.StepiCSD(**icsd_input[m])
        icsd_output.update({'icsd_step' : _icsd.filter_csd(_icsd.get_csd()) * 1E-9})

    elif m == 'spline':
        _icsd = icsd.SplineiCSD(**icsd_input[m])
        icsd_output.update({'icsd_spline' : _icsd.filter_csd(_icsd.get_csd()) * 1E-9})

        my_errors['spline'], my_diams['spline'], diam_best['spline'] = \
            minimize_icsd_error_brute(m)
        icsd_input[m].update({'diam' : diam_best['spline']*pq.m})

        print 'best diameter spline: %.5e' % diam_best['spline']

        _icsd = icsd.SplineiCSD(**icsd_input[m])
        icsd_output.update({'icsd_spline' : _icsd.filter_csd(_icsd.get_csd()) * 1E-9})

    elif m == 'std':
        _icsd = icsd.StandardCSD(**icsd_input[m])
        icsd_output.update({'csd_std' : _icsd.filter_csd(_icsd.get_csd()) * 1E-9})

print('Compute correlation coefficients between CSD and estimates')
my_corrcoefs = {}
for m in icsd_input:
    if m == 'delta':
        my_corrcoefs['delta'], _, _ = maximize_icsd_correlation_brute(m)
    elif m == 'step':
        my_corrcoefs['step'], _, _ = maximize_icsd_correlation_brute(m)
    elif m == 'spline':
        my_corrcoefs['spline'], _, _ = maximize_icsd_correlation_brute(m)
print('Compute correlation coefficients between CSD and estimates done')


### Mean EPSP,  synapses.i,  synapses.v with st.dev. ####
syn_i = pl.empty((sum(syn_n), cells[0].tvec.size))
syn_v = pl.empty((sum(syn_n), cells[0].tvec.size))
somav = pl.empty((pop_params['n'], cells[0].tvec.size))
syn_count = 0

for n in xrange(pop_params['n']):
    somav[n, ] = cells[n].somav
    for i in xrange(len(cells[n].synapses)):
        syn_i[syn_count, ] = cells[n].synapses[i].i
        syn_v[syn_count, ] = cells[n].synapses[i].v
        syn_count += 1

mean_EPSP = somav.mean(axis=0)
SD_EPSP = somav.std(axis=0)
RT_EPSP = risetime(tvec, mean_EPSP, limit=[0.1, 0.9])

mean_syn_i = syn_i.mean(axis=0)
SD_syn_i = syn_i.std(axis=0)
RT_syn_i = risetime(tvec, -mean_syn_i, limit=[0.1, 0.9])

mean_syn_v = syn_v.mean(axis=0)
SD_syn_v = syn_v.std(axis=0)
RT_syn_v = risetime(tvec, mean_syn_v, limit=[0.1, 0.9])

# Least square fitting of mean_EPSP,  mean_syn_i,  mean_syn_v, LFP to
# double exponential curve
v_EPSP = fit_alpha(v_EPSP_measured, tvec, mean_EPSP)
LS_EPSP = fp(v_EPSP, tvec)
R2_EPSP = R2(mean_EPSP, LS_EPSP)

v0 = [1.5, 0.5, 1, 0, mean_syn_i.min()]
v_syn_i = fit_alpha(v0, tvec, mean_syn_i)
LS_syn_i = fp(v_syn_i, tvec)
R2_syn_i = R2(mean_syn_i, LS_syn_i)

v0 = [1.5, 0.5, 1, v_init, 8]
v_syn_v = fit_alpha(v0, tvec, mean_syn_v)
LS_syn_v = fp(v_syn_v, tvec)
R2_syn_v = R2(mean_syn_v, LS_syn_v)

v0 = [1.5, 1, 4, 0, -0.025]
v_LFP = fit_alpha(v0, tvec, lfp_filtered[7, :])
LS_LFP = fp(v_LFP, tvec)
R2_LFP = R2(lfp_filtered[7, :], LS_LFP)

######### SAVING SIM-RESULTS
savestuff = {
    'tvec': tvec,
    'lfp_filtered' : lfp_filtered,
    'el_sd' : electrodeLFP[:16, :],
    'el_100' : electrodeLFP[16:32, :],    
    'el_200' : electrodeLFP[32:48, :],   
    'el_300' : electrodeLFP[48:64, :],    
    'el_400' : electrodeLFP[64:80, :],    
    'el_500' : electrodeLFP[80:96, :],    
    'el_r' : el_r_LFP,
    'el_r_x' : el_r_par['x'],
    'CSD' : CSD,
    'CSD_filtered': CSD_filtered,
    'CSD_diam' : CSD_diam,
    'CSDr' : CSDr,
    'CSDr_r' : CSDr_r,
    'diam_best_delta' : diam_best['delta'],
    'diam_best_step' : diam_best['step'],
    'diam_best_spline' : diam_best['spline'],
    # add errors and diameters for all tested values:
    'my_diams_delta' : my_diams['delta'],
    'my_diams_step' : my_diams['step'],
    'my_diams_spline' : my_diams['spline'],
    'my_errors_delta' : my_errors['delta'],
    'my_errors_step' : my_errors['step'],
    'my_errors_spline' : my_errors['spline'],
    # add correlation coefficients
    'my_corrcoefs_delta' : my_corrcoefs['delta'],
    'my_corrcoefs_step' : my_corrcoefs['step'],
    'my_corrcoefs_spline' : my_corrcoefs['spline'],
    #
    'somav' : somav,
    'syn_i' : syn_i,
    'syn_v' : syn_v,
    'syn_n' : pl.array(syn_n)[withsyn_i],
    'mean_EPSP' : mean_EPSP,
    'SD_EPSP' : SD_EPSP,
    'LS_EPSP' : LS_EPSP,
    'R2_EPSP' : R2_EPSP,
    'v_EPSP' : v_EPSP,
    'mean_syn_i' : mean_syn_i,
    'SD_syn_i' : SD_syn_i,
    'LS_syn_i' : LS_syn_i,
    'R2_syn_i' : R2_syn_i,
    'v_syn_i' : v_syn_i,
    'mean_syn_v' : mean_syn_v,
    'SD_syn_v' : SD_syn_v,
    'LS_syn_v' : LS_syn_v,
    'R2_syn_v' : R2_syn_v,
    'v_syn_v' : v_syn_v,
    'LS_LFP' : LS_LFP,
    'R2_LFP' : R2_LFP,
    'v_LFP' : v_LFP,
    'icsd_delta' : icsd_output['icsd_delta'],
    'icsd_step' : icsd_output['icsd_step'],
    'icsd_spline' : icsd_output['icsd_spline'],
    'csd_std' : icsd_output['csd_std'],
    'CSD76pt' : CSD76pt,
    'CSD76ptF' : CSD76ptF,
    'som_pos0_x' : pop_soma_pos0['xpos'],
    'som_pos0_y' : pop_soma_pos0['ypos'],
    'som_pos0_z' : pop_soma_pos0['zpos'],
    'som_pos_x' : som_pos_x,
    'som_pos_y' : som_pos_y,
    'som_pos_z' : som_pos_z,
    'syn_pos_x' : syn_pos_x,
    'syn_pos_y' : syn_pos_y,
    'syn_pos_z' : syn_pos_z,
}
################################################################################

c_saved = {}
if len(cells.keys()) < 50:
    for n in xrange(len(cells.keys())):
        c_saved[n] = cells[n]
        del c_saved[n].imem
        del c_saved[n].somav
else:
    for n in xrange(50):
        c_saved[n] = cells[n]
        del c_saved[n].imem
        del c_saved[n].somav

cells = {}
if __name__ == '__main__':
    freeze_support()
    NUMBER_OF_PROCESSES = cpu_count() #leaving one core to handle threads
    task_queue = Queue()
    done_queue = Queue()

    for task in TASKS:
        task_queue.put(int(task))
    for i in xrange(NUMBER_OF_PROCESSES):
        Process(target=cell_thread_PSC,
                args=(task_queue, done_queue)).start()
    for n in xrange(TASKS.size):
        cells[n] = done_queue.get()
    for i in xrange(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')

    task_queue.close()
    done_queue.close()

### Mean EPSC with standard deviation (regardles of syn. n.):
EPSC = pl.empty((pop_params['n'], cells[0].tvec.size))
for n in xrange(pop_params['n']):
    for i in xrange(len(cells[n].pointprocesses)):
        EPSC[n, ] = cells[n].pointprocesses[i].i
mean_EPSC = EPSC.mean(axis=0)
SD_EPSC = EPSC.std(axis=0)
RT_EPSC = risetime(tvec, -mean_EPSC, limit=[0.1, 0.9])

###### Least square fitting of mean_EPSC
v0 = [1.5, 0.1, 1.0, 0., mean_EPSC.min()]
v_EPSC = fit_alpha(v0, tvec, mean_EPSC)
LS_EPSC = fp(v_EPSC, tvec)
R2_EPSC = R2(mean_EPSC, LS_EPSC)


PSet.update({'simulated' : True})
PSet.save(url=os.path.join(datafolder, '{}.pset'.format(PSet['uuid'])))

######## SAVING STUFF ########
savestuff.update(**{'EPSC' : EPSC})
savestuff.update(**{'v_EPSC' : v_EPSC})
savestuff.update(**{'mean_EPSC' : mean_EPSC})
savestuff.update(**{'SD_EPSC' : SD_EPSC})
savestuff.update(**{'LS_EPSC' : LS_EPSC})
savestuff.update(**{'R2_EPSC' : R2_EPSC})

f = h5py.File(os.path.join(datafolder, 'simres.h5'), 'w')
for i in xrange(len(savestuff.keys())):
    dset = f.create_dataset(savestuff.keys()[i],
                            pl.shape(savestuff.values()[i]))
    dset[...] = savestuff[savestuff.keys()[i]]

f.close()

f = file('results_tau.txt', 'a')
f.write('%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\n' % (psetid, morphology,
                    v_EPSP[1], v_EPSP[2], v_EPSC[1], v_EPSC[2]))
f.close()


from initialize_simulations import get_md5s
if PSet['uuid'] in get_md5s('PS_simres_RS') + get_md5s('PS_simres_FS') + get_md5s('PS_simres_P4'):
# 
# if P_i <= 1:
    f = file(os.path.join(datafolder, 'c_savedPickle.cpickle'), 'wb')
    cPickle.dump(c_saved, f)
    f.close()

print('Script completed in %i seconds') % int(time() - tstart)
