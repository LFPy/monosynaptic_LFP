#!/usr/bin/env python
'''
Initialize simulations, create jobscripts, submit to the cluster queue
(using Slurm) from the login node. Adapt to Your cluster accordingly

Usage:

    python initialize_simulations.py


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
import pylab as pl
from NeuroTools.parameters import ParameterSpace, ParameterRange
import os
from hashlib import md5
import pickle
import pprint
import time
from glob import glob
#pl.seed(1234)


##### FUNCTION DECLARATIONS ########

def check_all_crit(param, search_keys, search_values):
    if (len(search_keys)>0):
        if (param[search_keys[0]] == search_values[0]):
            check = check_all_crit(param,\
            search_keys[1::],\
            search_values[1::])
            return check
        else:
            check = False
            return check
    elif (len(search_keys) == 0):
        check = True
        return check

def search_param_list(param_list, search_dict):
    search_keys = search_dict.keys()
    search_values = search_dict.values()
    idx_list = []
    uuid_list = []
    for i_param in range(len(param_list)):
        if check_all_crit(param_list[i_param], search_keys, search_values):
            idx_list.append(i_param)
            uuid_list.append(param_list[i_param].uuid)

    print "Found", len(idx_list), "simulation(s) matching your criteria"
    return idx_list, uuid_list


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

################################################################################
# Main simulation parameterspaces
################################################################################

PS = {}

PS['simres_RS'] = ParameterSpace({
    'morphology' : os.path.join('morphologies', 'C120398A-P1.CNG_sansAxon.hoc'),      #L4 Stellate Cell, rat
    'rm' : ParameterRange([11250]),
    'Ra' : ParameterRange([150]),
    'cm' : ParameterRange([0.9]),
    'custom_code' : ParameterRange([['add_spines.hoc']]),
    'active' : ParameterRange([False]),
    'v_init' : ParameterRange([-66.]), #Beierlein

    'pop_params_n' : ParameterRange([4000]),
    'pop_geom' : ParameterRange([[500, -250, 250]]),

    'tau1' : ParameterRange([0.2]),
    'tau2' : ParameterRange([2.0]), #was 2, [1.75]),
    'weight' : ParameterRange([0.0004]),
    'e' : ParameterRange([0]),
    'spiketime' : ParameterRange([2.4]),

    'disttype' : ParameterRange(['hard_sphere']), #type of synaptic distribution probability

    'sigma' : ParameterRange([[165]]), #spread of distribution
    'my' : ParameterRange([35]),        #offset
    'mean_n_syn' : ParameterRange([7]),

    'section_syn' : ParameterRange([['apic', 'dend']]),

    'icsd_diam' : ParameterRange([400E-6]),

    'electrode_r' : ParameterRange([15]),

    'simulated' : False,

    'randomseed' : ParameterRange([474472279, 3495581941, 2563836960,  400995787, 3077891106, 123456789]),
})


PS['simres_P4'] = ParameterSpace({
    'morphology' : ParameterRange([
        os.path.join('morphologies', 'C060998B-P4.CNG_sansAxon.hoc'), # PC
        os.path.join('morphologies', 'C150897B-P2.CNG_sansAxon.hoc'), # PC
        os.path.join('morphologies', 'C231296A-P4B2.CNG_sansAxon.hoc'), # PC
        os.path.join('morphologies', 'FLUO9-RIGHT.CNG_sansAxon.hoc'), # PC
        os.path.join('morphologies', 'VD100726B-IDB.CNG_sansAxon.hoc'), # PC
        ]),      #L4 pyramidal Cell, rat
    'rm' : ParameterRange([11250]),
    'Ra' : ParameterRange([150]),
    'cm' : ParameterRange([0.9]),
    'custom_code' : ParameterRange([['add_spines.hoc']]),
    'active' : ParameterRange([False]),
    'v_init' : ParameterRange([-66.]), #Beierlein

    'pop_params_n' : ParameterRange([4000]),
    'pop_geom' : ParameterRange([[500, -250, 250]]),

    'tau1' : ParameterRange([0.2]),
    'tau2' : ParameterRange([2.0]), #was 2, [1.75]),
    'weight' : ParameterRange([0.0004]),
    'e' : ParameterRange([0]),
    'spiketime' : ParameterRange([2.4]),

    'disttype' : ParameterRange(['hard_sphere']), #type of synaptic distribution probability

    'sigma' : ParameterRange([[165]]), #spread of distribution
    'my' : ParameterRange([35]),        #offset
    'mean_n_syn' : ParameterRange([7]),

    'section_syn' : ParameterRange([['apic', 'dend']]),

    'icsd_diam' : ParameterRange([400E-6]),

    'electrode_r' : ParameterRange([15]),

    'simulated' : False,

    'randomseed' : 123456789,
})


PS['simres_FS'] = ParameterSpace({
    'morphology' : os.path.join('morphologies', 'C120398A-I4.CNG_sansAxon.hoc'), # L4 Basket cell, rat
    'rm' : ParameterRange([5625]),
    'Ra' : ParameterRange([150]),
    'cm' : ParameterRange([0.9]),
    'custom_code' : ParameterRange([[]]),
    'active' : ParameterRange([False]),
    'v_init' : ParameterRange([-64.0]), #beierlein

    'pop_params_n' : ParameterRange([1000]),
    'pop_geom' : ParameterRange([[500, -250, 250]]),

    'tau1' : ParameterRange([0.05]), #was 0.09
    'tau2' : ParameterRange([0.20]), #was 0.45
    'weight' : ParameterRange([0.00175,]),
    'e' : ParameterRange([0]),
    'spiketime' : ParameterRange([2.4]),

    'disttype' : ParameterRange(['hard_sphere']), #type of synaptic distribution probability
    'sigma' : ParameterRange([[165]]), #spread of distribution
    'my' : ParameterRange([0]), #offset
    'mean_n_syn' : ParameterRange([15]),

    'section_syn' : ParameterRange(['somaproximal']),

    'icsd_diam' : ParameterRange([400E-6]),
    
    'electrode_r' : ParameterRange([15]),

    'simulated' : False,

    'randomseed' : ParameterRange([1817008249, 1862380098, 4050593744,  885958766, 1612109401, 123456789]),
})


PS['simres_RXXX'] = ParameterSpace({
    'morphology' : os.path.join('morphologies', 'C120398A-P1.CNG_sansAxon.hoc'),      #L4 Stellate Cell
    'rm' : ParameterRange([11250]),
    'Ra' : ParameterRange([150]),
    'cm' : ParameterRange([0.9]),
    'custom_code' : ParameterRange([['add_spines.hoc']]),
    'active' : ParameterRange([False]),
    'v_init' : ParameterRange([-66.]),

    'pop_params_n' : ParameterRange([4000]),
    'pop_geom' : ParameterRange([[500, -250, 250]]),

    'tau1' : ParameterRange([0.2]),
    'tau2' : ParameterRange([2.0]), #was 2, [1.75]),
    'weight' : ParameterRange([0.0004]),
    'e' : ParameterRange([0]),
    'spiketime' : ParameterRange([1]),

    'disttype' : ParameterRange(['hard_cyl']), #type of synaptic distribution
    'sigma' : ParameterRange([
        [100, 50],
        [100, 100],
        [100, 200],
        [100, 300],
        [100, 400],
        # [100, 500] # 
        ]), #spread of distrb.
    'my' : ParameterRange([0]),        #offset
    'mean_n_syn' : ParameterRange([7]),

    'section_syn' : ParameterRange([['apic', 'dend']]),

    'icsd_diam' : ParameterRange([400E-6]),

    'electrode_r' : ParameterRange([15]),

    'simulated' : False,

    'randomseed' : 123456789,
})


PS['simres_Gaussian'] = ParameterSpace({
    'morphology' : os.path.join('morphologies', 'C120398A-P1.CNG_sansAxon.hoc'),      #L4 Stellate Cell, rat
    'rm' : ParameterRange([11250]),
    'Ra' : ParameterRange([150]),
    'cm' : ParameterRange([0.9]),
    'custom_code' : ParameterRange([['add_spines.hoc']]),
    'active' : ParameterRange([False]),
    'v_init' : ParameterRange([-66.]),

    'pop_params_n' : ParameterRange([4000]),
    'pop_geom' : ParameterRange([[500, -250, 250]]),

    'tau1' : ParameterRange([0.2]),
    'tau2' : ParameterRange([2.0]), #was 2, [1.75]),
    'weight' : ParameterRange([0.0004]),
    'e' : ParameterRange([0]),
    'spiketime' : ParameterRange([1]),

    'disttype' : ParameterRange(['anisotrop']), #type of synaptic distribution
    'sigma' : ParameterRange([[100, 100, 100]]), #spread of distrb.
    'my' : ParameterRange([0]), #offset
    'mean_n_syn' : ParameterRange([7]),

    'section_syn' : ParameterRange([['apic', 'dend']]),

    'electrode_r' : ParameterRange([15]),

    'icsd_diam' : ParameterRange([400E-6]),

    'simulated' : False,

    'randomseed' : 123456789,
})


PS['simres_Cylindrical'] = ParameterSpace({
    'morphology' : os.path.join('morphologies', 'C120398A-P1.CNG_sansAxon.hoc'),      #L4 Stellate Cell, rat
    'rm' : ParameterRange([11250]),
    'Ra' : ParameterRange([150]),
    'cm' : ParameterRange([0.9]),
    'custom_code' : ParameterRange([['add_spines.hoc']]),
    'active' : ParameterRange([False]),
    'v_init' : ParameterRange([-66.]),

    'pop_params_n' : ParameterRange([4000]),
    'pop_geom' : ParameterRange([[500, -250, 250]]),

    'tau1' : ParameterRange([0.2]),
    'tau2' : ParameterRange([2.0]), #was 2, [1.75]),
    'weight' : ParameterRange([0.0004]),
    'e' : ParameterRange([0]),
    'spiketime' : ParameterRange([1]),

    'disttype' : ParameterRange(['hard_cyl']), #type of synaptic distribution
    'sigma' : ParameterRange([[100,  200]]), #spread of distrb.
    'my' : ParameterRange([0]), #offset
    'mean_n_syn' : ParameterRange([7]),

    'section_syn' : ParameterRange([['apic', 'dend']]),

    'electrode_r' : ParameterRange([15]),

    'icsd_diam' : ParameterRange([400E-6]),

    'simulated' : False,

    'randomseed' : 123456789,
})


PS['simres_Spherical'] = ParameterSpace({
    'morphology' : os.path.join('morphologies', 'C120398A-P1.CNG_sansAxon.hoc'),      #L4 Stellate Cell, rat
    'rm' : ParameterRange([11250]),
    'Ra' : ParameterRange([150]),
    'cm' : ParameterRange([0.9]),
    'custom_code' : ParameterRange([['add_spines.hoc']]),
    'active' : ParameterRange([False]),
    'v_init' : ParameterRange([-66.]),

    'pop_params_n' : ParameterRange([4000]),
    'pop_geom' : ParameterRange([[500, -250, 250]]),

    'tau1' : ParameterRange([0.2]),
    'tau2' : ParameterRange([2.0]), #was 2, [1.75]),
    'weight' : ParameterRange([0.0004]),
    'e' : ParameterRange([0]),
    'spiketime' : ParameterRange([1]),

    'disttype' : ParameterRange(['hard_sphere']), #type of synaptic distribution
    'sigma' : ParameterRange([200]), #spread of distrb.
    'my' : ParameterRange([0]), #offset
    'mean_n_syn' : ParameterRange([7]),

    'section_syn' : ParameterRange([['apic', 'dend']]),

    'electrode_r' : ParameterRange([15]),

    'icsd_diam' : ParameterRange([400E-6]),

    'simulated' : False,

    'randomseed' : 123456789,
})


PS['simres_RS_seed'] = ParameterSpace({
    'morphology' : os.path.join('morphologies', 'C120398A-P1.CNG_sansAxon.hoc'),      #L4 Stellate Cell, rat
    'rm' : ParameterRange([11250]),
    'Ra' : ParameterRange([150]),
    'cm' : ParameterRange([0.9]),
    'custom_code' : ParameterRange([['add_spines.hoc']]),
    'active' : ParameterRange([False]),
    'v_init' : ParameterRange([-66.]), #Beierlein

    'pop_params_n' : ParameterRange([4000]),
    'pop_geom' : ParameterRange([[500, -250, 250]]),

    'tau1' : ParameterRange([0.2]),
    'tau2' : ParameterRange([2.0]), #was 2, [1.75]),
    'weight' : ParameterRange([0.0004]),
    'e' : ParameterRange([0]),
    'spiketime' : ParameterRange([2.4]),

    'disttype' : ParameterRange(['hard_sphere']), #type of synaptic distribution probability

    'sigma' : ParameterRange([[165]]), #spread of distribution
    'my' : ParameterRange([35]),        #offset
    'mean_n_syn' : ParameterRange([7]),

    'section_syn' : ParameterRange([['apic', 'dend']]),

    'icsd_diam' : ParameterRange([400E-6]),

    'electrode_r' : ParameterRange([15]),

    'simulated' : False,

    'randomseed' : ParameterRange([3910340497, 1164141840, 2224105389,
                                   3706981294, 2204067231,
                                   2782494339, 1603452434, 2825923326,
                                   1328781694, 2662348289]),
})


PS['simres_FS_seed'] = ParameterSpace({
    'morphology' : os.path.join('morphologies', 'C120398A-I4.CNG_sansAxon.hoc'), # L4 Basket cell, rat
    'rm' : ParameterRange([5625]),
    'Ra' : ParameterRange([150]),
    'cm' : ParameterRange([0.9]),
    'custom_code' : ParameterRange([[]]),
    'active' : ParameterRange([False]),
    'v_init' : ParameterRange([-64.0]), #beierlein

    'pop_params_n' : ParameterRange([1000]),
    'pop_geom' : ParameterRange([[500, -250, 250]]),

    'tau1' : ParameterRange([0.05]), #was 0.09
    'tau2' : ParameterRange([0.20]), #was 0.45
    'weight' : ParameterRange([0.00175,]),
    'e' : ParameterRange([0]),
    'spiketime' : ParameterRange([2.4]),

    'disttype' : ParameterRange(['hard_sphere']), #type of synaptic distribution probability
    'sigma' : ParameterRange([[165]]), #spread of distribution
    'my' : ParameterRange([0]), #offset
    'mean_n_syn' : ParameterRange([15]),

    'section_syn' : ParameterRange(['somaproximal']),

    'icsd_diam' : ParameterRange([400E-6]),
    
    'electrode_r' : ParameterRange([15]),

    'simulated' : False,

    'randomseed' : ParameterRange([2091789212, 2727741963, 1863543399,
                                   1560910521, 1221154565,
                                   1688305177, 2906496959, 3316370564,
                                   3178327809, 3233636196]),
})


PS['simres_FS_seed_50'] = ParameterSpace({
    'morphology' : os.path.join('morphologies', 'C120398A-I4.CNG_sansAxon.hoc'), # L4 Basket cell, rat
    'rm' : ParameterRange([5625]),
    'Ra' : ParameterRange([150]),
    'cm' : ParameterRange([0.9]),
    'custom_code' : ParameterRange([[]]),
    'active' : ParameterRange([False]),
    'v_init' : ParameterRange([-64.0]), #beierlein

    'pop_params_n' : ParameterRange([500]),
    'pop_geom' : ParameterRange([[500, -250, 250]]),

    'tau1' : ParameterRange([0.05]), #was 0.09
    'tau2' : ParameterRange([0.20]), #was 0.45
    'weight' : ParameterRange([0.00175,]),
    'e' : ParameterRange([0]),
    'spiketime' : ParameterRange([2.4]),

    'disttype' : ParameterRange(['hard_sphere']), #type of synaptic distribution probability
    'sigma' : ParameterRange([[165]]), #spread of distribution
    'my' : ParameterRange([0]), #offset
    'mean_n_syn' : ParameterRange([15]),

    'section_syn' : ParameterRange(['somaproximal']),

    'icsd_diam' : ParameterRange([400E-6]),
    
    'electrode_r' : ParameterRange([15]),

    'simulated' : False,

    'randomseed' : ParameterRange([2091789212, 2727741963, 1863543399,
                                   1560910521, 1221154565,
                                   1688305177, 2906496959, 3316370564,
                                   3178327809, 3233636196]),
})


def get_md5(PSet):
    '''get md5 hash of parameterset'''
    try:
        assert 'uuid' not in PSet.keys()
    except AssertionError as ae:
        raise ae, 'uuid already set.'
    P = PSet.items()
    P.sort()
    m = md5()
    m.update(pickle.dumps(P))
    return m.hexdigest()    


def get_md5s(PS_prefix):
    '''return list of md5 hash keys from parameterspace prefix'''
    md5s = []
    PS = ParameterSpace(os.path.join('parameters', PS_prefix+'.pspace'))
    for i, PSet in enumerate(PS.iter_inner()):
        id = get_md5(PSet)
        md5s += [id]
    return md5s


###### Executed code ###########################################################
if __name__ == '__main__':
    
    ############################################################################
    # REMOVING OLD SIMULATION RESULTS AND JOBSCRIPT FILES, SET UP NEW FILES
    # (adapt accordingly for the computing facilities in use)
    ############################################################################
    # if not os.path.isdir('savedata'):
    #     os.mkdir('savedata')
    # if not os.path.isdir('parameters'):
    #     os.mkdir('parameters')
    # delete_files = ['savedata/*', 'parameters/*', 'results_tau.txt']
    # for f in delete_files:
    #     os.system('rm -r {}'.format(f))
    # 
    # # Delay execution
    # time.sleep(2)
    
    
    # Create new parameter files:
    PSet_ids = []
    PSet_list = []
    keys = pl.sort(PS.keys())
    for i in keys:
        for PSet in PS[i].iter_inner():
            # create unique hash from PSet, convert dict to sorted list
            id = get_md5(PSet)
            if id not in PSet_ids: # avoid duplicate jobs with identical params
                PSet_ids.append(id)
                PSet['uuid'] = id
                PSet_list.append(PSet)
                url = 'parameters/{}.pset'.format(id)
                PSet.save(url)
        url = 'parameters/PS_{}.pspace'.format(i)
        PS[i].save(url)
    
    
    lwalltime = []
    for i in range(len(PSet_list)):
        lwalltime.append('{:02d}:{:02d}:{:02d}'.format(2, 0, 0))
    pyrunscript = 'lfp_simulation.py'    
    
    jobscript = '''#!/bin/bash
##################################################################
#SBATCH --job-name monosynapseLFP
#SBATCH --time {}
#SBATCH -o logs/{}.txt
#SBATCH -e logs/{}.txt
#SBATCH -N 1    # using all cores via multiprocessing
#SBATCH --mem-per-cpu=16GB
#SBATCH --ntasks=1
#SBATCH --exclusive
##################################################################
unset DISPLAY # DISPLAY somehow problematic with Slurm
python {} {}
'''
    
    for directory, fending in zip(['jobs', 'logs'], ['job', 'txt']):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        else:
            os.system('rm {}'.format(os.path.join(directory,
                                                  '*.{}'.format(fending))))
    
    for id in PSet_ids:
        #write job scripts
        f = file(os.path.join('jobs', '{}.job'.format(id)), 'w')
        f.write(jobscript.format(lwalltime[i], id, id, pyrunscript, os.path.join('parameters', id+'.pset')))
        f.close()
    
    
    f = file('results_tau.txt', 'w')
    f.write('UUID\tmorphology\tEPSP\t\tEPSC\n')
    f.close()
    
    search_dict = {}
    print('search criteria: %s' % search_dict)
    [idx_list, uuid_list] = search_param_list(PSet_list,search_dict)
    print idx_list, uuid_list
    
    if which('sbatch') is not None:
        for id in PSet_ids[::-1]:
            os.system('sbatch {}'.format(os.path.join('jobs', '{}.job'.format(id))))
    