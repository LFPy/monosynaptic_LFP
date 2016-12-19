Files and data used for the study:
==================================

Hagen E, Fossum JC, Pettersen KH, Alonso JM, Swadlow HA, Einevoll GT.
"Focal local field potential (LFP) signature of the single-axon monosynaptic
thalamocortical connection" (submitted)


Files:
======

initialize_simulations.py:
    Defines parameters, initializes simulations, creates jobscripts, submits to
    cluster queue using the Slurm job manager. Adapt accordingly.
    
lfp_simulation.py:
    Main simulation procedure used for the LFP predictions

create_figures.py:
    Produce figures from simulation output

tools.py:
    Auxiliary function declarations and classes used by the above scripts

morphologies/*.hoc:
    morphology files converted to NEURON's HOC language

add_spines.hoc:
    HOC language file for modifying surface area dependent on spine density

data/*:
    Files with the experimental stLFPs from previous studies by
    Swadlow et al. (2002) and Stoelzel et al. (2008)

README.txt:
    This file
    

Workflow:
=========

In order to generate all simulated data, the workflow is as follows:

-   put all files on a compute cluster ideally running the Slurm job manager.
    If the cluster is using a different setup, adapt initialize_simulations.py
    accordingly. Make sure that all Python dependencies are met, installing
    various packages from Python package index for example
    (pip install <package-name> --user)
-   run initialize_simulations.py:
        $ python initialize_simulations.py
    
    The script will delete all simulation output and corresponding files by
    default, create different parameter files with different md5 hashes
    in the subfolder "parameters" using the Neurotools.parameterspace module,
    jobscripts will be put in the folder "jobs". The script will also send all
    jobs to the queue before terminating. The folder "logs" should when jobs
    start contain various output for each job. The folder "savedata" should
    contain simulated data for each parametercombination.
-   Make sure no jobs are running with "squeue -u $USER"
-   Run the script create_figures.py to produce all figures
        $ python create_figures.py
    pdf images will be stored in this folder.
    

License:
========

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
