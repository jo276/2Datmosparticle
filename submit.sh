#!/bin/bash

#PBS -N 2DHaze
#PBS -l walltime=10:00:00
#PBS -l nodes=1:ppn=36
#PBS -m n
#PBS -q dirac25x
#PBS -A dp100

cd $PBS_O_WORKDIR

python runner_use.py