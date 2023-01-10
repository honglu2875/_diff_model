#!/bin/bash

conda install python=3.10 git tmux
conda install -c conda-forge openmpi-mpicc mpi4py
ln -s $CONDA_PREFIX/lib $CONDA_PREFIX/lib64

