#!/bin/bash

conda install python=3.10 git tmux
conda install -c conda-forge openmpi-mpicc mpi4py
conda install -c conda-forge gxx_linux-64
ln -s $CONDA_PREFIX/lib $CONDA_PREFIX/lib64

