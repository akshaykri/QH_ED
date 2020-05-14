# QH_ED
many-body Quantum Hall code

QH_ED1.ipynb:
  - jupyter notebook with demo of features

Python files:
- hilbert.py:
  - numba jitted functions for implementing matrix-vector products
  
- hilbertnoNumba.py:
  - same as above, but bare python

- landau.py:
  - class Torus contains information about geometry
  - class Potential contains information about interaction
 
- runSim.py:
  - script to be called by Slurm with various command line parameters
  
- utils.py:
  - various utility functions
