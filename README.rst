

AMPS inversion code. Not for public use.

There are two parts to this code. In the data_preparation folder, there are scripts that are used to produce a static datafile which is later used as input in the inversion code. This is in the data_preparation folder. The inversion code is included in the base directory. 

The data preparation files download Swarm data using viresclient, calculate CHAOS magnetic field predictions using chaosmagpy, calculate magnetic coordinates and basis vector components, and external parameters such as dipole tilt, IMF etc.. Required input files are CHAMP dataset (which I got from Chris), OMNI dataset, and F10.7 data. Ask me to get access to these. Running the scripts in alphabetical order should produce a static datafile which is the only input to the inversion code. The latest such datafile can be found here: https://www.dropbox.com/s/mm2biktz23yxxt9/modeldata_v1_update.hdf5?dl=0 


Inversion code dependencies (apart from standard library things):
- numpy
- dask
- h5py
- scipy