

SHEIC inversion code. Not for public use. Adapted from Kalle's AMPS inversion code.

There are two parts to this code. In the data_preparation folder, there are scripts that are used to produce a static datafile which is later used as input in the inversion code. This is in the data_preparation folder. The inversion code is included in the base directory. 

The data preparation files do the following:
# 01_make_swarm_CT2Hz_hdfs.py          : Open up all the .cdf.zip files and put their contents, along with some stuff in Modified Apex-110 coordinates, into an HDF file.
# 02_f107_download_and_filter.py       : Add F10.7
# 03_omni_download_1min_data.py        : Download OMNI data (IMF components, solar wind speed and density, SYM-H, + others?
# 04_omni_process_1min_data.py         : Process OMNI data (calculate IMF clock angle mean,variance, average over 30-min window, etc.)
# 05_add_omni_f107_dptilt_substorms.py : Sort index of each column in HDF file, add F10.7, OMNI, dipole tilt, B0_IGRF, og the rest to an HDF file
# 06_add_crosstrack_vector_info.py     : Calculate cross-track convection in MA-110 coordinates, add these to HDF:
#                                        ['Viy_d1','Viy_d2',
#                                         'Viy_f1','Viy_f2',
#                                         'yhat_d1', 'yhat_d2',
#                                         'yhat_f1', 'yhat_f2',
#                                         'gdlat', 'alt']
# 07_make_model_dataset.py             : Read HDF store files, calculate all the weights, (optionally) retain only measurements with a particular quality flag,
#                                        and then store weights, coordinates, and measurements in  a format that can be streamed using dask

Running the scripts in numerical order should produce a static datafile which is the only input to the inversion code. The latest such datafile can be found on Spencer's laptop: "modeldata_v2_update.hdf5". This includes data from Swarm A and Swarm B, through August 11, 2021.

The FINAL inversion code is "hdl_model_iteration__Lowes1966_regularization.py"

Data preparation script dependencies
===================================
Publicly available:
- cdflib
- numpy
- pandas
- requests
- urllib
- glob

Private stuff (obtainable from me, Spencer)
- hatch_python_utils
- swarmProcHelper.py

Inversion code dependencies (apart from standard library things):
- numpy
- dask
- h5py
- scipy.linalg
