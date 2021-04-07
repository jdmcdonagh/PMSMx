# PMSM-X

PMSM-X is an electromagnetic model of a permanent magnet synchronous motor built within the finite element platform [FEniCS-X](https://fenicsproject.org) using the Python interface. 

This model constituted my BEng dissertation titled "Development of a 3D Low-Frequency Electrical Machine within the Finite Element Platform FEniCS-X for Exascale Multi-Physics Simulations".

## Formulation

The model uses the A-V formulation in the time domain. Simplifications have been made such as linear magnetic behaviour and rotation modelled through a motion voltage term. The complete strong form is shown below: 

![equation](https://latex.codecogs.com/gif.latex?%5Cnu%5Cmathrm%7B%5Cnabla%7D%5E2A%3D%7B-J%7D_s&plus;%5Csigma%5Cfrac%7B%5Cpartial%20A%7D%7B%5Cpartial%20t%7D&plus;%5Csigma%5Cnabla%20V-%5Cmathrm%7B%5Cnabla%7D%5Ctimes%5Cleft%28%5Cnu%5Cmu_0M%5Cright%29-%7B%5Csigma%5Comega%7D_%5Cupsilon%20r%5Ctimes%28%5Cmathrm%7B%5Cnabla%7D%5Ctimes%20A%29)

## Meshes

Four meshes were created within ANSYS Maxwell 3D and converted into the [HDF5 file format](https://www.hdfgroup.org/solutions/hdf5/) using h5py. The coarsest mesh comprised 125,478 elements while the finest mesh comprised 8,356,409 elements. The largest mesh could not be stored due to the large file size.

<img src="/img/Mesh1.png" alt="Mesh1" width="45%"/><img style="float:right; !important" src="/img/Mesh4.png" alt="Mesh1" width="45%"/>