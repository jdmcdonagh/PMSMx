
# PMSMx

PMSMx is an electromagnetic model of a permanent magnet synchronous motor built within the finite element platform [FEniCSx](https://fenicsproject.org) using the Python interface.

This model constituted my dissertation titled "Development of a 3D Low-Frequency Electrical Machine within the Finite Element Platform FEniCSx for Exascale Multi-Physics Simulations".

## Publication

Further development of this model led to a [publication](https://doi.org/10.1016/j.finel.2022.103755) titled "Modelling a permanent magnet synchronous motor in FEniCSx for parallel high-performance simulations"

### Abstract
There are concerns that the extreme requirements of heavy-duty vehicles and aviation will see them left behind in the electrification of the transport sector, becoming the most significant emitters of greenhouse gases. Engineers extensively use the  [finite element method](https://www.sciencedirect.com/topics/engineering/finite-element-method "Learn more about finite element method from ScienceDirect's AI-generated Topic Pages")  to analyse and improve the performance of electric machines, but new highly [scalable methods](https://www.sciencedirect.com/topics/computer-science/scalable-method "Learn more about scalable methods from ScienceDirect's AI-generated Topic Pages") with a linear (or near) time complexity are required to make extreme-scale models viable. This paper introduces a three-dimensional permanent magnet  [synchronous motor](https://www.sciencedirect.com/topics/engineering/synchronous-motor "Learn more about synchronous motor from ScienceDirect's AI-generated Topic Pages")  model using FEniCSx, a finite element platform tailored for efficient computing and data handling at scale. The model demonstrates comparable [magnetic flux density](https://www.sciencedirect.com/topics/engineering/magnetic-flux-density "Learn more about magnetic flux density from ScienceDirect's AI-generated Topic Pages")  distributions to a verification model built in Ansys Maxwell with a maximum deviation of 7% in the motor’s static regions. Solving the largest mesh, comprising over eight million cells, displayed a speedup of 198 at 512 processes. A preconditioned [Krylov subspace method](https://www.sciencedirect.com/topics/computer-science/krylov-subspace-method "Learn more about Krylov subspace method from ScienceDirect's AI-generated Topic Pages") was used to solve the system, requiring 92% less memory than a direct solution. It is expected that advances built on this approach will allow system-level multiphysics simulations to become feasible within electric machine development. This capability could provide the near real-world accuracy needed to bring electric  [propulsion systems](https://www.sciencedirect.com/topics/engineering/propulsion-system "Learn more about propulsion systems from ScienceDirect's AI-generated Topic Pages")  to large vehicles.

### Citation

McDonagh, J., Palumbo, N., Cherukunnath, N., Dimov, N., & Yousif, N. (2022). Modelling a permanent magnet synchronous motor in FEniCSx for parallel high-performance simulations. _Finite Elements in Analysis and Design_, _204_, 103755. doi:10.1016/j.finel.2022.103755
