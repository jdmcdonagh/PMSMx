from meshinfo import PmsmMeshValues
from excitations import SupplyCurrentDensity, PMMagnetization

import os, math, ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from ufl import (FiniteElement, VectorElement, MixedElement, 
    TestFunction, TrialFunction, TestFunctions, TrialFunctions,
    SpatialCoordinate, grad, cross, curl, inner, system, 
    as_vector, FacetNormal, dot, sqrt, avg)
from dolfinx import (DirichletBC, Function, FunctionSpace, log,
    TimingType, list_timings)
from dolfinx.common import Timer
from dolfinx.cpp.mesh import GhostMode
from dolfinx.io import XDMFFile
from dolfinx.fem import (assemble_matrix, assemble_vector,
    assemble_scalar, apply_lifting, set_bc, locate_dofs_topological)

t_run = Timer("00 Overall Run Time")

# Log output options
# loglevel = log.LogLevel.INFO
loglevel = log.LogLevel.WARNING
# loglevel = log.LogLevel.ERROR
log.set_log_level(loglevel)
log.set_output_file("output.txt")


### Helper Methods #######################################################

def SumFunctions(fncspace, name, A, B):
    fnc = Function(fncspace)
    fnc.name = name
    fnc.vector.set(0.0)
    fnc.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, 
                           mode=PETSc.ScatterMode.FORWARD)
    fnc.vector.axpy(1, A.vector)
    fnc.vector.axpy(1, B.vector)
    return fnc

def AssembleSystem(a, L, bcs, name):
    if mesh.mpi_comm().rank == 0: 
        log.log(loglevel, name + ": Assembling LHS Matrix")
    if not bcs:
        A = assemble_matrix(a)
    else:
        A = assemble_matrix(a, bcs)
    A.assemble()

    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, name + ": Assembling RHS Vector")
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    return A, b

def OutputMesh(mesh, domains, facets, outname):
    with XDMFFile(MPI.COMM_WORLD,
                 os.path.join(filedir, "models", outname + ".xdmf"),
                 "w",
                 encoding=XDMFFile.Encoding.HDF5) as file:
       file.write_mesh(mesh)
       file.write_meshtags(facets)
       file.write_meshtags(domains)

def SaveSolution(fnc, t, file, name, units):
    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, name + ": Saving Solution")
    file.write_function(fnc, round(t,3))
    
    fncmax, fncmin = fnc.vector.max()[1], fnc.vector.min()[1]
    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, name + " Max: " + \
            str("{:.3e}".format(fncmax)) + " " + units)
        log.log(loglevel, name + " Min: " + \
            str("{:.3e}".format(fncmin)) + " " + units)


### Parameters ###########################################################

t_01 = Timer("01 Initialise Variables")
if MPI.COMM_WORLD.rank == 0:
    log.log(loglevel, "Defining Model Parameters")

# Mesh File Name
meshname = "mesh1.xdmf"

# Frequency (Hz)
freq = 50.0

# Rotational velocity (RPM)
motorrpm = 600.0

# Rotational velocity (rad/s)
omega_v = motorrpm/9.5492965964254

# Angular frequency (Hz)
omega_f = 2.*math.pi*freq

# Relative permeability (-)
mur_air = 1.0
mur_cop = 0.999991
mur_mag = 1.04457
mur_stl = 100.0

# Permeability of free space (H/m)
mu_0 = 1.25663706143592e-06

# Permeability (H/m)
mu_air = mur_air * mu_0
mu_cop = mur_cop * mu_0
mu_mag = mur_mag * mu_0
mu_stl = mur_stl * mu_0

# Electrical conductiviy (S/m)
sigma_air = 1.00e-32
sigma_cop = 1.00e-32
sigma_mag = 1.00e-32
sigma_stl = 2.00e+06

# Supply current (A)
sp_current  = 35.00
# Current density magnitude (A/m^2)
jsource_amp = sp_current/2.47558E-05

# Remanent magnetic flux density (T)
msource_mag_T = 1.09999682447133
# Permanent Magnetization (A/m)
msource_mag   = (msource_mag_T*1e7)/(4*math.pi)

# Uncomment to neglect respective source term     
# jsource_amp = 1.00e-32
# msource_mag = 1.00e-32

# Initial time / Time step / Final time (s)
t         = 0.000
dt        = 0.001
t_final   = 0.005
# dt      = 0.2 if freq == 0 else 0.02/freq
# t_final = 0.8 if freq == 0 else (0.02/freq)*50

# Quadrature degree (-)
quaddeg = 3

# Vector / Scalar order (-)
order_v = 1
order_s = 1

# Helper params
resdirname   = "Results_" + str(int(freq)) + "Hz"
skiptimeloop = False

t_01.stop()

 
### Mesh #################################################################

t_02 = Timer("02 Read Mesh")
filedir = os.getcwd()
meshpath = os.path.join(filedir, "models", meshname)
if MPI.COMM_WORLD.rank == 0: 
    log.log(loglevel, "Reading Mesh: " + meshpath)

with XDMFFile(MPI.COMM_WORLD,
              meshpath,
              "r",
              encoding=XDMFFile.Encoding.HDF5) as file:
    mesh = file.read_mesh(ghost_mode=GhostMode.none,
                          name="mesh",
                          xpath=r"/Xdmf/Domain")
    mesh.topology.create_connectivity_all()
    mt_facets  = file.read_meshtags(mesh, "facets")
    mt_domains = file.read_meshtags(mesh, "domains")

dx = ufl.Measure("dx", subdomain_data=mt_domains, domain=mesh) \
                (metadata={"quadrature_degree": quaddeg})
ds = ufl.Measure("ds", subdomain_data=mt_facets,  domain=mesh) \
                (metadata={"quadrature_degree": quaddeg})
dS = ufl.Measure("dS", subdomain_data=mt_facets,  domain=mesh) \
                (metadata={"quadrature_degree": quaddeg})

mshval = PmsmMeshValues(meshname)
# OutputMesh(mesh, mt_domains, mt_facets, "mesh22")
t_02.stop()


### Function Spaces ######################################################

if MPI.COMM_WORLD.rank == 0: log.log(loglevel, "Creating Function Spaces")
cell = mesh.ufl_cell()

CG_V  = VectorElement("CG", cell, order_v)
CG_F  = FiniteElement("CG", cell, order_s)
DG_V  = VectorElement("DG", cell, 0)

CG_VF = MixedElement([CG_V, CG_F])

V_V    = FunctionSpace(mesh, CG_V)
V_VF   = FunctionSpace(mesh, CG_VF)
V_DGV  = FunctionSpace(mesh, DG_V) 


### UFL Functions ########################################################

(x,y,z) = SpatialCoordinate(mesh)
radius  = as_vector((x,y,0))
normal  = FacetNormal(mesh)
vec0    = as_vector((0,0,0))
x_unit  = as_vector((1,0,0))
y_unit  = as_vector((0,1,0))
z_unit  = as_vector((0,0,1))


### Vector Potential (A) & Scalar Potential (V) ##########################

###### A & V Equation Definition #########################################

t_03 = Timer("03 Define Problem")
if mesh.mpi_comm().rank == 0:
    log.log(loglevel, "AV: Defining Excitations")
u_a, u_v = TrialFunctions(V_VF)
v_a, v_v = TestFunctions(V_VF)

# Create current excitation
jsexp = SupplyCurrentDensity()
jsexp.amp = jsource_amp
jsexp.omega = omega_f
jsource = Function(V_V)
jsource.interpolate(jsexp.eval)
jsource.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                           mode=PETSc.ScatterMode.FORWARD)

view_jsource = PETSc.Viewer().createASCII("JSource.txt")
view_jsource.view(jsource.vector)

# Create magnetization excitation
msexp = PMMagnetization()
msexp.mag = msource_mag
msource = Function(V_V)
msource.interpolate(msexp.eval)
msource.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                           mode=PETSc.ScatterMode.FORWARD)

view_msource = PETSc.Viewer().createASCII("MSource.txt")
view_msource.view(msource.vector)

if mesh.mpi_comm().rank == 0:
    log.log(loglevel, "AV: Defining Formulation")

# Create initial A function
A0 = Function(V_V)
A0.vector.set(0.)
A0.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

# Domain labelling (dx)
# 1| Outer Air Region        9| Coil 4 Region       17| Magnet 5 Region
# 2| R/S Air Gap Region     10| Coil 5 Region       18| Magnet 6 Region
# 3| Shaft Region           11| Coil 6 Region       19| Magnet 7 Region
# 4| Inner Coil Air Region  12| Rotor Region        20| Magnet 8 Region
# 5| Stator Region          13| Magnet 1 Region     21| Magnet 9 Region
# 6| Coil 1 Region          14| Magnet 2 Region     22| Magnet 10 Region
# 7| Coil 2 Region          15| Magnet 3 Region     23| Inner Ends Air Region
# 8| Coil 3 Region          16| Magnet 4 Region     --| 

steel_domain       = (dx(3) + dx(5) + dx(12))
coil_domain        = (dx(6) + dx(7) + dx(8) + dx(9) + \
                      dx(10) + dx(11))
magnet_domain      = (dx(13) + dx(14) + dx(15) + dx(16) + \
                      dx(17) + dx(18) + dx(19) + dx(20) + \
                      dx(21) + dx(22))
air_domain         = (dx(1) + dx(2) + dx(4) + dx(23))

rot_steel_domain   = (dx(3) + dx(12))
rot_mag_domain     = (dx(13) + dx(14) + dx(15) + dx(16) + \
                      dx(17) + dx(18) + dx(19) + dx(20) + \
                      dx(21) + dx(22))

f_a = + inner((1/mu_stl)*grad(u_a),grad(v_a))*steel_domain \
      + inner((1/mu_cop)*grad(u_a),grad(v_a))*coil_domain \
      + inner((1/mu_mag)*grad(u_a),grad(v_a))*magnet_domain \
      + inner((1/mu_air)*grad(u_a),grad(v_a))*air_domain \
\
      + (1/dt)*inner(sigma_stl*(u_a-A0),v_a)*steel_domain \
      + (1/dt)*inner(sigma_cop*(u_a-A0),v_a)*coil_domain \
      + (1/dt)*inner(sigma_mag*(u_a-A0),v_a)*magnet_domain \
\
      + inner(sigma_stl*grad(u_v),v_a)*steel_domain \
      + inner(sigma_cop*grad(u_v),v_a)*coil_domain \
      + inner(sigma_mag*grad(u_v),v_a)*magnet_domain \
\
      - inner(sigma_stl*cross(omega_v*radius,curl(u_a)),v_a)*rot_steel_domain \
      - inner(sigma_mag*cross(omega_v*radius,curl(u_a)),v_a)*rot_mag_domain \
\
      - inner(jsource,v_a)*coil_domain \
      - inner((mu_0/mu_mag)*msource,curl(v_a))*magnet_domain \

f_v = + inner(sigma_stl*grad(u_v),grad(v_v))*steel_domain \
      + inner(sigma_cop*grad(u_v),grad(v_v))*coil_domain \
      + inner(sigma_mag*grad(u_v),grad(v_v))*magnet_domain \
      + inner(sigma_air*grad(u_v),grad(v_v))*air_domain \
\
      - (1/dt)*inner(sigma_stl*(u_a-A0),grad(v_v))*steel_domain \
      - (1/dt)*inner(sigma_cop*(u_a-A0),grad(v_v))*coil_domain \
      - (1/dt)*inner(sigma_mag*(u_a-A0),grad(v_v))*magnet_domain \
\
      + inner(sigma_stl*cross(omega_v*radius,curl(u_a)),grad(v_v))*rot_steel_domain \
      + inner(sigma_mag*cross(omega_v*radius,curl(u_a)),grad(v_v))*rot_mag_domain \

form_av = f_a + f_v
a_av, L_av = system(form_av)


###### A & V Boundary Conditions #########################################

# Apply boundary conditions
if mesh.mpi_comm().rank == 0:
    log.log(loglevel, "AV: Applying Boundary Conditions")
bcs_av = []

u0 = Function(V_VF)
u0.vector.set(0.)
u0.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

x_faces = locate_dofs_topological(V_VF,
              mesh.topology.dim - 1,
              mt_facets.indices[np.logical_or(mt_facets.values == 1,
                                              mt_facets.values == 2)])
y_faces = locate_dofs_topological(V_VF,
              mesh.topology.dim - 1,
              mt_facets.indices[np.logical_or(mt_facets.values == 3,
                                              mt_facets.values == 4)])                                              
z_faces = locate_dofs_topological(V_VF,
              mesh.topology.dim - 1,
              mt_facets.indices[np.logical_or(mt_facets.values == 5,
                                              mt_facets.values == 6)]) 
                                              
bcs_av.append( DirichletBC(u0.sub(0).sub(1), x_faces) )
bcs_av.append( DirichletBC(u0.sub(0).sub(2), x_faces) )
bcs_av.append( DirichletBC(u0.sub(0).sub(0), y_faces) )
bcs_av.append( DirichletBC(u0.sub(0).sub(2), y_faces) )
bcs_av.append( DirichletBC(u0.sub(0).sub(0), z_faces) )
bcs_av.append( DirichletBC(u0.sub(0).sub(1), z_faces) )

t_03.stop()


###### Initialise Derived Quantity Functions #############################

# Magnetic Flux Density (B)
u_b, v_b = TrialFunction(V_DGV), TestFunction(V_DGV)
B = Function(V_DGV)
B.name = "B"

# Eddy Current Density (Je)
u_je, v_je = TrialFunction(V_DGV), TestFunction(V_DGV)
Je = Function(V_DGV)
Je.name = "Je"

# Source Current Density (Js)
u_js, v_js = TrialFunction(V_DGV), TestFunction(V_DGV)
Js = Function(V_DGV)
Js.name = "Js"


###### Initialise Solvers ################################################

# A & V Solver
A_av = assemble_matrix(a_av, bcs_av)
A_av.assemble()
AV = Function(V_VF)

t_04 = Timer("04 Initialise Solvers")
# Direct solver
# solver_av = PETSc.KSP().create(mesh.mpi_comm())
# solver_av.setOperators(A_av)
# solver_av.setType("preonly")
# solver_av.getPC().setType("lu")
# opts_av = PETSc.Options("AV_")
# opts_av["ksp_monitor"] = None
# opts_av["ksp_view"] = None
# solver_av.setFromOptions()

# Iterative solver
solver_av = PETSc.KSP().create(mesh.mpi_comm())
solver_av.setOptionsPrefix("AV_")
solver_av.setOperators(A_av)
solver_av.setTolerances(rtol=1e-08, max_it=100000)
solver_av.setType("gmres")
solver_av.getPC().setType("bjacobi")
opts_av = PETSc.Options("AV_")
opts_av["ksp_monitor"] = None
opts_av["ksp_view"] = None
solver_av.setFromOptions()

# Direct Derived Quantity Solver
solver_dq = PETSc.KSP().create(mesh.mpi_comm())
solver_dq.setOptionsPrefix("DQ_")
solver_dq.setType("preonly")
solver_dq.getPC().setType("lu")
opts_dq = PETSc.Options("DQ_")
opts_dq["ksp_monitor"] = None
opts_dq["ksp_view"] = None
solver_dq.setFromOptions()

# Iterative Derived Quantity Solver
# solver_dq = PETSc.KSP().create(mesh.mpi_comm())
# solver_dq.setOptionsPrefix("DQ_")
# solver_dq.setTolerances(rtol=1e-8)
# solver_dq.setType("gmres")
# solver_dq.getPC().setType("none")
# opts_dq = PETSc.Options("DQ_")
# opts_dq["ksp_monitor"] = None
# opts_dq["ksp_view"] = None
# solver_dq.setFromOptions()

t_04.stop()


### Initialise Solution Files ############################################

file_a = XDMFFile(mesh.mpi_comm(), "A.xdmf", "w")
file_a.write_mesh(mesh)
file_v = XDMFFile(mesh.mpi_comm(), "V.xdmf", "w")
file_v.write_mesh(mesh)

file_b = XDMFFile(mesh.mpi_comm(), "B.xdmf", "w")
file_b.write_mesh(mesh)

file_je = XDMFFile(mesh.mpi_comm(), "Je.xdmf", "w")
file_je.write_mesh(mesh)

file_js = XDMFFile(mesh.mpi_comm(), "Js.xdmf", "w")
file_js.write_mesh(mesh)

file_jt = XDMFFile(mesh.mpi_comm(), "Jt.xdmf", "w")
file_jt.write_mesh(mesh)


### Time Loop ############################################################

while not skiptimeloop:

    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, "==================================================")
        log.log(loglevel, "Solving System at " + str(round(t,3)) + " seconds")
        log.log(loglevel, "==================================================")

    jsemax, jsemin = jsource.vector.max()[1], jsource.vector.min()[1]
    msemax, msemin = msource.vector.max()[1], msource.vector.min()[1]

    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, "Js Expression Max: " + \
            str("{:.3e}".format(jsemax)) + " A/m^2")
        log.log(loglevel, "Js Expression Min: " + \
            str("{:.3e}".format(jsemin)) + " A/m^2")
        log.log(loglevel, "Ms Expression Max: " + \
            str("{:.3e}".format(msemax)) + " A/m^2")
        log.log(loglevel, "Ms Expression Min: " + \
            str("{:.3e}".format(msemin)) + " A/m^2")


###### A & V #############################################################

    t_05 = Timer("05 Assemble System")
    A_av, b_av = AssembleSystem(a_av, L_av, bcs_av, "AV")
    t_05.stop()

    t_06 = Timer("06 Solve for A & V")
    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, "AV: Solving")
    solver_av.solve(b_av, AV.vector)
    solver_av.view()
    t_06.stop()
    
    t_07 = Timer("07 Save A & V Solution")
    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, "AV: Saving Solution")
    A, V = AV.sub(0).collapse(), AV.sub(1).collapse()
    A.name, V.name = "A", "V"
    SaveSolution(A, t, file_a, "A", "Tm")
    SaveSolution(V, t, file_v, "V", "V")
    t_07.stop()
    

###### B #################################################################

    t_08 = Timer("08 Calculate B")

    form_b = + inner(u_b,v_b)*dx \
             - inner(curl(A),v_b)*dx \

    a_b, L_b = system(form_b)

    A_b, b_b = AssembleSystem(a_b, L_b, [], "B")
    
    A_b = assemble_matrix(a_b)
    A_b.assemble()
    
    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, "B: Calculating")
    solver_dq.setOperators(A_b)
    solver_dq.solve(b_b, B.vector)
    t_08.stop()

    t_09 = Timer("09 Save B Solution")
    SaveSolution(B, t, file_b, "B", "Tm")
    t_09.stop()


###### Je ################################################################
    
    t_10 = Timer("10 Calculate Je")
    
    # equals 0 at 1st time step
    a_term = ufl.conditional(ufl.lt(t,dt), vec0, A-A0)
    v_term = ufl.conditional(ufl.lt(t,dt), vec0, grad(V))
    
    form_je = + inner(u_je,v_je)*dx \
              + (1/dt)*inner(sigma_stl*(a_term),v_je)*steel_domain \
              + inner(sigma_stl*(v_term),v_je)*steel_domain \

    a_je, L_je = system(form_je)

    A_je, b_je = AssembleSystem(a_je, L_je, [], "Je")

    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, "Je: Calculating")
    solver_dq.setOperators(A_je)
    solver_dq.solve(b_je, Je.vector)
    t_10.stop()
    
    t_11 = Timer("11 Save Je Solution")
    SaveSolution(Je, t, file_je, "Je", "A/m^2")
    t_11.stop()
    

###### Js ################################################################

    t_12 = Timer("12 Calculate Js")
    
    form_js = + inner(u_js,v_js)*dx \
              - inner(jsource,v_js)*coil_domain \
    
    a_js, L_js = system(form_js)

    A_js, b_js = AssembleSystem(a_js, L_js, [], "Js")

    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, "Js: Calculating")
    solver_dq.setOperators(A_js)
    solver_dq.solve(b_js, Js.vector)
    t_12.stop()
    
    t_13 = Timer("13 Save Js Solution")
    SaveSolution(Js, t, file_js, "Js", "A/m^2")
    t_13.stop()
    
    
###### Jt ################################################################
    
    t_14 = Timer("14 Calculate Jt")
    
    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, "Jt: Calculating")
    Jt = SumFunctions(V_DGV, "Jt", Je, Js)
    t_14.stop()
    
    t_15 = Timer("15 Save Jt Solution")
    SaveSolution(Jt, t, file_jt, "Jt", "A/m^2")
    t_15.stop()
    

###### Is ################################################################
    
    t_16 = Timer("16 Calculate Is")
    
    # Source Current Calculation
    coil1 = assemble_scalar(dot(jsource,z_unit)*dS(mshval.coils[0]))
    coil2 = assemble_scalar(dot(jsource,z_unit)*dS(mshval.coils[1]))
    coil3 = assemble_scalar(dot(jsource,z_unit)*dS(mshval.coils[2]))
    coil4 = assemble_scalar(dot(jsource,z_unit)*dS(mshval.coils[3]))
    coil5 = assemble_scalar(dot(jsource,z_unit)*dS(mshval.coils[4]))
    coil6 = assemble_scalar(dot(jsource,z_unit)*dS(mshval.coils[5]))
    
    coil_is = []
    coil_is.append( mesh.mpi_comm().allreduce(coil1, op=MPI.SUM) )
    coil_is.append( mesh.mpi_comm().allreduce(coil2, op=MPI.SUM) )
    coil_is.append( mesh.mpi_comm().allreduce(coil3, op=MPI.SUM) )
    coil_is.append( mesh.mpi_comm().allreduce(coil4, op=MPI.SUM) )
    coil_is.append( mesh.mpi_comm().allreduce(coil5, op=MPI.SUM) )     
    coil_is.append( mesh.mpi_comm().allreduce(coil6, op=MPI.SUM) )
    t_16.stop()
    
    if mesh.mpi_comm().rank == 0:
        for i in range(len(coil_is)):
            log.log(loglevel, "Current in Coil " + \
                str(i+1) + \
                ": " + \
                str("{:.3e}".format(abs(coil_is[i]))) + \
                " Amperes")


###### Pc ################################################################

    t_17 = Timer("17 Calculate Pc")
    
    # Copper loss density in the internal coils of the motor
    coil_pc = []
    coil_pc.append( (1/sigma_cop)*coil_is[0]**2 )
    coil_pc.append( (1/sigma_cop)*coil_is[1]**2 )
    coil_pc.append( (1/sigma_cop)*coil_is[2]**2 )
    coil_pc.append( (1/sigma_cop)*coil_is[3]**2 )
    coil_pc.append( (1/sigma_cop)*coil_is[4]**2 )
    coil_pc.append( (1/sigma_cop)*coil_is[5]**2 )
    t_17.stop()

    if mesh.mpi_comm().rank == 0:
        for i in range(len(coil_pc)):
            log.log(loglevel, "Copper loss in Coil " + \
                str(i+1) + \
                ": " + \
                str("{:.3e}".format(abs(coil_pc[i]))) + \
                " Watts")


###### Pe ################################################################

    t_18 = Timer("18 Calculate Pe")

    # Eddy current losses
    eddy_loss = ((1/sigma_stl)*(Je**2))*steel_domain
    
    Pe = mesh.mpi_comm().allreduce(assemble_scalar(eddy_loss), op=MPI.SUM)
    t_18.stop()

    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, "Eddy current losses in steel domains " + \
            str("{:.3e}".format(Pe)) + " Watts")


###### T #################################################################
    
    t_19 = Timer("19 Calculate T")
    
    if mesh.mpi_comm().rank == 0: log.log(loglevel, "T: Calculating")

    t_surfaces =   (dS(mshval.torqs[0]) \
                  + dS(mshval.torqs[1]) \
                  + dS(mshval.torqs[2]) \
                  + dS(mshval.torqs[3]) \
                  + dS(mshval.torqs[4]) \
                  + dS(mshval.torqs[5]) \
                  + dS(mshval.torqs[6]))
    
    # translate position from cartesian to cylindrical
    cyl_r = sqrt(x**2 + y**2)
    cyl_t = ufl.atan_2(y, x)
    cyl_t = ufl.conditional(ufl.lt(cyl_t,0), cyl_t+2*ufl.pi, cyl_t)
    cyl_z = z
    
    # create matrix to translate cartesian unit vector to cylindrical
    trans_matrix = ufl.as_matrix([[ufl.cos(cyl_t)   , ufl.sin(cyl_t), 0],
                                  [-1*ufl.sin(cyl_t), ufl.cos(cyl_t), 0],
                                  [0                , 0             , 1]])
    
    # translate the cartesian defined B vectors to cylindrical
    b_trans = (trans_matrix*B)
    b_r     = avg(b_trans[0])
    b_t     = avg(b_trans[1])
    b_z     = avg(b_trans[2])
    
    b_1 = (1/mu_0)*((b_r**2 - b_t**2 - b_z**2)/2)
    b_2 = (1/mu_0)*(b_t*b_r)
    b_3 = (1/mu_0)*(b_z*b_r)
    
    mst_fx  = (b_1*ufl.cos(cyl_t) - b_2*ufl.sin(cyl_t))*t_surfaces
    mst_fy  = (b_1*ufl.sin(cyl_t) + b_2*ufl.cos(cyl_t))*t_surfaces
    mst_fr  = b_1*t_surfaces
    mst_ft  = b_2*t_surfaces
    mst_fz  = b_3*t_surfaces
    mst_tz  = (cyl_r*b_2)*t_surfaces
    
    Fx = mesh.mpi_comm().allreduce(assemble_scalar(mst_fx), op=MPI.SUM)
    Fy = mesh.mpi_comm().allreduce(assemble_scalar(mst_fy), op=MPI.SUM)
    Fr = mesh.mpi_comm().allreduce(assemble_scalar(mst_fr), op=MPI.SUM)
    Ft = mesh.mpi_comm().allreduce(assemble_scalar(mst_ft), op=MPI.SUM)
    Fz = mesh.mpi_comm().allreduce(assemble_scalar(mst_fz), op=MPI.SUM)
    Tz = mesh.mpi_comm().allreduce(assemble_scalar(mst_tz), op=MPI.SUM)
    
    t_19.stop()
    
    if mesh.mpi_comm().rank == 0:
        log.log(loglevel, "Tz: " + str("{:.3e}".format(Tz)))
        log.log(loglevel, "Fx: " + str("{:.3e}".format(Fx)))
        log.log(loglevel, "Fy: " + str("{:.3e}".format(Fy)))
        log.log(loglevel, "Fr: " + str("{:.3e}".format(Fr)))
        log.log(loglevel, "Ft: " + str("{:.3e}".format(Ft)))
        log.log(loglevel, "Fz: " + str("{:.3e}".format(Fz)))


###### Update Parameters #################################################
    
    t_20 = Timer("20 Update Variables")
    
    t = t + dt
    if round(t,8) > round(t_final,8):
        break
    
    jsexp.t = t
    jsource.interpolate(jsexp.eval)
    jsource.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                               mode=PETSc.ScatterMode.FORWARD)
    
    A.vector.copy(result=A0.vector)
    
    t_20.stop()


### Ending ###############################################################

# Close result files
file_a.close()
file_v.close()
file_b.close()
file_je.close()
file_js.close()
file_jt.close()

if mesh.mpi_comm().rank == 0:
    log.log(loglevel, "==================================================")
    log.log(loglevel, "Simulation Finished")
    log.log(loglevel, "==================================================")

t_run.stop()
list_timings(mesh.mpi_comm(), [TimingType.wall])