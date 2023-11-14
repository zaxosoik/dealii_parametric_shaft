# Parametric Shaft Study with deal.II
## Overview
**This repository contains code for analysing a ship's shafting system parametrically**. It receives input from an parameters.xml file, which can be maniplulated by deal.II's ParameterGui. The parameters include the length of the shaft, the radius, material properties, engine specifics, ship speed, solver parameters, and bearing parameters. For the time being only one bearing is beign considered, but the code can be easily modified to receive multiple bearings.
The FEA analysis can be dynamic, which includes inertia forces and damping, or static. 
The goal of the code is to be easily accessible, scalable and parametric-design oriented. It utilises MPI so that it is scalable, as well as PetsC matrices and solvers. In the code the shaft is divided into partitions (MPI cores) which can be examined in Paraview.
The code is written and used in ubuntu OS, it uses cmake to create the Makefile and the libraries with which deal.II is compilled with are: HDF5 (used to output the results for postprocessing), MPI, Petsc, Arpack, MPI, ScalaPack, petsc4py, with complex numbers, Slepc.
The executable also calls the python postprocessing script which in turn outputs the maximum norm of stress per dx of the shaft and a 3D surface plot that examines the outer shell points, ploting their norm of stress in the z-axis, angle=0 is at z=radius of the shaft.
In the code the Engine Power applied on the propeller side of the shaft is calculated based on the propeller law and divided by the ship's velocity to calculate the froce. The torque is also calculated based on the propeller law and applied on each point as a force with it's arm being the distance from the centroid.
The bearing in the present code is simulated by a vertical movement of all of the boundary, but can be easily modified to act as a spring or have as an input the pressure produced by the oil film (if the the Reynolds equation is solved) and can potentialy simulate an imperfect alignment.
## Goals
- [X] Parametrically creating and meshing the shaft
- [X] Calculating the forces from the propeller
- [X] Static Analysis of the shaft without a bearing
- [X] Implementing the bearing as a displacement
- [X] Dynamic Simulation using Newmark's method for time integration
- [X] Plotting results to compare against analytical solutions
- [X] Automatic postprocessing
- [ ] Implementing bearing with the Reynold's Equation solution
- [X] Implementing the Flywheel
- [X] Calculating Eigenvalues without consuming too much RAM   
- [ ] Plotting Campbell diagrams
- [ ] Batchrun a large volume of shafting systems
- [ ] Physics' Informed Neural Network
- [ ] 1D Simulation of the Engines Kinematics
### Future Goals
- [ ] Full FEA of the Engines Kinematics
- [ ] Adding frictional effects on bearings (friction coefficient)

## Changes
1. v3 Creation of the repository
2. v4 Correcting the refine_initial_step to set boundary_ids for bearings
