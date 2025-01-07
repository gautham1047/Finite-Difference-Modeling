# Navier-Stokes (+ etc) Solver

These are all the solvers that I've created so far. They use finite differnces to approximate various equations.

All of them should be complete with the exception of the navier-stokes solver. The most up-to-date solver for the Navier-Stokes Equation is in "stokes-test.ipynb".

## Explainations of Each File

- finite_differences.py is for solving ordinary differnetial equations (up to second order). I think it may still have some code errors from when I was testing stuff out. 
- poissonv_2.ipynb solves the poisson equation (or more accurately, Laplace's equation, because I account for any possible source function).
- heatEquation1D.ipynb solves the heat equation in one dimension. heatEquation2D.ipynb solves the heat equation in two dimensions, and gives you the option of using Neumann boundary conditions instead of dirichlet boundary conditions. 
- *Dirichlet boundary conditions prescribe the value of the function at the edges, while Neumann boundary conditions tell you the value of the derivative at the edges. 
- heatEquationNeumann2D.ipynb is a version that only handles Neumann boundary conditions. Only the y boundary conditions can be Neumann boundary conditions for computataional reasons. 
- CD_eq_2d.ipynb is for the convection-diffusion equation. 
- navier-stokes.ipynb is an old attempt at solving the Navier-Stokes equation, and stokes-test.ipynb is a more current, but still buggy, version of the Navier-Stokes solver. There are multiple versions of this latest file

