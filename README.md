# Finite Difference Method Solvers

This project contains a collection of Python scripts that use the finite difference method to find numerical solutions for various ordinary and partial differential equations (ODEs and PDEs).

## Requirements

This project uses the following Python libraries:

*   **NumPy**: For numerical operations and matrix manipulation.
*   **Matplotlib**: For plotting results and creating animations.
*   **SymPy**: For symbolic mathematics, used to define analytical solutions and source functions.

You can install them using pip:
```bash
pip install numpy matplotlib sympy
```

## Setup

The `heatEquation2D.py` script saves animations as GIF files. Before running it, you must create a folder named `tmp` in the root directory of the project.

```bash
mkdir tmp
```

## Solver Descriptions

Below is an explanation of the primary scripts in this repository.

### Ordinary Differential Equations

*   `finite_differences.py`: Solves a second-order linear ODE of the form `a*y + b*y' + c*y'' = 0` with specified boundary conditions. It constructs a system of linear equations `Ax = B` and solves for the function `y` over a given interval.

### Poisson's Equation (`∇²u = Q`)

There are multiple scripts for solving the 2D Poisson's equation.

*   `poisson.py`: An implementation that solves the 2D Poisson's equation on a rectangular domain with Dirichlet boundary conditions. It manually constructs the block tridiagonal matrix representing the discretized Laplacian operator.
*   `poisson_fd.py`: A more advanced implementation that also solves the 2D Poisson's equation. It uses Kronecker products to construct the Laplacian matrix and employs restriction/prolongation matrices to handle boundary conditions. This approach is more general and can be easier to adapt.
*   `poissonData.py` & `poissonFuncts.py`: These appear to be refactored modules for the logic found in `poisson.py`, separating the data (boundary conditions, domain size) from the matrix-generation functions.

### Heat Equation (`α∇²u = uₜ`)

*   `heatEquation1D.py`: Solves the 1D heat equation using an implicit method. It handles Dirichlet boundary conditions using restriction and prolongation matrices.
*   `heatEquation2D.py`: Solves the 2D heat equation with homogenous Dirichlet boundary conditions using an explicit forward-time, centered-space (FTCS) scheme. It generates animations of both the approximate solution and the analytical solution for comparison, and also visualizes the error over time.

### Navier-Stokes Equation

*   `navier-stokes.ipynb` & `stokes-test.ipynb`: These are Jupyter notebooks representing attempts to solve the Navier-Stokes equation. As noted, `stokes-test.ipynb` is the more current but still experimental version.

### Other Notebooks

*   `poissonv_2.ipynb`: Solves the Poisson equation (or Laplace's equation).
*   `heatEquation1D.ipynb`: Solves the 1D heat equation.
*   `heatEquation2D.ipynb`: Solves the 2D heat equation, with options for Neumann boundary conditions.
*   `heatEquationNeumann2D.ipynb`: A version specifically for 2D heat equation with Neumann boundary conditions.
*   `CD_eq_2d.ipynb`: Solves the 2D convection-diffusion equation.

> **Note on Boundary Conditions:**
> *   **Dirichlet** conditions prescribe the value of the function at the boundary (e.g., `u(0, t) = 1`).
> *   **Neumann** conditions prescribe the value of the derivative of the function at the boundary (e.g., `u'(0, t) = 0`).
