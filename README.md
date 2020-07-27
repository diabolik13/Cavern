# Nonlinear 2D Finite Element Modeling: Cyclic Energy Storage in Salt Caverns with Creep Deformation Physics
![](animation.gif)

### Introduction
The aim of this 2D Finite Element Method (FEM) based simulator is to calculate deformations and corresponding stress distribution in a salt cavern storage surronding due to a set of forces acting on the cavern's wall.

The capabilities of the simulator include quantification of displacement, strain and stress of linear elastic model, tertiary and secondary creep model, cyclic loading model, heterogeneous model and combination of the aforementioned models.

Input of the problem is given by a set of parameters, namely rock salt and overburden densities, depth of the roof of the cavern, temperature of the rock salt domain and set of material properties.

### Getting Started
Please follow these steps to be able to run this project:

 1. Install required dependencies. It is highly recommended to install all dependencies in pipenv virtual environment (see [guide](https://realpython.com/pipenv-guide/)) using the pipfile (see [guide](https://pipenv-fork.readthedocs.io/en/latest/basics.html)).

 2. Download [ParaView](https://www.paraview.org/) to see the results of the simulation.

 3. Decide which format you want to use for saving the simulation results: *.xdmf, *.gif or *.png. You can also export the resulting set of every parameter to *xls spreadsheet by using the following command:
     ```shell
     write_xls(filename, output)
     ```

 4. Run the 'main.py' script from terminal with:
    ```shell
    python main.py
    ```
    or in your IDE.

### Structure of the code
The simulator is written in Finite Element Method Object Oriented Programming (FEMOOP) structure, which means that the FEM routine is implemented using classes and methods. The main classes, that form the engine of the simulator are contained in the 'classes.py' library and namely are:

 1. 'class Mesh(object):' - Instance of class `Mesh` 
