# Software for Development of Low-Cost Partial Hand Prostheses for the 3D-Printed Prosthetics Community

This software provides a solution to those who 3D-print prosthetics in communities, like eNABLE, for creating custom partial hand prostheses. It allows a user to upload a scan of a residual limb and semi-automatically generate a prosthetic hand optimized for that individual.

## Description

* Users can toggle and manipulate a pseudo skeleton of joints to define the finger generation
* Users can "paint" the residual limb where they want to define the socket

## Getting Started

1. Setting up a new Conda environment through the ternminal with the correct dependencies:1. Create a new conda environment with Python 3.9 using: `conda create -n hapticsHarnessGenerator python=3.9`
2. Install VTKBool with: `conda install -c conda-forge vtkbool`
3. Install Pyvista with: `conda install -c conda-forge pyvista`
4. Install PyQt5 with: `pip install PyQt5`
5. ...

### Software Operation

1. Upload a scan of the residual limb as a .ply file
2. While in the "Finger Generator" tab:
   1. Position the pseudo skeleton of joints by clicking and dragging on the joints
   2. Disable/enable joints using the sliders for each finger
   3. Click "Generate Fingers" to generate the 3D files for the fingers
3. Switch to the "Socket Generator" tab:
   1. Enable the residual limb and use the paint tools to select areas where the socket will be generate; the lighter the color of the paint, the more breathable/flexible the socket will be in that area
   2. Click "Generate Soft Socket" then click "Generate Hardshell of Socket" to complete the socket generation
4. The .stl files for 3D printing will be generated under the "toBePrinted" folder

### Dependencies

- Python
- VTK
- Pyvista
- Numpy
- SciPy
- PyQt5
- VTKBool

## Authors

Katherine Robertson | 24robertsonk@sagehillschool.org
