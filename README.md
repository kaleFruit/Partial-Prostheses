# Software for Development of Low-Cost Partial Hand Prostheses for the 3D-Printed Prosthetics Community

This software provides a solution to those who 3D-print prosthetics in communities, like eNABLE, for creating custom partial hand prostheses. It allows a user to upload a scan of a residual limb and semi-automatically generate a prosthetic hand optimized for that individual.

## Description

- Users can toggle and manipulate a pseudo skeleton of joints to define the finger generation
- Users can "paint" the residual limb where they want to define the socket

## Current Items in the Works

- Creating a desktop executable
- Adding file upload/download to the GUI
- Creating a better arm/wrist band
- Changing the socket to a Voronoi pattern

## Getting Started

1. Set up the Conda environment:
   1. Create a new Conda environment with Python 3.9 using: `conda create -n partialProsGen python=3.10`
   2. After activating the environment, install VTKBool with: `conda install -c conda-forge vtkbool`
   3. Install Pyvista with: `conda install -c conda-forge pyvista`
   4. Install Scipy with: `conda install scipy`
   5. Install PyQt5 with: `pip install PyQt5`
   6. Install Open3D with: `pip install open3d`
   7. Install Pandas with: `pip install pandas`
   8. Install plyfile with: `pip install plyfile`
   9. Install tqdm with: `pip install tqdm`
2. After downloading the files off github, run main.py

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
- open3d
- Pandas

## Authors

Katherine Robertson
