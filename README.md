# Software for Development of Low-Cost Partial Hand Prostheses for the 3D-Printed Prosthetics Community

This software provides a solution to those who 3D-print prosthetics in communities, like eNABLE, for creating custom partial hand prostheses. It allows a user to upload a scan of a residual limb and semi-automatically generate a prosthetic hand optimized for that individual.

## Description

## Getting Started

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

## Authors

Katherine Robertson | 24robertsonk@sagehillschool.org
