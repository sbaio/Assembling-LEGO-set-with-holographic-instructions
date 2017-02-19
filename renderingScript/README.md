# Rendering tool for STL files

## Installation

* Download python 3 : sudo apt-get install python3
* Download numpy : sudo pip3 install numpy
* Install blender as a python module : [blender as a python module](https://wiki.blender.org/index.php/User:Ideasman42/BlenderAsPyModule)
* Compile opencv yourself in order to have the python3 wrap : [opencv download link](http://docs.opencv.org/3.1.0/d7/d9f/tutorial_linux_install.html)

##Use

* Check that the first line of render.py points to the python interpreter where you have installed numpy blender and opencv. To check this, start the interpreter and try the following commands for which you should have no error : 
  * `import numpy`
  * `import bpy`
  * `import cv2`
* Set the parameters at the beginning of the file render.py
* Create the assembly files (see next section) and put it in the assembly rules folder that you have specified in render.py
* run render.py with `./render.py`
 
# The assembly files

The assembly files are csv files that explain blender how to build a renderable assembly from the raw STL files. As a consequence, you must create as much assembly file as classes that you want to render. An example assembly file is given in assembly1.csv. You can edit these files with your favourite spreadsheet software.

The structure of this file is pretty simple. The first line gives the description of the fields used for each column. Each line after the first one represents one part that is added to the assembly.

Here are the required features for each part : 

* The name of the STL file so that the algorithm can find it. (Just the name is sufficient. The path is known from the parameters that you must have set in render.py)
* The diffuse color of the part (RGB)
* Its world matrix which represents its position in space. In order to have it, we recommand to place the assembly in blender. Once the assembly is perfectly fitted, do the following procedure for each part in the assembly : select the part with right click ; open the python3 console in blender and type `bpy.context.active_object.matrix_world`.


