#!/usr/local/bin/python3
import bpy
import os, sys
import numpy
import csv
from math import pi
from bpy import context
from mathutils import Vector, Matrix
import numpy as np
import cv2

#Add current directory to the system path 
curDir = os.path.realpath(__file__)
curDir = os.path.dirname( curDir )
sys.path.append(curDir)
from makeTrainData import *

### BEGINNING OF THE USER PARAMETERS ###

#Set the folder where the STL files are located
localSTLFolder = "/home/thefroggy/Documents/MVA/ObjectRecognition/project/Data/LegoPiecesSTL/"

#Set the folder where the images are going to be rendered
# !!! Absolute path mandatory !!! #
# !!! Must be an empty folder !!! #
renderingFolder = "/home/thefroggy/Documents/MVA/ObjectRecognition/project/Data/LegoPiecesBlender/"

#Set the assembly rules folder
assemblyFolder = "/home/thefroggy/Documents/MVA/ObjectRecognition/project/Data/assemblyRules/"

#Set the folder where the backgrounds are located
backgroundFolder = "/home/thefroggy/Documents/MVA/ObjectRecognition/project/Data/Backgrounds/"

#Set the folder where the assembled data are going to be stored
assembledDataFolder = "/home/thefroggy/Documents/MVA/ObjectRecognition/project/Data/images/"

#Set the folder where the annotations will be stored
annotationFolder = "/home/thefroggy/Documents/MVA/ObjectRecognition/project/Data/labels/"

#Set the folder where the train files will be stored
# Train files includes class names and images list
trainFileFolder = "/home/thefroggy/Documents/MVA/ObjectRecognition/project/Data/train_cfg/"

#Number of point of views
nPov = 200

#Number of train files to generate
nbTrain = 3000

#Percentage of files used for evaluation
p = 0.1

### END OF THE USER PARAMETERS ###

#Parsing the assembly files
def parseAssemblies(path):
  #Define variables that associate a number to an assembly
  assToInt = dict()
  intToAss = []
  assemblies = dict()
  i = 0
  for file in os.listdir(path): #For each assembly file
    if file.endswith(".csv"):
      assToInt[file[0:-4]] = i
      intToAss.append(file[0:-4])
      i = i + 1
      with open(path+file, 'rt') as f:
        reader = csv.reader(f,delimiter=';')
        assembly = []
        for row in reader: #For each file in the current assembly
          if row[0] == "PieceName":
            continue
          curPiece = dict()
          curPiece["Name"] = row[0]
          curPiece["RGB"] = (int(row[1]),int(row[2]),int(row[3]))
          curPiece["Matrix"] = Matrix(((float(row[4]),float(row[5]),float(row[6]),float(row[7])),(float(row[8]),float(row[9]),float(row[10]),float(row[11])),(float(row[12]),float(row[13]),float(row[14]),float(row[15])),(float(row[16]),float(row[17]),float(row[18]),float(row[19]))))
          assembly.append(curPiece)
        assemblies[file[0:-4]] = assembly
  return assemblies,assToInt,intToAss

#Computing the radius
def computeBoxRad(obj,center):
  bestRad = 0.
  for b in obj.bound_box:
    bVec=Vector([b[0],b[1],b[2]])
    curRad = (bVec-center).length
    bestRad = curRad if curRad > bestRad else bestRad 
  return bestRad

#Function to clean the blender workspace
def remove_obj_lamp_and_mesh(context):
    scene = context.scene
    objs = bpy.data.objects
    meshes = bpy.data.meshes
    for obj in objs:
        if obj.type == 'MESH' or obj.type == 'LAMP':
            scene.objects.unlink(obj)
            objs.remove(obj)
    for mesh in meshes:
        meshes.remove(mesh)

#Get a random point on the surface of the sphere (center,radius)
def getRandomPointAroundSphere(center,radius):
   theta = numpy.random.uniform(0.,1.)*pi
   phi = numpy.random.uniform(0.,2.)*pi
   x = radius * numpy.sin( theta ) * numpy.cos( phi )
   y = radius * numpy.sin( theta ) * numpy.sin( phi )
   z = radius * numpy.cos( theta )
   return (x+center[0],y+center[1],z+center[2])

#Make a material with the specified name, and the specified color
def getMaterial(fileName,red,green,blue):
  mat = bpy.data.materials.new(name="Material"+fileName)
  mat.diffuse_color = [red, green, blue]
  return mat

#Creating a lamp with an appropriate energy
def makeLamp(rad):
  #Creating a lamp
  # Create new lamp datablock
  lamp_data = bpy.data.lamps.new(name="Lamp", type='POINT')
  lamp_data.distance = rad * 2.5
  lamp_data.energy = rad/28.6
  # Create new object with our lamp datablock
  lamp_object = bpy.data.objects.new(name="Lamp", object_data=lamp_data)
  # Link lamp object to the scene so it'll appear in this scene
  scene = bpy.context.scene
  scene.objects.link(lamp_object)
  return lamp_object

#Render the current frame with a redirection of the flow in a log file
def renderWithoutOutput():
  # redirect output to log file
  logfile = 'blender_render.log'
  open(logfile,'a').close()
  old = os.dup(1)
  sys.stdout.flush()
  os.close(1)
  os.open(logfile, os.O_WRONLY)
  #Render
  bpy.ops.render.render(write_still=True) 
  # disable output redirection
  os.close(1)
  os.dup(old)
  os.close(old)

#Grows src bounding box to upd bounding box
def growBox(src,upd):
  minX = min(src[0][0],upd[0][0])
  minY = min(src[0][1],upd[0][1])
  minZ = min(src[0][2],upd[0][2])
  maxX = max(src[6][0],upd[6][0])
  maxY = max(src[6][1],upd[6][1])
  maxZ = max(src[6][2],upd[6][2])
  newBox = []
  newBox.append(Vector([minX,minY,minZ]))
  newBox.append(Vector([minX,minY,maxZ]))
  newBox.append(Vector([minX,maxY,maxZ]))
  newBox.append(Vector([minX,maxY,minZ]))
  newBox.append(Vector([maxX,minY,minZ]))
  newBox.append(Vector([maxX,minY,maxZ]))
  newBox.append(Vector([maxX,maxY,maxZ]))
  newBox.append(Vector([maxX,maxY,minZ]))
  return newBox

#Setting up the environment
bpy.context.scene.render.alpha_mode = 'TRANSPARENT' 
bpy.context.scene.render.resolution_x = 1080
bpy.context.scene.render.resolution_y = 1080

#Rendering procedure for the assemblies

#Parsing the assemblies
assemblies,assToInt,intToAss = parseAssemblies(assemblyFolder)
print('Found ' + str(len(assemblies)) + ' assemblies to be rendered')
#List that stores the rendered images
renderedImages = []
for key,assembly in assemblies.items(): #For each assembly
  remove_obj_lamp_and_mesh(bpy.context) #Cleaning the blender workspace
  os.makedirs(renderingFolder+key) #Making an appropriate directory
  print('Processing assembly '+key)
  globalBoundingBox = []
  for fileInfo in assembly: #For each file in the assembly
    file = fileInfo["Name"]
    #Import the mesh in current file in blender
    bpy.ops.import_mesh.stl(filepath=localSTLFolder, filter_glob="*.STL",  files=[{"name":localSTLFolder+file}], directory=".")
    #Get the current object
    obj = context.active_object
    #Move object to its place
    obj.matrix_world = fileInfo["Matrix"]
    # Assign material to object
    mat = getMaterial(file,*fileInfo["RGB"])
    if obj.data.materials:
    # assign to 1st material slot
      obj.data.materials[0] = mat
    else:
    # no slots
      obj.data.materials.append(mat)
    if len(globalBoundingBox) == 0:
      globalBoundingBox = [Vector(b) for b in obj.bound_box]
    else:
      globalBoundingBox = growBox(globalBoundingBox,[Vector(b) for b in obj.bound_box]) 
  obj = context.active_object
  #Get the coordinates of the center of mass and the radius
  global_bbox_center = 0.125 * sum((b for b in globalBoundingBox), Vector())
  rad = computeBoxRad(obj,global_bbox_center)
  #Add a lamp (its intensity depends on rad)
  lamp_object = makeLamp(rad)
  rad = 5 * rad #We will place the camera away from the object
  for i in range(nPov): #For every point on view
    xCam,yCam,zCam = getRandomPointAroundSphere(global_bbox_center,rad)
    if(len(bpy.data.cameras) == 1):
      objCam = bpy.data.objects['Camera']
      #Place the camera
      objCam.location = (xCam,yCam,zCam)
      bpy.data.cameras[bpy.context.scene.camera.name].clip_end = 1E6
      #Point the camera to the object
      direction = global_bbox_center - Vector([xCam,yCam,zCam])
      # point the cameras '-Z' and use its 'X' or 'Y' as up
      dirUp = 'X' if numpy.random.randint(2) == 0 else 'Y'
      rot_quat = direction.to_track_quat('-Z', dirUp)
      # assume we're using euler rotation
      context.scene.camera.rotation_mode = 'XYZ'
      objCam.rotation_euler = rot_quat.to_euler()
      #Place the lamp
      lamp_object.location = (xCam, yCam, zCam+10)
      #Render the scene
      imagePath = renderingFolder+key+'/'+key+'_'+str(i)+'.png'
      bpy.data.scenes['Scene'].render.filepath = imagePath
      renderWithoutOutput()
      cropToBoundingBox(imagePath) 
      renderedImages.append([imagePath,assToInt[key]])
print('Assemblies succesfully rendered!')

#Procedure to generate the train data out of the renderings and the background

print('Beginning the generation of the train data')
#Create the file storing the list of train images
trainList = open(trainFileFolder+'train.txt','w')
#Create the file storing the list of val images
valList = open(trainFileFolder+'val.txt','w')
#Create the file that stores the class names
nameList = open(trainFileFolder+'names.txt','w')
#print('Length of intToAss : '+str(len(intToAss)))
for name in intToAss:
  nameList.write(name+'\n')
#Load the backgrounds
backgrounds = []
for bgName in os.listdir(backgroundFolder):
  backgrounds.append(backgroundFolder+bgName)
#Generate nbTrain train data
for i in range(nbTrain):
  #Load a random background
  bgNumber = np.random.randint(0,len(backgrounds))
  #Load at most 3 class
  nbClassToAdd = np.random.randint(1,min(3,len(intToAss)+1))
  classToAdd = []
  for j in range(nbClassToAdd):
    classToAddIndex = np.random.randint(0,len(renderedImages))
    classToAdd.append(renderedImages[classToAddIndex])
  fileName = 'train_'+str(i)
  #Generate the corresponding train image
  insertAndAnnotate(classToAdd,backgrounds[bgNumber],fileName,assembledDataFolder,annotationFolder)
  #Add it in the train or val list
  if np.random.binomial(1,p) == 0:
    trainList.write(assembledDataFolder+fileName+'.jpg'+'\n')
  else:
    valList.write(assembledDataFolder+fileName+'.jpg'+'\n')

