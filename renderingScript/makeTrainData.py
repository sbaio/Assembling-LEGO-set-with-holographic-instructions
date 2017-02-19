import cv2
import numpy as np

#Crop an image to its bounding box
def cropToBoundingBox(imagePath):
  #print(imagePath)
  img = cv2.imread(imagePath,cv2.IMREAD_UNCHANGED)
  #Use the alpha channel as a mask
  threshold = img[:,:,3]
  #Get the bounding box
  min_rec = cv2.boundingRect(threshold)
  #Crop the picture
  img = img[min_rec[1]:min_rec[1]+min_rec[3],min_rec[0]:min_rec[0]+min_rec[2]]
  #Save it
  cv2.imwrite(imagePath,img)
  #cv2.rectangle(img,(min_rec[0],min_rec[1]),(min_rec[0]+min_rec[2],min_rec[1]+min_rec[3]),(0,255,0),2)
  #cv2.imshow('image',img)
  #cv2.waitKey(0)

#Converts the bounding box to the YOLO format (function from the YOLO developers)
def convert(size, box):
  dw = 1./size[0]
  dh = 1./size[1]
  x = (box[0] + box[1])/2.0
  y = (box[2] + box[3])/2.0
  w = box[1] - box[0]
  h = box[3] - box[2]
  x = x*dw
  w = w*dw
  y = y*dh
  h = h*dh
  return (x,y,w,h)

#insert the images in renderList in the image located in backgroundPath. 
#Gives the result the name fileName and save it in savePath
#Adds the corresponding annotation in the annotationPath folder
def insertAndAnnotate(renderList,backgroundPath,fileName,savePath,annotationPath):
  bg = cv2.imread(backgroundPath,cv2.IMREAD_UNCHANGED)
  annotationFile = open(annotationPath+fileName+'.txt','w')
  for imagePath,imageClass in renderList:
    curImg = cv2.imread(imagePath,cv2.IMREAD_UNCHANGED)
    #Resize the picture randomly between 5% and 90% of the background.
    #Threshold the percentage if it exceeds the curImg dimension.
    bigDimension = 0 if curImg.shape[0] > curImg.shape[1] else 1
    smallDimension = int(not bigDimension)
    percentage = np.random.uniform(0.05,min(0.9,float(curImg.shape[bigDimension])/float(bg.shape[bigDimension])))
    newBigDimension = int(percentage*bg.shape[bigDimension])
    newSmallDimension = int(newBigDimension * curImg.shape[smallDimension] / curImg.shape[bigDimension])
    newShape = (newBigDimension,newSmallDimension) if curImg.shape[0] < curImg.shape[1] else (newSmallDimension,newBigDimension)
    curImg = cv2.resize(curImg,newShape) 
    #Choose a random offset
    x_offset = np.random.randint(0,bg.shape[1]-curImg.shape[1])
    y_offset = np.random.randint(0,bg.shape[0]-curImg.shape[0])
    #Insert the image
    for c in range(0,3):
      bg[y_offset:y_offset+curImg.shape[0], x_offset:x_offset+curImg.shape[1], c] = curImg[:,:,c] * (curImg[:,:,3]/255.0) +  bg[y_offset:y_offset+curImg.shape[0], x_offset:x_offset+curImg.shape[1], c] * (1.0 - curImg[:,:,3]/255.0)
    #Add the annotation
    boundBox = convert((bg.shape[1],bg.shape[0]),(float(x_offset),float(x_offset+curImg.shape[1]),float(y_offset),float(y_offset+curImg.shape[0])))
    annotationFile.write(str(imageClass) + " " + " ".join([str(a) for a in boundBox]) + '\n')
  #cv2.rectangle(bg,(x_offset,y_offset),(x_offset+curImg.shape[1],y_offset+curImg.shape[0]),(0,255,0),2)
  #Write the image
  cv2.imwrite(savePath+fileName+".jpg",bg)
  #cv2.imshow('image',bg)
  #cv2.waitKey(0)

#insertAndAnnotate([["/home/thefroggy/Documents/MVA/ObjectRecognition/project/Data/LegoPiecesBlender/assembly1/assembly1_0.png",0]],"/home/thefroggy/Documents/MVA/ObjectRecognition/project/Data/Backgrounds/IMG_0529.JPG","test","/home/thefroggy/Documents/MVA/ObjectRecognition/project/Data/train/","/home/thefroggy/Documents/MVA/ObjectRecognition/project/Data/annotations/")
