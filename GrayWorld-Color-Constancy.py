
#Importing Libraries 

import numpy as np
import PIL
from PIL import Image
import cv2

import sys

def frompilImage(pimg):
    nimg = np.asarray(pimg)           //Converted to array Format 
    return nimg

def returnPIL(nimg):
    return Image.fromarray(np.uint8(nimg))


def GRAYWORLD(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])                                              #Channel Average calculated 
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[1] = np.minimum(nimg[1]*(mu_g/np.average(nimg[1])),255)                
    return  nimg.transpose(1, 2, 0).astype(np.uint8)


"""
if __name__=="__main__":

    TrainImage = Image.open("Put Image" )
    #TrainImage.show()                                      #Sanity check for image        
    returnPil(GRAYWORLD(frompilImage(img))).show()                 
    
   """ 
    
    #Running Gray World Algorithm for Multiple Images 


    
 #Here we have looped through all images iterativley and applied grayworld and saved in other location    
    
     
import cv2
import os

def loadImages(path= "D:/Pingo/Project/train1/healthy"):

    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpeg")]
    
    
    
    """
    temp= os.listdir(path)
    images= []
    for i in temp:
        if(i.endswith("jpeg")):
            images.append(i)
    print(images)            #list of all images
    """  

filenames=loadImages()
images= []

for file in filenames:
    images.append(cv2.imread(file))                  #will do in groups 
    
print(images)   
    

num=0

for image in images:
    
    image=to_pil(grey_world(from_pil(image)))             //Grayworld Algorithm
    #image.show()
    cv2.imwrite('D:/Pingo/Project/saved2/unhealthy/'+ str(num) + " .jpeg" , np.float32(image))
    num+=1
    
    
    
    
    
