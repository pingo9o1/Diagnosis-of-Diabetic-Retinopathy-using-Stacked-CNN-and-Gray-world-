
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




if __name__=="__main__":

    TrainImage = Image.open("Put Image" )
    #TrainImage.show()                                      #Sanity check for image        
    returnPil(GRAYWORLD(frompilImage(img))).show()                 
    
    
    
    
    
    
    
    
    
