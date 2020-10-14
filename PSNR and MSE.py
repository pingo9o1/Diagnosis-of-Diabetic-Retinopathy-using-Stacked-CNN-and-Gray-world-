
"""
Here we calculate PSNR and MSE between the compressed image and original image for quailty check and further analysis 

"""



#importing required Libraries 

from math import log10, sqrt
import cv2
import numpy as np

def Peak_signal_to_noise_ratio(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):                               # MSE is zero means no noise is present in the signal . 
                                                 
        return 100
        
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr, mse 
  
def main(): 
     original = cv2.imread("D:/Pingo/Project/VGGGraphgs/advanced-351.png",0)                #Image read as B/W
     Row, Column = original.shape
     compressed = cv2.imread("D:/Pingo/Project/VGGGraphgs/test3cc.png", 0) 
     compressed= cv2.resize(compressed, (Row, Column))                                      #compressed image is resized as original image                                                                                                     
     value1,value2 = PSNR(original, compressed) 
     print(f"PSNR value is {value1} dB and MSE is {value2} ") 
       
if __name__ == "__main__": 
    main() 



#Sanity check for images 

original = cv2.imread("D:/Pingo/Project/VGGGraphgs/advanced-351.png")
compressed = cv2.imread("D:/Pingo/Project/VGGGraphgs/test3cc.png", 1) 

Row, Column = original.shape
compresseds= cv2.resize(compressed, (Row, Column))
img_gray = cv2.cvtColor(compresseds, cv2.COLOR_BGR2GRAY)

#displaying Image for Sanity Check
cv2.imshow('filtered image', img_gray)
cv2.waitKey()
cv2.destroyAllWindows()
