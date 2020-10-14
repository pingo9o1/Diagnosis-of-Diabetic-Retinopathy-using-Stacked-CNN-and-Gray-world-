
#Running Gray World Algorithm for Multiple Images 


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
    
