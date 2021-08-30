import os
import cv2
from crop_images import crop_image
from skimage.color import rgb2gray


def LoadImages(foldername):

    image_dict={}
    
    for root, _, files in os.walk(foldername):  
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        images=[]
        image_dict[os.path.basename(root)]=images
        
        for file in files:           
            img_data =cv2.imread(root+"/"+ file)
            imageGrey=processSinglePhoto(img_data)
            images.append(imageGrey)
            
    return image_dict


def processSinglePhoto(img):
    imageCropped=crop_image(img)
    imageResized=cv2.resize(imageCropped,(224,224),interpolation=cv2.INTER_CUBIC)    
    imageGrey=rgb2gray(imageResized)
    return imageGrey    