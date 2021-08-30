import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from load_images import LoadImages, processSinglePhoto
from keras import models as models
from numpy import array
import numpy as np
import os
from crop_images import crop_image
from skimage.color import rgb2gray
import scipy as sp     
import prepare_files as prp
from PIL import Image
from scipy import stats

def predictClasses(inputImage, model):
    tumors = [0,1,2,3]
    inputImage = inputImage.reshape(224,224,1)
    img = np.array(inputImage).astype('float32')
    image = np.expand_dims(img, axis=0)
    prediction = model.predict(image)
    gliomaResult = round(prediction[0][0],3)
    meningiomaResult = round(prediction[0][1],3)
    pituaryResult = round(prediction[0][2],3)
    noTumorResult = round(prediction[0][3],3)
    preds = [gliomaResult,meningiomaResult,pituaryResult,noTumorResult]
    ind = np.argmax(preds)
    return preds,tumors[ind]

def flip_lr(img,axis):
    return np.flip(img, axis=axis)

def shift(images, shift, axis):
    return np.roll(images, shift, axis=axis)

def rotate(images, angle):
    return sp.ndimage.rotate(
        images, angle, reshape=False)

# def noVoting(tumors,knownClass):
#     tumorHits=0
#     for tumor in tumors:
#         img = tumor.img
#         _,ind = predictClasses(img,model)
#         if ind == knownClass:
#             tumorHits+=1    
#     mean = tumorHits/len(tumors)
#     return mean


def hardVoting(tumors,knownClass, model):
    tumorHits=0
    for tumor in tumors:      
        img = tumor.img
        flippedHor= flip_lr(img,axis=1)
        flippedVer= flip_lr(img,axis=0)
        shiftedLeft= shift(img,-3,0)
        shiftedRight= shift(img,5,0)
        rotated10 = rotate(img,10)
        rotated30 = rotate(img,30)
        _,c = predictClasses(img,model=model)
        _,cFH = predictClasses(flippedHor,model=model)
        _,cFV = predictClasses(flippedVer,model=model)      
        _,cSL = predictClasses(shiftedLeft,model=model)
        _,cSR = predictClasses(shiftedRight,model=model)
        _,cR10 = predictClasses(rotated10,model=model)
        _,cR30 = predictClasses(rotated30,model=model)

        classes = []
        classes.append(c)
        classes.append(cFH)
        classes.append(cFV) 
        classes.append(cSL) 
        classes.append(cSR)       
        classes.append(cR10) 
        classes.append(cR30) 
        
        modeResult = stats.mode(classes)
        mode = modeResult.mode[0]
        if mode == knownClass:
            tumorHits+=1
    acc = tumorHits/len(tumors)
    return acc


def softVoting(tumors,knownClass,model):
    imageHitForInd = 0
    percentegSum = 0
    for tumor in tumors:
        img = tumor.img
        flippedHor= flip_lr(img,axis=1)
        flippedVer= flip_lr(img,axis=0)
        shiftedLeft= shift(img,-3,0)
        shiftedRight= shift(img,5,0)
        rotated10 = rotate(img,10)
        rotated30 = rotate(img,30)

        p,_ = predictClasses(img,model=model)
        pFH,_ = predictClasses(flippedHor,model=model)
        pFV,_ = predictClasses(flippedVer,model=model)      
        pSL,_ = predictClasses(shiftedLeft,model=model)
        pSR,_ = predictClasses(shiftedRight,model=model)
        pR10,_ = predictClasses(rotated10,model=model)
        pR30,_ = predictClasses(rotated30,model=model)

        predictions = [p,pFH,pFV,pSL,pSR,pR10,pR30]
        predictions = np.array(predictions)

        c1 = np.array(predictions[:,0])
        c2 = np.array(predictions[:,1])
        c3 = np.array(predictions[:,2])
        c4 = np.array(predictions[:,3])
        meanC1 = np.mean(c1)
        meanC2 = np.mean(c2)
        meanC3 = np.mean(c3)
        meanC4 = np.mean(c4)
        mean = np.array([meanC1,meanC2,meanC3,meanC4])
        maxInd = np.argmax(mean)

        if maxInd == knownClass:
            imageHitForInd+=1    
            percentegSum += mean[maxInd]

    meanPcc = percentegSum/len(tumors)
    meanInd = imageHitForInd/len(tumors)
    return meanPcc, meanInd

#folder=LoadImages("ImagesOut")
#glioma=folder["glioma"]
# meningioma=folder["meningioma"]
# pituitary=folder["pituitary"]
# no=folder["no"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# gc = prp.fileToClass(glioma,0)
# mc = prp.fileToClass(meningioma,1)
# pc = prp.fileToClass(pituitary,2)
# nc = prp.fileToClass(no,3)

# accNoVotingGlioma = noVoting(tumors=gc,knownClass=0)
# accNoVotingMeningioma = noVoting(tumors=mc,knownClass=1)
# accNoVotingPituitary = noVoting(tumors=pc,knownClass=2)
# accNoVotingNo = noVoting(tumors=nc,knownClass=3)

# accHardVotingGlioma = hardVoting(tumors=gc,knownClass=0)
# accHardVotingMeningioma = hardVoting(tumors=mc,knownClass=1)
# accHardVotingPituitary = hardVoting(tumors=pc,knownClass=2)
# accHardVotingNo = hardVoting(tumors=nc,knownClass=3)

# _,accSoftVotingGlioma = softVoting(tumors=gc,knownClass=0, model= mod)
# _,accSoftVotingMeningioma = softVoting(tumors=mc,knownClass=1)
# _,accSoftVotingPituitary = softVoting(tumors=pc,knownClass=2)
# _,accSoftVotingNo = softVoting(tumors=nc,knownClass=3)

# accNoVoting = (accNoVotingGlioma+accNoVotingMeningioma+accNoVotingPituitary+accNoVotingNo)/4
# accHardVoting = (accHardVotingGlioma+accHardVotingMeningioma+accHardVotingPituitary+accHardVotingNo)/4
# accSoftVoting = (accSoftVotingGlioma+accSoftVotingMeningioma+accSoftVotingPituitary+accSoftVotingNo)/4


# print("No voting",accNoVoting)
# print("Hard voting",accHardVoting)
# print("Soft voting",accSoftVoting)
    
