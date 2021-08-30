from load_images import LoadImages
import numpy as np
import random as r
from tumor import Tumor



def getRandomLists(data,numberOfImages):
    testing = []
    training = []
    #rangeType = int(len(data)*percent)

    print('Dividing lists....')

    for i in range(numberOfImages):
        move = r.randrange(numberOfImages-i)
        element = data[move]
        data = np.delete(data,move,0)
        testing.append(element)

    training = data
    print('Finish dividing lists....')
    return testing,training

def getRandomListsByPercent(data,percent):
    testing = []
    training = []
    rangeType = int(len(data)*percent)

    for i in range(rangeType):
        move = r.randrange(len(data)-i)
        element = data[move]
        data = np.delete(data,move,0)
        testing.append(element)

    training = data
    print('Finish dividing lists....')
    return training,testing

def getTumorsList(glioma,meningioma,pituary,no):
    tumors = []
    print('Getting tumors....')

    tumors.extend(glioma)
    tumors.extend(meningioma)
    tumors.extend(pituary)
    tumors.extend(no)
    print('Finish getting tumors....')
    return tumors


def fileToClass(files,typeOfCancer):
    cancerClass = []
    for cancer in files:
        cancerClass.append(Tumor(cancer,typeOfCancer))
    return cancerClass