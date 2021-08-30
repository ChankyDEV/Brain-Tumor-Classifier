from load_images import LoadImages
from create_model import createModel
from plot_matrix import plot_confusion_matrix
import numpy as np
from numpy import array
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd    
from tumor import Tumor
import prepare_files as prp
import random as r
from test_time_augmentation import hardVoting, softVoting
from keras import models as models


def learnModelWithDataset(glioma,meningioma,pituitary,no,testing,training,model):
    r.shuffle(training)
    r.shuffle(testing)
    
    train = []
    label = []
    for element in training:       
        train.append(element.img)
        label.append(element.label)

    test = []
    label_test = []
    for element in testing:
        test.append(element.img)
        label_test.append(element.label)

    train= array(train).reshape(array(train).shape[0],224,224,1)
    train = np.array(train).astype('float32')

    test= array(test).reshape(array(test).shape[0],224,224,1)
    test = array(test).astype('float32')

    label = np.array(label).astype('float32')
    label = np_utils.to_categorical(label,4)

    label_test = np.array(label_test).astype('float32')
    label_test = np_utils.to_categorical(label_test,4)


    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])



    history = model.fit(train, label,
                        batch_size=32, epochs=10)

    #model.save('tumorClassifier.h5')

    test_loss, test_acc = model.evaluate(test, label_test)
    print('Test accuracy without voting:', test_acc)

    mod = models.load_model('tumorClassifier.h5')

    glioma_acc_ind = hardVoting(tumors=glioma,knownClass=0, model=mod)
    meningioma_acc_ind = hardVoting(tumors=meningioma,knownClass=1, model=mod)
    pituitary_acc_ind = hardVoting(tumors=pituitary,knownClass=2, model=mod)
    no_acc_ind = hardVoting(tumors=no,knownClass=3, model=mod)
    hard_acc = (glioma_acc_ind+meningioma_acc_ind+pituitary_acc_ind+no_acc_ind)/4
    print('Test accuracy with hard voting:', hard_acc)

    _,soft_glioma_acc_ind = softVoting(tumors=glioma,knownClass=0, model=mod)
    _,soft_meningioma_acc_ind = softVoting(tumors=meningioma,knownClass=1, model=mod)
    _,soft_pituitary_acc_ind = softVoting(tumors=pituitary,knownClass=2, model=mod)
    _,soft_no_acc_ind = softVoting(tumors=no,knownClass=3, model=mod)
    soft_acc = (soft_glioma_acc_ind+soft_meningioma_acc_ind+soft_pituitary_acc_ind+soft_no_acc_ind)/4
    print('Test accuracy with soft voting:', soft_acc)


# LOAD FILES
loadedImages=LoadImages("ImagesOut")
gliomasTogether=loadedImages['glioma']
meningiomaTogether=loadedImages['meningioma']
pituaryTogether=loadedImages['pituitary']
noTumorTogether=loadedImages['no']

gliomaObject = prp.fileToClass(gliomasTogether,0)
meningiomaObject = prp.fileToClass(meningiomaTogether,1)
pituaryObject = prp.fileToClass(pituaryTogether,2)
noObject = prp.fileToClass(noTumorTogether,3)

numOfImages = 200
testDatasetPercent = 0.3

GliomaTumorImages,testGliomaTumorImages = prp.getRandomListsByPercent(gliomaObject,testDatasetPercent)
MeningiomaTumorImages,testMeningiomaTumorImages = prp.getRandomListsByPercent(meningiomaObject,testDatasetPercent)
PituitaryTumorImages,testPituitaryTumorImages = prp.getRandomListsByPercent(pituaryObject,testDatasetPercent)
NoTumorImages,testNoTumorImages = prp.getRandomListsByPercent(noObject,testDatasetPercent)

AllImagesTEST = []
AllImagesTEST.extend(testGliomaTumorImages)
AllImagesTEST.extend(testMeningiomaTumorImages)
AllImagesTEST.extend(testPituitaryTumorImages)
AllImagesTEST.extend(testNoTumorImages)


AllImagesTRAIN = []
AllImagesTRAIN.extend(GliomaTumorImages)
AllImagesTRAIN.extend(MeningiomaTumorImages)
AllImagesTRAIN.extend(PituitaryTumorImages)
AllImagesTRAIN.extend(NoTumorImages)


model=createModel()
learnModelWithDataset(testGliomaTumorImages,testMeningiomaTumorImages,testPituitaryTumorImages,testNoTumorImages,AllImagesTEST,AllImagesTRAIN,model)
