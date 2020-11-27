from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import seaborn as sn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold

import matplotlib.pyplot as plt
from numpy import save
from numpy import load
import sklearn
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


#path for loading history of the greed search
model_name = 'resnet' 
saving_path = '/home/kons/workspace/data_analytics/code/project/' + model_name + '/'

modelsParameters = load(saving_path + model_name + '_modelsParameters.npy')
histData_greed_search = load(saving_path + model_name + '_histData_greed_search.npy')
interp_tprs_greed_search = load(saving_path + model_name + '_interp_tprs_greed_search.npy')
confusion_matrices_greed_search = load(saving_path + model_name + '_confusion_matrices_greed_search.npy')

print(histData_greed_search.shape)

histData_averged = np.zeros((histData_greed_search.shape[0],histData_greed_search.shape[2], \
                                                                    histData_greed_search.shape[3]))
for i in range(len(histData_averged)):
    for cvI in range(histData_greed_search.shape[1]):
        for paramI in range(histData_greed_search.shape[2]):
            for epochI in range(histData_greed_search.shape[3]):
                histData_averged[i,paramI,epochI] += histData_greed_search[i,cvI,paramI,epochI]
histData_averged /= histData_greed_search.shape[1]

bestAccuracy = 0
worstAccuracy = 1
for i in range(len(histData_averged)):
    for epochI in range(histData_greed_search.shape[3]):
        valAccuracy = histData_averged[i, 2, epochI]
        if valAccuracy > bestAccuracy:
            bestAccuracy = valAccuracy
            bestParams = modelsParameters[i]
            bestParams = np.append(bestParams,epochI)
            bestSetOfParamsI = i
        if valAccuracy < worstAccuracy:
            worstAccuracy = valAccuracy
            worstParams = modelsParameters[i]
            worstParams = np.append(worstParams,epochI)
            worstSetOfParamsI = i
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

print([bestSetOfParamsI,bestParams[-1]])
result = np.append([bestSetOfParamsI],bestParams)
result = np.append([result], histData_averged[bestSetOfParamsI,:,int(bestParams[-1])])

print(bestAccuracy, bestParams)
print(worstAccuracy, worstParams)

# [bestSetOfParamsI, bestParams, evaluation]
np.savetxt(saving_path+"bestModel_"+model_name+".csv", result,delimiter=",",fmt="%.4f")
#       [сохранить лучшие показатели для модельки (+параметры модельки), график модели без тьюна, график с тьюном, график roc, conf_matrix]

plt.plot(histData_averged[bestSetOfParamsI,0], label = 'train accuracy')
plt.plot(histData_averged[bestSetOfParamsI,1], label = 'train loss')
plt.plot(histData_averged[bestSetOfParamsI,2], label = 'validation accuracy')
plt.plot(histData_averged[bestSetOfParamsI,3], label = 'validation loss')
plt.plot(histData_averged[bestSetOfParamsI,4], label = 'validation f1_score')
plt.plot(histData_averged[bestSetOfParamsI,5], label = 'validation precision')
plt.plot(histData_averged[bestSetOfParamsI,6], label='validation recall')
plt.xlabel('Epoch')
plt.ylabel('Accuracy measure')
plt.legend(loc='upper right') 

axes = plt.gca()
axes.set_ylim([0,1.1])
plt.xticks(range(0,histData_greed_search.shape[-1],2))  
plt.savefig(saving_path + "bestModel_" + model_name + '.png')
plt.show()

plt.plot(histData_averged[worstSetOfParamsI,0], label = 'train accuracy')
plt.plot(histData_averged[worstSetOfParamsI,1], label = 'train loss')
plt.plot(histData_averged[worstSetOfParamsI,2], label = 'validation accuracy')
plt.plot(histData_averged[worstSetOfParamsI,3], label = 'validation loss')
plt.plot(histData_averged[worstSetOfParamsI,4], label = 'validation f1_score')
plt.plot(histData_averged[worstSetOfParamsI,5], label = 'validation precision')
plt.plot(histData_averged[worstSetOfParamsI,6], label='validation recall')
plt.xlabel('Epoch')
plt.ylabel('Accuracy measure')
plt.legend(loc='upper right') 

axes = plt.gca()
axes.set_ylim([0,1.1])
plt.xticks(range(0,histData_greed_search.shape[-1],2))  
plt.savefig(saving_path + "worstModel_" + model_name + '.png')
plt.show()

print(interp_tprs_greed_search.shape)

bestModelTpr = np.zeros(interp_tprs_greed_search.shape[-1])
for cvI in range(interp_tprs_greed_search.shape[1]):
    for fprI in range(interp_tprs_greed_search.shape[-1]):
        bestModelTpr[fprI] += interp_tprs_greed_search[bestSetOfParamsI,cvI,int(bestParams[-1]),fprI]
bestModelTpr /= interp_tprs_greed_search.shape[1]
bestModelTpr[-1] = 1.0
mean_fpr = np.linspace(0, 1, 100)
mean_auc = auc(mean_fpr, bestModelTpr)
plt.plot(mean_fpr, bestModelTpr, color='b',
    label=r'Mean ROC (AUC = %0.2f)' % (mean_auc),
    lw=2, alpha=.8) 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')    
plt.savefig(saving_path + "bestModel_ROC_" + model_name + '.png')     
plt.show()
np.save(saving_path + model_name + '_bestModelTpr.npy', bestModelTpr)

print(confusion_matrices_greed_search.shape)
sumOfConfMatrices = np.zeros(confusion_matrices_greed_search.shape[-2:])
sums = 0
for cvI in range(confusion_matrices_greed_search.shape[1]):
    sumOfConfMatrices += confusion_matrices_greed_search[bestSetOfParamsI,cvI,int(bestParams[-1]),:,:]
sumOfConfMatrices = sumOfConfMatrices.astype('int')
print(sumOfConfMatrices)
print(np.sum(sumOfConfMatrices))
class_names = ['COVID-19','Viral Pneumonia', 'Normal']
df_cm = pd.DataFrame(sumOfConfMatrices, index=class_names, columns=class_names)
# plt.figure(figsize=(10,7))
sn.set(font_scale=1) # for label size
sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 15}) # font size
plt.subplots_adjust(left=0.23, bottom=0.13) 
plt.savefig(saving_path + "bestModel_confusion_matrix_" + model_name + '.png')   
plt.show()

