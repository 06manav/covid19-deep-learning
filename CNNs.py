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

#!

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    histData = np.zeros((7,num_epochs))
    mean_fpr = np.linspace(0, 1, 100)
    confusion_matrices = np.zeros((num_epochs,3,3))
    interp_tprs = np.zeros((num_epochs,100))
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            outputs_all = np.array([]); labels_all = np.array([]);
            all_preds = torch.tensor([]).to(device)
            all_labels = torch.tensor([]).to(device)
            outputs_all_for_scores = np.array([])
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        labels = labels.long()
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        labels = labels.long()
                        
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                all_preds = torch.cat((all_preds, outputs),dim=0)
                all_labels = torch.cat((all_labels, labels.long()),dim=0)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                outputs_all = np.append(outputs_all,outputs.cpu().detach().numpy())
                labels_all = np.append(labels_all,labels.cpu().detach().numpy())
                outputs_all_for_scores = np.append(outputs_all_for_scores, preds.cpu().detach().numpy())
            
            labels_all_for_scores = labels_all.reshape(np.prod(labels_all.shape))
            labels_all = label_binarize(labels_all, classes=[0, 1, 2])
            outputs_all = outputs_all.reshape(labels_all.shape[0],3)
            F1_score = f1_score(labels_all_for_scores, outputs_all_for_scores, labels=[0, 1, 2],average="macro")
            prec_score = precision_score(labels_all_for_scores, outputs_all_for_scores, average="macro")
            rec_score = recall_score(labels_all_for_scores, outputs_all_for_scores, average="macro")
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            stacked = torch.stack((all_labels, all_preds.argmax(dim=1)),dim=1).long()
            
            num_classes = 3
            cmt = torch.zeros(num_classes,num_classes, dtype=torch.int64)
            for p in stacked:
                tl, pl = p.tolist()
                cmt[tl, pl] = cmt[tl, pl] + 1
            cmt = cmt.detach().numpy()
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                histData[2,epoch] = epoch_acc
                histData[3,epoch] = epoch_loss
                histData[4,epoch] = F1_score
                histData[5,epoch] = prec_score
                histData[6,epoch] = rec_score
                confusion_matrices[epoch] = cmt;
                
                fpr, tpr, roc_auc = getPRs(labels_all, outputs_all)
                interp_tpr = np.interp(mean_fpr, fpr['micro'], tpr['micro'])
                interp_tpr[0] = 0.0
                interp_tprs[epoch] = interp_tpr;
            else:
                histData[0,epoch] = epoch_acc
                histData[1,epoch] = epoch_loss
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, histData, interp_tprs, confusion_matrices
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def predict(batch, model):
    with torch.no_grad():
        out = model(batch)
        print(out[0])
        _, predicted = torch.max(out, 1)
        predicted = predicted.numpy()
    return predicted

def getModel(lr,momentum):
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

    ################################################
    # data
    ########
    # Create training and validation datasets


    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=momentum)#,nesterov=True)#######################################
    #optimizer_ft = optim.Adam(params_to_update, lr=0.0005)
    return model_ft, optimizer_ft   

def oneClassProblem(y_score, positive_class):
    y_score = y_score[:, positive_class]
    return y_score

def getPRs(y_test, y_hat):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_hat[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_hat.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc
        
                       
# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./data/hymenoptera_data"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# inception has 299x299 images as input, so  images should be preprocessed differently
model_name = 'inception'

# Number of classes in the dataset
num_classes = 3

# Batch size for training (change depending on how much memory you have)
batch_size = 20

# Number of epochs to train for
num_epochs = 20

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
############################

#optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
#optimizer_ft = optim.Adam(params_to_update, lr=0.001)
###################################################

num_folds = 5
# Define the K-fold Cross Validator
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)

#loading path with preprocessed images
loading_path = ''
datasetX = load(str(loading_path) + 'datasetX.npy')
datasetY = load(str(loading_path) + 'datasetY.npy')

print(datasetX.shape,datasetY.shape)
img_width, img_height = datasetX.shape[2:4]
datasetX = datasetX.reshape(3*219,img_width,img_height,3)
datasetX = np.transpose(datasetX, (0,3, 1, 2))

datasetX = datasetX / 1.0 + 0.00 #####################################################################################################
#plt.imshow(datasetX[0][0])
#plt.show()
datasetY = datasetY.reshape(3*219)
print(datasetX.shape,datasetY.shape)
indx=np.arange(len(datasetX))          # create a array with indexes for X data
for i in range(3):
    np.random.shuffle(indx)

datasetX_=datasetX[indx]
datasetY_=datasetY[indx]

datasetX = datasetX_[:-int(len(datasetX)*0.2)]
datasetY = datasetY_[:-int(len(datasetY)*0.2)]

datasetX_test = datasetX_[-int(len(datasetX)*0.2):]
datasetY_test = datasetY_[-int(len(datasetY)*0.2):]

modelsParameters = []
# arrays to store all the data through the greed search
histData_greed_search = []
interp_tprs_greed_search = []
confusion_matrices_greed_search = []

#params for greed search
numOfBatches = [8,22];
learningRates = [0.0005,0.001]
momentums = [0.9,0.95]
for batch_size in numOfBatches:
    for lr in learningRates:
        for momentum in momentums:
            print('#'*70,'\n','current params: {}, {}, {}'.format(batch_size,lr,momentum)); print('#'*70)
            modelsParameters.append([batch_size,lr,momentum])
            histData_cv = []
            interp_tprs_cv = []
            confusion_matrices_cv = []
            # Define the K-fold Cross Validator
            kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
            for trainIndex, valIndex in kfold.split(datasetX, datasetY):
                tensor_x = torch.Tensor(datasetX[trainIndex]) # transform to torch tensor
                tensor_x = tensor_x.to(device)
                tensor_y = torch.Tensor(datasetY[trainIndex])
                tensor_y = tensor_y.to(device)

                my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
                my_dataloader_train = DataLoader(my_dataset,batch_size=batch_size)#,drop_last=True) # create your dataloader
                
                tensor_x = torch.Tensor(datasetX[valIndex]) # transform to torch tensor
                tensor_x = tensor_x.to(device)
                tensor_y = torch.Tensor(datasetY[valIndex])
                tensor_y = tensor_y.to(device)

                my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
                my_dataloader_val = DataLoader(my_dataset,batch_size=batch_size)#,drop_last=True) # create your dataloader
                #print("#### ", my_dataset.type())
                my_dataloader = {'train' : my_dataloader_train, 'val' : my_dataloader_val}
                
                # Setup the loss fxn
                criterion = nn.CrossEntropyLoss()
                ### Initialize the model for this run
                model_ft, optimizer_ft = getModel(lr, momentum)
                ###
                # Train and evaluate
                model, val_acc_history, histData, interp_tprs, confusion_matrices = train_model(model_ft, \
                                                             my_dataloader, criterion, optimizer_ft, num_epochs=num_epochs, \
                                                                                    is_inception = (model_name=="inception"))
                histData_cv.append(histData)
                interp_tprs_cv.append(interp_tprs)
                confusion_matrices_cv.append(confusion_matrices)
            histData_greed_search.append(histData_cv)
            interp_tprs_greed_search.append(interp_tprs_cv)
            confusion_matrices_greed_search.append(confusion_matrices_cv)

#path for saving history of the greed search 
saving_path = ''
np.save(saving_path + model_name + '_modelsParameters.npy', np.array(modelsParameters))
np.save(saving_path + model_name + '_histData_greed_search.npy', np.array(histData_greed_search))
np.save(saving_path + model_name + '_interp_tprs_greed_search.npy', np.array(interp_tprs_greed_search))
np.save(saving_path + model_name + '_confusion_matrices_greed_search.npy', np.array(confusion_matrices_greed_search))


'''    
plt.plot(histData[0], label = 'train accuracy')
plt.plot(histData[1], label = 'train loss')
plt.plot(histData[2], label = 'validation accuracy')
plt.plot(histData[3], label = 'validation loss')
plt.plot(histData[4], label = 'validation f1_score')
plt.plot(histData[5], label = 'validation precision')
plt.plot(histData[6], label='validation recall')
plt.xlabel('Epoch')
plt.ylabel('Accuracy measure')
plt.legend(loc='upper right')     
plt.show()
print(interp_tprs[0])    
#mean_tpr = np.mean(interp_tprs, axis=0)
mean_tpr = interp_tprs[-1]
mean_tpr[-1] = 1.0
mean_fpr = np.linspace(0, 1, 100)
mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
plt.figure(num_epochs)
plt.plot(mean_fpr, mean_tpr, color='b',
    label=r'Mean ROC (AUC = %0.2f)' % (mean_auc),
    lw=2, alpha=.8)   
plt.show()
print(confusion_matrices[0])
print(confusion_matrices[-1])
'''

'''                                                                  
model_ft.eval()

tensors = [torch.from_numpy(x) for x in datasetX_test]
tensors = [x.float() for x in tensors]
image_batch = torch.stack(tensors)
yhat = predict(image_batch, model_ft)
total_acc = np.sum(datasetY_test == yhat)
final_train_acc = total_acc/len(datasetX_test) 
print('########\n\n',  final_train_acc, '\n\n') 
print(yhat)

# Compute ROC curve and ROC area for each class
yhat = model_ft(image_batch)
y_hat = yhat.detach().numpy()

y_test = label_binarize(datasetY_test, classes=[0, 1, 2])

#Compute ROC curve and ROC area for each class

fpr, tpr, roc_auc = getPRs(y_test, y_hat)'''


'''

# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()'''

                                                  
    #######################
