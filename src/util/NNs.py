import torch
from torch import nn, optim
from torchvision import transforms, models, datasets
from torch.autograd import Variable
from torchvision import utils
import time
import datetime
import os
import sys
import copy
from sklearn import metrics
from relatorios import plot_confusion_matrix, plot_loss_accuracy, generate_classification_report
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import KFold


def initialize_model(model_name, num_classes):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(weights='DEFAULT')
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(weights='DEFAULT')
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(weights='DEFAULT')
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(weights='DEFAULT')
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(weights='DEFAULT')
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(weights='DEFAULT')
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

def evaluate_model(model, dataloader, device):
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true += labels.tolist()
            y_pred += predicted.tolist()

    return y_true, y_pred

def train_model(model, dataloaders, optimizer, basic_parameters, fold, date_now, device='gpu'):# Verifica se a GPU está disponível
   
    
        
        # Tempo total do treinamento (treinamento e validação)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = float('inf')

        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        
        model_name = basic_parameters.get('model_name')
        num_epochs = basic_parameters.get('epochs')
        batch_size = basic_parameters.get('batch_size')

        # Cria a pasta com o nome do modelo
        output_dir = r'outputs/' + model_name
        os.makedirs(output_dir, exist_ok=True)

        # Cria a pasta para separar os testes
        result_dir = output_dir + '\\' + model_name +'_' + date_now
        os.makedirs(result_dir, exist_ok=True)

        # Abre o arquivo para salvar o resultado
        f = open(f'{result_dir}/{model_name}_fold_{fold}.txt', 'w')
        
        for epoch in range(num_epochs):
            f.write(f'Epoch {epoch}/{num_epochs - 1}\n')
            f.write('-' * 10 + '\n')

            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                # Inicia contagem de tempo da época
                time_epoch_start = time.time()

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                # Perda (loss) nesta época
                running_loss = 0.0
                # Amostras classificadas corretamente nesta época
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if model_name == 'inception' and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = basic_parameters.get('criterion')(outputs, labels)
                            loss2 = basic_parameters.get('criterion')(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = basic_parameters.get('criterion')(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Atualiza a perda da época
                    running_loss += loss.item() * inputs.size(0)
                    # Atualiza o número de amostras classificadas corretamente na época.
                    running_corrects += torch.sum(preds == labels.data)
                # Perda desta época
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                # Acurácia desta época
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                # Tempo total desta época
                time_epoch = time.time() - time_epoch_start

                f.write(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ({time_epoch:.4f} seconds) \n')

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ({time_epoch:.4f} seconds)')

                if phase == 'train':
                    train_loss_list.append(epoch_loss)
                    train_acc_list.append(epoch_acc)
                else:
                    val_loss_list.append(epoch_loss)
                    val_acc_list.append(epoch_acc)

                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            time_epoch = time.time() - since

            f.write(f'Time: {time_epoch:.0f}s\n')
            f.write('\n')

            print(f'Time: {time_epoch:.0f}s')
            print('\n')

        time_elapsed = time.time() - since
        f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
        f.write(f'Number of epochs: {num_epochs}. Batch size: {batch_size}\n')
        f.write(f'Best val loss: {best_loss:.4f} Best val acc: {best_acc:.4f}\n')

        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val loss: {best_loss:.4f} Best val acc: {best_acc:.4f}')

        # Save the confusion matrix
        y_true, y_pred = evaluate_model(model, dataloaders['val'], device=device)
        # Confusion matrix
        conf_mat_val = metrics.confusion_matrix(y_true, y_pred)
        f.write(f'\nConfusion Matrix:\n{conf_mat_val}\n')

        # print(f'Confusion Matrix:\n{conf_mat_val}')

        # Classification report 
        class_rep_val = generate_classification_report(model, dataloaders['val'],basic_parameters.get('class_names'), device)    
        f.write(f'\nClassification report:\n{class_rep_val}\n')


        f.close()

        # Save the plot
        plt.figure()
        plot_confusion_matrix(conf_mat_val, classes=basic_parameters.get('class_names'))
        plt.savefig(f'{result_dir}/{model_name}_fold_{fold}_cf_mat.pdf')

        #Plota gráfico de loss e accuracy por epoch
        plot_loss_accuracy(train_loss_list, val_loss_list, train_acc_list, val_acc_list, model_name, fold, result_dir)

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model