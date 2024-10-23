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
from .relatorios import plot_confusion_matrix, plot_loss_accuracy, generate_classification_report
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

def train_model(model, dataloaders, optimizer, basic_parameters, fold, date_now, device):
    # Definição do diretório de saída
    output_dir = os.path.join("outputs", basic_parameters.get('model_name', ''))
    os.makedirs(output_dir, exist_ok=True)  # Garante que o diretório será criado

    # Cria a pasta para armazenar os resultados de cada teste
    result_dir = os.path.join(output_dir, f"{basic_parameters.get('model_name', '')}_{date_now}")
    os.makedirs(result_dir, exist_ok=True)

    # Abertura de arquivo para registro de resultados
    result_file = os.path.join(result_dir, f"{basic_parameters.get('model_name', '')}_fold_{fold}.txt")
    with open(result_file, 'w') as f:
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
        best_acc = 0.0

        train_loss_list, train_acc_list = [], []
        val_loss_list, val_acc_list = [], []

        for epoch in range(basic_parameters.get('epochs')):
            print(f'Epoch {epoch}/{basic_parameters.get("epochs") - 1}\n{"-" * 10}')
            f.write(f'Epoch {epoch}/{basic_parameters.get("epochs") - 1}\n{"-" * 10}\n')

            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()
                running_loss, running_corrects = 0.0, 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = basic_parameters.get('criterion')(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                f.write(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    train_loss_list.append(epoch_loss)
                    train_acc_list.append(epoch_acc)
                else:
                    val_loss_list.append(epoch_loss)
                    val_acc_list.append(epoch_acc)

                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
        f.write(f'Best val Loss: {best_loss:.4f} Acc: {best_acc:.4f}\n')

        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {best_loss:.4f} Acc: {best_acc:.4f}')

        # Avaliação final e geração da matriz de confusão
        y_true, y_pred = evaluate_model(model, dataloaders['val'], device)
        conf_mat = metrics.confusion_matrix(y_true, y_pred)
        f.write(f'\nConfusion Matrix:\n{conf_mat}\n')

        class_report = generate_classification_report(
            model, dataloaders['val'], basic_parameters.get('class_names'), device
        )
        f.write(f'\nClassification Report:\n{class_report}\n')

    # Salvar gráficos e matriz de confusão
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=basic_parameters.get('class_names'))
    plt.savefig(os.path.join(result_dir, f"{basic_parameters.get('model_name', '')}_fold_{fold}_confusion_matrix.pdf"))

    plot_loss_accuracy(train_loss_list, val_loss_list, train_acc_list, val_acc_list, basic_parameters.get('model_name', ''), fold, result_dir)

    model.load_state_dict(best_model_wts)  # Carregar os melhores pesos do modelo
    return model
