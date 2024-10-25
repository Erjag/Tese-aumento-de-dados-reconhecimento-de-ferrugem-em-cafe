import os
import sys
current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)
sys.path.append(os.path.abspath("../util"))
import random
import time
import datetime
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms, models, datasets
from torch.autograd import Variable
from torchvision import utils
from sklearn import metrics
from sklearn.model_selection import KFold
import gc 
from util.NNs import initialize_model, train_model
from util.relatorios import plot_confusion_matrix, plot_loss_accuracy, generate_classification_report
from util.Augmentation_funcs import  ConditionalAugmentation
from util.AdjustColorSpace import *
from util.AddShadow import *
from util.AdjustHueSaturation import *
from util.ApplyDirectionalLight import *
#import splitfolders
import gc
import torch
import math
import cv2
import traceback


torch.cuda.empty_cache()
# Verifique se a GPU está disponível
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gc.collect()

def control_prob(augmentations, probabilities):
    if len(augmentations) != len(probabilities):
        raise ValueError("O número de augmentações e probabilidades deve ser o mesmo.")

    def apply_transform(img):
        for aug, prob in zip(augmentations, probabilities):
            if random.random() < prob:
                img = aug(img)  # Aplica a transformação
        return img

    return apply_transform
def ensure_tensor(img):
    """Converte para tensor se ainda não for."""
    if isinstance(img, Image.Image):
        return transforms.ToTensor()(img)
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:  # Imagem em escala de cinza (H, W)
            img = np.expand_dims(img, axis=-1)  # (H, W) -> (H, W, 1)
        return torch.from_numpy(img).permute(2, 0, 1).float()   # (H, W, C) -> (C, H, W)
    elif isinstance(img, torch.Tensor):
        return img
    else:
            raise TypeError(f"[ERROR] Tipo inesperado: {type(img)}")

def create_data_transforms(aug_type,input_size):
    train_transforms = None
    aug = [
    "Sem Augmentação",                # Nenhuma transformação aplicada
    "Base + Augmentação Básica",       # Augmentação básica adicionada à base
    "Base + Augmentação Avançada",     # Augmentação avançada adicionada à base
    "Somente Augmentação Básica",      # Apenas augmentação básica aplicada
    "Somente Augmentação Avançada",    # Apenas augmentação avançada aplicada
    "Augmentação Básica + Avançada"    # Ambas augmentações aplicadas juntas
    ]
    print(f"Augmentation {aug_type}: {aug[int(aug_type)]}")
 
    base_augmentation = transforms.Compose([
        transforms.Resize(input_size),  # Redimensiona para o input_size
        transforms.CenterCrop(input_size),  # Faz um crop central para garantir o tamanho correto # Converte para tensor
        ensure_tensor,
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normaliza com base no ImageNet
    ])
    basic_augmentation = transforms.Compose([  
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translação
        transforms.RandomRotation(30),
        ensure_tensor
    ])
    advanced_augmentation = transforms.Compose([
        AdjustColorSpace(),
        AddShadow(),
        AdjustHueSaturation(hue_shift=10, saturation_scale=1.2, brightness_shift=20),
        ApplyDirectionalLight(light_angle=120, light_intensity=0.9),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        ensure_tensor
    ])
    augmentations = [basic_augmentation, advanced_augmentation]
    probabilities = [0.5, 0.5]
    if aug_type == '0':
        train_transforms = transforms.Compose([
                base_augmentation                
            ])
    if aug_type == '1':
        random_augment = control_prob([basic_augmentation], [0.5])
        train_transforms = transforms.Compose([
                        base_augmentation,
                        control_prob([basic_augmentation], [0.5])
                        #ConditionalAugmentation(basic_augmentation, probability=0.5),
                    ])
    if aug_type == '2':
        random_augment = control_prob([advanced_augmentation], [0.5])
        train_transforms = transforms.Compose([
                        base_augmentation,
                        random_augment
                        #ConditionalAugmentation(advanced_augmentation, probability=0.5)
                    ])
    if aug_type == '3':
        train_transforms = transforms.Compose([
                        base_augmentation,
                        basic_augmentation
                    ])
    if aug_type == '4':
        train_transforms = transforms.Compose([
                        base_augmentation,
                        advanced_augmentation
                    ])
    if aug_type == '5':
        random_augment = control_prob(augmentations, probabilities)
        train_transforms = transforms.Compose([
                        base_augmentation,
                        random_augment
                    ])
    data_transforms = {
                    'train': train_transforms,
                    'val': transforms.Compose([
                        base_augmentation
                    ]),
                }    
    return data_transforms


MAGNIFICATION_SCALE = {
    "20.0": 1.0,
    "10.0": 2.0,
    "5.0": 4.0,
    "2.5": 8.0,
    "1.25": 16.0,
    "0.625": 32.0,
    "0.3125": 64.0,
    "0.15625": 128.0,
    "0.078125": 256.0
}
def get_scale_by_magnification(magnification):
    return MAGNIFICATION_SCALE[str(magnification)]


def main():
    #dataset_dir = "C:/Users/Augusto/Documents/tese/Tese-aumento-de-dados-reconhecimento-de-ferrugem-em-cafe/files/imagens/Photos"
    #model_dir = "C:/Users/Augusto/Documents/tese/Tese-aumento-de-dados-reconhecimento-de-ferrugem-em-cafe/files/classificadores"
    #color_model = "LAB"
    magnification = 0.625
    scale = get_scale_by_magnification(magnification)
    tile_size = 20
    tile_size_original = int(scale * tile_size)
    patch_size = (tile_size_original, tile_size_original)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('\nDevice: {0}'.format(device))
    #print(torch.cuda.get_device_name(0))
    # !nvidia-smi
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # Define o caminho base do diretório de imagens divididas 
    #base_dir = r'C:\Users\Augusto\Documents\tese\files\imagens\Photos'
    base_dir = r"C:/Users/Augusto/Documents/tese/Tese-aumento-de-dados-reconhecimento-de-ferrugem-em-cafe/files/imagens/Photos"
    for model in ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet']:#['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']:   
        model_since = time.time()
        basic_parameters = {
        #'num_classes' : 2,
        #'class_names': ['healthy', 'unhealthy'],
        'num_classes' : 6,
        'class_names': ['healthy','red_spider_mite', 'rust_level_1','rust_level_2','rust_level_3','rust_level_4'],
        'batch_size' : 32,
        'lr' : 0.001, # Taxa de aprendizado
        'mm' : 0.9, # Mommentum
        'epochs' : 1,
        'model_name' : model, # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
        'criterion' : nn.CrossEntropyLoss(), # Função de perda
        'data_augmentation' :['2','3','4','5'] #'0','1','2','3','4','5']
        #['0','1','2','3'] # 0 - None, 1 - none + aug-basic, 2 - none + aug-avanced, 3 -aug-basic, 4 - aug-avanced, 5 - aug-basic + aug-avanced
        }
        model_ft, input_size = initialize_model(basic_parameters.get('model_name'), basic_parameters.get('num_classes'))
        # Definir as transformações básicas
        
        for type_aug in basic_parameters.get('data_augmentation'):
            type_aug_since = time.time()
            # Obtenção do parâmetro isAugment a partir de basic_parameters
            # Configuração do pipeline de transformações com base em isAugment
            data_transforms = create_data_transforms(type_aug,input_size)

            image_datasets = datasets.ImageFolder(base_dir, data_transforms['train'])
            # # Pretrainned
            # model_ft = model_ft.to(device)
            date_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            #aumentar numeros de splits depois
            kf = KFold(n_splits=2, shuffle=True, random_state=SEED)
            folds = kf.split(image_datasets)
            try:
                for fold, (train_idx, val_idx) in enumerate(folds):            
                    model = basic_parameters.get('model_name')
                    print(f'FOLD {fold}, Model: {model}')
                    
                    train_dataset = torch.utils.data.Subset(image_datasets, train_idx)
                    val_dataset = torch.utils.data.Subset(image_datasets, val_idx)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=basic_parameters.get('batch_size'), shuffle=True, num_workers=0,pin_memory=torch.cuda.is_available())
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=basic_parameters.get('batch_size'), shuffle=False , num_workers=0,pin_memory=torch.cuda.is_available())
                    
                    dataloaders_dict = {'train': train_loader, 'val': val_loader}

                    # Reiniciar o modelo em cada fold
                    #model_ft, input_size = initialize_model(basic_parameters.get('model_name'), basic_parameters.get('num_classes'))
                    
                    model_ft = model_ft.to(device)
                    # Imprime o modelo
                    # print(f'Model: {str(model_ft)}')
                    # Otimizador
                    optimizer = optim.SGD(model_ft.parameters(), lr=basic_parameters.get('lr'), momentum=basic_parameters.get('mm'))  
                        
                    model_ft = train_model(model_ft, dataloaders_dict, optimizer, basic_parameters, fold,date_now, device)
                    model_ft.eval()
                    print(f'\n{"-" * 50}\n')
                    break
            except Exception as e:
                print(f"Erro: {e}")
                traceback.print_exc()
            type_aug_time = time.time() - type_aug_since
            print(f"Tempo para Augmentation {type_aug}: {type_aug_time // 60:.0f}m {type_aug_time % 60:.0f}s\n{'-' * 50}\n")
        model_time = time.time() - model_since
        print(f"Tempo total para {model}: {model_time // 60:.0f}m {model_time % 60:.0f}s\n{'-' * 50}\n")
        print(f'\n{"-" * 50}\n')

                    
                
                
if __name__ == "__main__":
    main()         