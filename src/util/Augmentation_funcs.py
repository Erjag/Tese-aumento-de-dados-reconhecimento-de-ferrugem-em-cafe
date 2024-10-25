import os
import sys
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image  # Import necessário


# Definir uma transformação personalizada para aplicar data augmentation condicionalmente
class ConditionalAugmentation:
  def __init__(self, augmentation_transforms, probability=0.5):
    self.augmentation_transforms = augmentation_transforms
    self.probability = probability

  def __call__(self, img):
    """
    Função para aplicar transformações de aumento de dados com validações e mensagens de debug detalhadas.
    """
    try:
        should_apply = random.random() < self.probability
        # #print(f"[DEBUG][IN] class: {self.__class__.__name__}")
        # if isinstance(img, torch.Tensor):
        #   #print(f"  [DEBUG] Imagem é um Tensor: shape={img.shape}, dtype={img.dtype}")
        #   img = transforms.ToPILImage()(img)
        # if isinstance(img, np.ndarray):
        #   #print(f"  [DEBUG] Imagem é um NumPy array: shape={img.shape}, dtype={img.dtype}")
        #   img = transforms.ToPILImage()(img) 
        # #validar_tipo_img(img, Image.Image)

        # if img.ndim == 3 and img.shape[0] == 1:
        #     print(f"    [DEBUG] Removendo dimensão extra: shape original {img.shape}")
        #     img = img.squeeze(0)  # Remover a primeira dimensão
        #     print(f"    [DEBUG] Nova shape após squeeze: {img.shape}")

        # Aplicar a transformação com uma probabilidade definida
        if should_apply:
            #print(f"    [DEBUG] Aplicando transformação com probabilidade {self.probability}")
            #tensor_img = torch.from_numpy(img).permute(2, 0, 1)
            img = transforms.ToPILImage()(self.augmentation_transforms(img))
            #print(f"    [DEBUG] Finalizado")

        # Converter para uint8 para garantir compatibilidade com PIL
        # if img.dtype != np.uint8:
        #     #print(f"[DEBUG] Convertendo dtype de {img.dtype} para uint8")
        #     self.log_info("Convertendo dtype para uint8", img)
        #     img = img.astype(np.uint8)

        # Imprimir tipo e forma antes de converter para PIL
        #print(f"[DEBUG] Antes da conversão pa
        # ra PIL: shape {img.shape}, dtype {img.dtype}")

        # Converter para PIL Image
        #print(f"    [DEBUG] should:{should_apply}")
        # if should_apply:
        #   validar_tipo_img(img, np.ndarray)
        # else:
        #    validar_tipo_img(img, Image.Image)
        
        # if isinstance(img, Image.Image):
        #   print("[DEBUG] Convertendo PIL para Tensor.")
        #   img = transforms.ToTensor()(img)
        #validar_tipo_img(img, Image.Image)
        #print(f"[DEBUG] Após conversão para PIL: {type(img)}")

    except Exception as e:
        self.log_error(e, img)
        #print(f"[ERROR][OUT] class: {self.__class__.__name__}")
        #print(f"[ERROR] Ocorreu um erro na transformação: {str(e)}")
        #print(f"[ERROR] Tipo da imagem: {type(img)}, Shape: {getattr(img, 'shape', 'Desconhecido')}, Dtype: {getattr(img, 'dtype', 'Desconhecido')}")
    #print(f"[DEBUG][OUT] class: {self.__class__.__name__}")
    return img

  def log_info(self, mensagem, img):
    """Imprime logs de informações sobre a imagem."""
    tipo = type(img)
    #print(f"    [INFO] {mensagem}: Tipo={tipo}")

  def log_error(self, e, img):
    """Imprime logs de erro detalhados."""
    tipo = type(img)
    #print(f"    [ERROR] Erro na transformação: {str(e)}")
    #print(f"    [ERROR] Tipo={tipo}")

def validar_tipo_img(img,expected_type):
    # Verificar se o tipo é PIL.Image.Image e imprimir o resultado
    #print(f"    [DEBUG] Verificando tipo da imagem...")
    if isinstance(img, expected_type):
        print(f"    [DEBUG][match] Tipo={type(img)}, esperado: {expected_type}")
    else:
        print(f"    [DEBUG][unmatch] Tipo={type(img)}, esperado: {expected_type}")
