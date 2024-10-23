import os
import sys
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image  # Import necessário


# Definir uma transformação personalizada para aplicar data augmentation condicionalmente
class ConditionalAugmentation:
  def __init__(self, augmentation_transforms, probability=0.5):
    self.augmentation_transforms = augmentation_transforms
    self.probability = probability

  def __call__(self, img):
    #print(f"ConditionalAugmentation pre augmentation_transforms: {type(img)}")
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if random.random() < self.probability:
      img = self.augmentation_transforms(img)
    # Adicionar log para verificar o tipo
    #print(f"Após augmentation_transforms: {type(img)}")

# Converter para PIL Image no final, se necessário
    if not isinstance(img, Image.Image):
      img = Image.fromarray(np.array(img))
    #print("Forçando conversão para PIL Image.")

    return img
    
