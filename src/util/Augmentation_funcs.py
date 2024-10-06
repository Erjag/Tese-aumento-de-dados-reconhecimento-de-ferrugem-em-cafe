import os
import sys
import random
import torchvision.transforms as transforms


# Definir uma transformação personalizada para aplicar data augmentation condicionalmente
class ConditionalAugmentation:
    def __init__(self, augmentation_transforms, probability=0.5):
        self.augmentation_transforms = augmentation_transforms
        self.probability = probability

    def __call__(self, img):
        # Aplica o aumento de dados apenas com a probabilidade especificada (50% neste caso)
        if random.random() < self.probability:
            img = self.augmentation_transforms(img)
        return img
    
