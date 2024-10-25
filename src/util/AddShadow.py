import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import random


#################### Synthetic Shadows ############################################################
class AddShadow:
    def __call__(self, image):
        """
        Adiciona uma sombra sintética à imagem.
        A sombra é criada como um polígono com quatro pontos aleatórios.
        print(f"AddShadow pre augmentation_transforms: {type(image)}")
        """
        
         # Converter para NumPy se necessário
        if isinstance(image, Image.Image):
            #print("[DEBUG] Convertendo PIL para NumPy.")
            image = np.array(image)
        if isinstance(image, torch.Tensor):
            #print("[DEBUG] Convertendo tensor para NumPy.")
            image = image.permute(1, 2, 0).numpy()
        # Garantir que a imagem esteja em formato NumPy correto
        if not isinstance(image, np.ndarray):
            raise TypeError(f"[ERROR] A imagem não é um array NumPy. Tipo: {type(image)}")

        height, width = image.shape[:2]
        
        # Definir quantos vértices o polígono de sombra terá
        top_y = width * random.uniform(0.2, 0.6)
        top_x = 0
        bottom_x = width
        bottom_y = width * random.uniform(0.4, 0.8)

         # Pontos do polígono
        shadow_polygon = np.array(
            [[top_x, top_y], [bottom_x, bottom_y], [bottom_x, height], [top_x, height]],
            dtype=np.int32
        )

         # Criar a máscara
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [shadow_polygon], 255)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Converter a máscara para o mesmo tipo da imagem
        mask = mask.astype(image.dtype)
        # Reduzir a intensidade de brilho nas áreas da sombra
        shadow_intensity = random.uniform(0.5, 0.7)
        shadowed_image = cv2.addWeighted(image, 1 - shadow_intensity, mask, shadow_intensity, 0)

        # Converter para uint8 antes de retornar
        shadowed_image = np.clip(shadowed_image, 0, 255).astype(np.uint8)
        
        return  Image.fromarray(shadowed_image)