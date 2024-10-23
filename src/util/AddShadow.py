import cv2
import numpy as np
from PIL import Image
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
        
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        height, width = image.shape[:2]
        
        # Definir quantos vértices o polígono de sombra terá
        top_y = width * random.uniform(0.2, 0.6)
        top_x = 0
        bottom_x = width
        bottom_y = width * random.uniform(0.4, 0.8)

        # Definir os pontos para o polígono de sombra
        shadow_polygon = np.array([[top_x, top_y], [bottom_x, bottom_y], [bottom_x, height], [top_x, height]], dtype=np.int32)
        
        # Criar uma máscara para a sombra
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [shadow_polygon], (0, 0, 0))
        
        # Reduzir a intensidade de brilho nas áreas da sombra
        shadow_intensity = random.uniform(0.5, 0.7)
        shadowed_image = cv2.addWeighted(image, 1 - shadow_intensity, mask, shadow_intensity, 0)
        """
        # Carregar uma imagem de exemplo
        image = cv2.imread('caminho/para/sua/imagem.jpg')

        # Converter de BGR (OpenCV) para RGB (Matplotlib)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Aplicar a sombra
        shadowed_image = add_shadow(image_rgb)
        print(f"pos augmentation_transforms: {type(shadowed_image)}")
        """
        return  Image.fromarray(shadowed_image)