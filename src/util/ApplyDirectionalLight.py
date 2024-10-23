import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ApplyDirectionalLight:
    def __init__(self, light_angle=45, light_intensity=0.8):
        self.light_angle = light_angle
        self.light_intensity = light_intensity

    def __call__(self, image):
        """
        Aplica iluminação direcional à imagem.
        - light_angle: O ângulo da fonte de luz em graus (0 a 360). 0 é da esquerda, 90 de cima, 180 da direita, e 270 de baixo.
        - light_intensity: A intensidade da luz, onde 1.0 é a intensidade máxima.
        
        print(f"ApplyDirectionalLight pre augmentation_transforms: {type(image)}")
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        height, width = image.shape[:2]
        
        # Criar um gradiente linear que simula a iluminação direcional
        # Calcular as coordenadas da direção da luz
        angle_rad = np.deg2rad(self.light_angle)
        x_light = np.cos(angle_rad)
        y_light = np.sin(angle_rad)
        
        # Criar uma máscara que irá armazenar os valores de iluminação
        light_mask = np.zeros_like(image, dtype=np.float32)
        
        # Preencher a máscara com o gradiente de iluminação
        for i in range(height):
            for j in range(width):
                distance = (x_light * (j - width // 2) + y_light * (i - height // 2))
                light_mask[i, j] = 1 - (distance / (np.sqrt(width**2 + height**2)) * self.light_intensity)
        
        # Aplicar a máscara de iluminação à imagem
        light_mask = np.clip(light_mask, 0, 1)
        lighted_image = (image.astype(np.float32) * light_mask).astype(np.uint8)
        
        """
            # Carregar uma imagem de exemplo
            image = cv2.imread('caminho/para/sua/imagem.jpg')

            # Converter de BGR (OpenCV) para RGB (Matplotlib)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Definir parâmetros para simulação de iluminação direcional
            light_angle = 120  # Ângulo da luz vindo da direita-superior
            light_intensity = 0.8  # Intensidade da luz

            # Aplicar a iluminação direcional
            lighted_image = apply_directional_light(image_rgb, light_angle, light_intensity)
        print(f"pos augmentation_transforms: {type(lighted_image)}")
        """
        return Image.fromarray(lighted_image)