import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class AdjustHueSaturation:
    def __init__(self, hue_shift=0, saturation_scale=1.0, brightness_shift=0):
        self.hue_shift = hue_shift
        self.saturation_scale = saturation_scale
        self.brightness_shift = brightness_shift

    def __call__(self, image):
        """
        Ajusta a matiz (hue), a saturação (saturation) e o brilho (brightness) da imagem.
        - hue_shift: Deslocamento da matiz em graus (de -180 a 180).
        - saturation_scale: Fator multiplicativo para a saturação (valores > 1 aumentam a saturação, valores < 1 a diminuem).
        - brightness_shift: Deslocamento do brilho (pode ser positivo ou negativo).
        print(f"AdjustHueSaturation pre augmentation_transforms: {type(image)}")
        """
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        #height, width = image.shape[:2]
        # Converter a imagem para o espaço de cores HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Ajustar a matiz
        hsv_image[:, :, 0] = (hsv_image[:, :, 0].astype(int) + self.hue_shift) % 180
        
        # Ajustar a saturação
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1].astype(float) * self.saturation_scale, 0, 255).astype(np.uint8)
        
        # Ajustar o brilho (canal V no espaço HSV)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2].astype(int) + self.brightness_shift, 0, 255)
        
        # Converter de volta para o espaço de cores RGB
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        """
            # Definir parâmetros para variação de luz e sombra
            hue_shift = random.randint(-15, 15)           # Pequeno deslocamento aleatório de matiz
            saturation_scale = random.uniform(0.7, 1.3)   # Variação na saturação (mais ou menos intensa)
            brightness_shift = random.randint(-50, 50)    # Variação no brilho (mais claro ou mais escuro)

            # Aplicar o ajuste de saturação e matiz
            adjusted_image = adjust_hue_saturation(image_rgb, hue_shift, saturation_scale, brightness_shift)
            print(f"pre augmentation_transforms: {type(adjusted_image)}")
        """
        
        return  Image.fromarray(adjusted_image)