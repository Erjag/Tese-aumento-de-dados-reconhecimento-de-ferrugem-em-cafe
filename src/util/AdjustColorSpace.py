import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class AdjustColorSpace:
    def __call__(self, image):
        """
        Converte a imagem para o espaço de cor LAB, ajusta os canais,
        e converte de volta para o espaço RGB.
        print(f"AdjustColorSpace pre augmentation_transforms: {type(image)}")
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Converter a imagem para o espaço de cores LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Separar os canais L, A, B
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Ajustar os canais
        # Aumentar o contraste do canal de luminosidade
        l_channel = cv2.equalizeHist(l_channel)

        # Ajuste opcional: aumentar ou diminuir o contraste de cor nos canais A e B
        a_channel = cv2.add(a_channel, 10)  # Pequeno aumento na tonalidade verde-vermelha
        b_channel = cv2.add(b_channel, 10)  # Pequeno aumento na tonalidade azul-amarela

        # Reunir os canais ajustados
        adjusted_lab_image = cv2.merge((l_channel, a_channel, b_channel))

        # Converter de volta para o espaço de cores RGB
        adjusted_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_LAB2RGB)

        # Converter de volta para PIL antes de retornar
        #print(f"Após augmentation_transforms: {type(adjusted_image)}")
        return Image.fromarray(adjusted_image)
