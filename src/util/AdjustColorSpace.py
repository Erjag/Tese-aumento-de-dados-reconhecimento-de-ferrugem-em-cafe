import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

class AdjustColorSpace:
    def __call__(self, image):
        """
        Converte a imagem para o espaço de cor LAB, ajusta os canais,
        e converte de volta para o espaço RGB.
        """
        # Se a imagem for um tensor PyTorch, converta para NumPy e reorganize os eixos
        if isinstance(image, torch.Tensor):
            #print(f"Convertendo tensor para NumPy: shape {image.shape}")
            image = image.permute(1, 2, 0).numpy()  # Converte de (C, H, W) para (H, W, C)

        # Se a imagem for PIL Image, converta para NumPy
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Validar se a imagem é um array NumPy e tem 3 canais
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Imagem inválida: shape {image.shape}. Esperado: (H, W, 3).")

        # Converter para o espaço de cor LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Separar os canais L, A, B
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Ajustar os canais
        # **Garantir que o canal L seja uint8**
        if l_channel.dtype != np.uint8:
            l_channel = (l_channel * 255 / l_channel.max()).astype(np.uint8)

        # Aplicar equalização de histograma no canal L
        l_channel = cv2.equalizeHist(l_channel)
        a_channel = cv2.add(a_channel, 10)  # Ajuste no canal A
        b_channel = cv2.add(b_channel, 10)  # Ajuste no canal B
        #print(l_channel, a_channel, b_channel)
        # Reunir os canais ajustados
        # Verificando se os tamanhos e dtypes são iguais antes de merge
        # Debug: Verificando tipos e formas dos canais
        # Garantir que todos os canais têm o mesmo tipo
        # l_channel = l_channel.astype(np.float32)
        # a_channel = a_channel.astype(np.float32)
        # b_channel = b_channel.astype(np.float32)
        l_channel = l_channel.astype(np.uint8)
        a_channel = a_channel.astype(np.uint8)
        b_channel = b_channel.astype(np.uint8)
        #assert l_channel.shape == a_channel.shape == b_channel.shape, "Os canais L, A, B têm tamanhos diferentes"
        #assert l_channel.dtype == a_channel.dtype == b_channel.dtype, "Os canais L, A, B têm tipos diferentes"

        adjusted_lab_image = cv2.merge((l_channel, a_channel, b_channel))

        # Converter de volta para RGB
        adjusted_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_LAB2RGB)

        # Garantir que o resultado seja compatível com PIL Image
        adjusted_image = Image.fromarray(adjusted_image.astype(np.uint8))
        
        #print(f"Imagem convertida para PIL: {type(adjusted_image)}")
        return transforms.ToTensor()(adjusted_image)
