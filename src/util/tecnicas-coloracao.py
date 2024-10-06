import cv2
import numpy as np
import matplotlib.pyplot as plt

######### transformação de espaço de cor  ########################################################
def adjust_color_space(image):
    """
    Converte a imagem para o espaço de cor LAB, ajusta os canais,
    e converte de volta para o espaço RGB.
    """
    # Converter a imagem para o espaço de cores LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Separar os canais L, A, B
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Ajustar os canais
    # Aumentar o contraste do canal de luminosidade
    l_channel = cv2.equalizeHist(l_channel)
    
    # Ajuste opcional: pode aumentar ou diminuir o contraste de cor nos canais A e B
    a_channel = cv2.add(a_channel, 10)  # Pequeno aumento na tonalidade verde-vermelha
    b_channel = cv2.add(b_channel, 10)  # Pequeno aumento na tonalidade azul-amarela
    
    # Reunir os canais ajustados
    adjusted_lab_image = cv2.merge((l_channel, a_channel, b_channel))
    
    # Converter de volta para o espaço de cores RGB
    adjusted_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_LAB2RGB)
    
    return adjusted_image

# Carregar uma imagem de exemplo
image = cv2.imread('caminho/para/sua/imagem.jpg')

# Converter de BGR (OpenCV) para RGB (Matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Aplicar a transformação de espaço de cor
adjusted_image = adjust_color_space(image_rgb)


