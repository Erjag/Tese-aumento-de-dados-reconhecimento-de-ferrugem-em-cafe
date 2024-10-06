import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

#################### Synthetic Shadows ############################################################
def add_shadow(image):
    """
    Adiciona uma sombra sintética à imagem.
    A sombra é criada como um polígono com quatro pontos aleatórios.
    """
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
    
    return shadowed_image

# Carregar uma imagem de exemplo
image = cv2.imread('caminho/para/sua/imagem.jpg')

# Converter de BGR (OpenCV) para RGB (Matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Aplicar a sombra
shadowed_image = add_shadow(image_rgb)

########################## Ajuste de Saturação e Matiz ###############################################
def adjust_hue_saturation(image, hue_shift=0, saturation_scale=1.0, brightness_shift=0):
    """
    Ajusta a matiz (hue), a saturação (saturation) e o brilho (brightness) da imagem.
    - hue_shift: Deslocamento da matiz em graus (de -180 a 180).
    - saturation_scale: Fator multiplicativo para a saturação (valores > 1 aumentam a saturação, valores < 1 a diminuem).
    - brightness_shift: Deslocamento do brilho (pode ser positivo ou negativo).
    """
    # Converter a imagem para o espaço de cores HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Ajustar a matiz
    hsv_image[:, :, 0] = (hsv_image[:, :, 0].astype(int) + hue_shift) % 180
    
    # Ajustar a saturação
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1].astype(float) * saturation_scale, 0, 255).astype(np.uint8)
    
    # Ajustar o brilho (canal V no espaço HSV)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2].astype(int) + brightness_shift, 0, 255)
    
    # Converter de volta para o espaço de cores RGB
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    return adjusted_image

# Carregar uma imagem de exemplo
image = cv2.imread('caminho/para/sua/imagem.jpg')

# Converter de BGR (OpenCV) para RGB (Matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Definir parâmetros para variação de luz e sombra
hue_shift = random.randint(-15, 15)           # Pequeno deslocamento aleatório de matiz
saturation_scale = random.uniform(0.7, 1.3)   # Variação na saturação (mais ou menos intensa)
brightness_shift = random.randint(-50, 50)    # Variação no brilho (mais claro ou mais escuro)

# Aplicar o ajuste de saturação e matiz
adjusted_image = adjust_hue_saturation(image_rgb, hue_shift, saturation_scale, brightness_shift)

########################## Simulação de Iluminação Direcional ############################################################
def apply_directional_light(image, light_angle=45, light_intensity=0.8):
    """
    Aplica iluminação direcional à imagem.
    - light_angle: O ângulo da fonte de luz em graus (0 a 360). 0 é da esquerda, 90 de cima, 180 da direita, e 270 de baixo.
    - light_intensity: A intensidade da luz, onde 1.0 é a intensidade máxima.
    """
    height, width = image.shape[:2]
    
    # Criar um gradiente linear que simula a iluminação direcional
    # Calcular as coordenadas da direção da luz
    angle_rad = np.deg2rad(light_angle)
    x_light = np.cos(angle_rad)
    y_light = np.sin(angle_rad)
    
    # Criar uma máscara que irá armazenar os valores de iluminação
    light_mask = np.zeros_like(image, dtype=np.float32)
    
    # Preencher a máscara com o gradiente de iluminação
    for i in range(height):
        for j in range(width):
            distance = (x_light * (j - width // 2) + y_light * (i - height // 2))
            light_mask[i, j] = 1 - (distance / (np.sqrt(width**2 + height**2)) * light_intensity)
    
    # Aplicar a máscara de iluminação à imagem
    light_mask = np.clip(light_mask, 0, 1)
    lighted_image = (image.astype(np.float32) * light_mask).astype(np.uint8)
    
    return lighted_image

# Carregar uma imagem de exemplo
image = cv2.imread('caminho/para/sua/imagem.jpg')

# Converter de BGR (OpenCV) para RGB (Matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Definir parâmetros para simulação de iluminação direcional
light_angle = 120  # Ângulo da luz vindo da direita-superior
light_intensity = 0.8  # Intensidade da luz

# Aplicar a iluminação direcional
lighted_image = apply_directional_light(image_rgb, light_angle, light_intensity)

