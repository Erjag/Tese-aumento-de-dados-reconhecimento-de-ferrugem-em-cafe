import os
import pandas as pd
import shutil


num_classes = 6

class sort_subDir_classes:
    def num_classes(num_class):
     num_classes = num_class

# Definir o caminho do arquivo RoCoLe-classes e das fotos
path_classes = r"C:\Users\Augusto\Documents\tese\files\classificadores\RoCoLe-classes.xlsx"
path_photos = r"C:\Users\Augusto\Documents\tese\files\imagens\Photos"

# Ler o arquivo xlsx utilizando a biblioteca pandas
classes_df = pd.read_excel(path_classes)


def rm_dir_classes():
    os.makedirs(os.path.join(path_photos, "healthy"), exist_ok=True)
    os.makedirs(os.path.join(path_photos, "rust_level_1"), exist_ok=True)
    os.makedirs(os.path.join(path_photos, "rust_level_2"), exist_ok=True)
    os.makedirs(os.path.join(path_photos, "rust_level_3"), exist_ok=True)
    os.makedirs(os.path.join(path_photos, "rust_level_4"), exist_ok=True)
    os.makedirs(os.path.join(path_photos, "unhealthy"), exist_ok=True)
    shutil.rmtree(os.path.join(path_photos, "unhealthy"))
    shutil.rmtree(os.path.join(path_photos, "healthy"))
    shutil.rmtree(os.path.join(path_photos, "rust_level_1"))
    shutil.rmtree(os.path.join(path_photos, "rust_level_2"))
    shutil.rmtree(os.path.join(path_photos, "rust_level_3"))
    shutil.rmtree(os.path.join(path_photos, "rust_level_4"))
   

rm_dir_classes()
if (num_classes == 6):
    # Criar as pastas "healthy" e "unhealthy" dentro do caminho das fotos
    os.makedirs(os.path.join(path_photos, "healthy"), exist_ok=True)
    os.makedirs(os.path.join(path_photos, "rust_level_1"), exist_ok=True)
    os.makedirs(os.path.join(path_photos, "rust_level_2"), exist_ok=True)
    os.makedirs(os.path.join(path_photos, "rust_level_3"), exist_ok=True)
    os.makedirs(os.path.join(path_photos, "rust_level_4"), exist_ok=True)
    os.makedirs(os.path.join(path_photos, "red_spider_mite"), exist_ok=True)
elif (num_classes == 2):
    # Criar as pastas "healthy" e "unhealthy" dentro do caminho das fotos
    os.makedirs(os.path.join(path_photos, "healthy"), exist_ok=True)
    os.makedirs(os.path.join(path_photos, "unhealthy"), exist_ok=True)

# Iterar sobre cada linha do DataFrame e copiar o arquivo correspondente para a pasta adequada
for index, row in classes_df.iterrows():
    filename = row["File"]
    if(num_classes==6):
        label = row["Multiclass.Label"]
        source_path = os.path.join(path_photos, filename)
        shutil.copyfile(source_path, os.path.join(path_photos, label, filename))
    else:
        label = row["Binary.Label"]
        source_path = os.path.join(path_photos, filename)
        shutil.copyfile(source_path, os.path.join(path_photos, label, filename))