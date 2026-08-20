import os
import torch
from torch.utils.data import Dataset as BaseDataset 
from PIL import Image
import numpy as np

class CloudDataset(BaseDataset):
    def __init__(self, folder_path, txt_path, transform=None):
        self.folder_path = folder_path
        self.txt_path = txt_path
        self.transform = transform

        # Nombres de las imágenes
        self.image_names = []

        # Leemos el .txt
        self.labels = {}
        with open(self.txt_path, 'r') as f:
            for line in f:
                partes = line.strip().split(';')
                if len(partes) >= 2:
                    try:
                        nombre = partes[0].strip()
                        # Convertir la segunda columna a número
                        altura = float(partes[1].strip()) 
                        
                        # Si es un número, verificamos que la imagen exista y la añadimos
                        if os.path.exists(os.path.join(self.folder_path, nombre)):
                            self.image_names.append(nombre)
                            self.labels[nombre] = altura
                    except ValueError:
                        # Si no es un número IGNORA el dato y pasa al siguiente
                        continue 

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        img_name = self.image_names[index]
        image_path = os.path.join(self.folder_path, img_name) 
    
        # Abrimos la imagen en formato PIL
        image = Image.open(image_path).convert('RGB')

        # 3. Aplicamos las transformaciones si las hay (AllSky, Resize, etc.)
        if self.transform:
            image = self.transform(image)
        else:
            # Si no hay transformaciones, lo pasamos a tensor manualmente
            image = np.array(image) / 255.0
            image = image.astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

        altura = self.labels.get(img_name)
        height = torch.tensor(altura/10000.0, dtype=torch.float32) # División entre 10000 para un entrenamiento más óptimo


        return image, height



