from torch.utils.data import Dataset
from PIL import Image
import os

# Bacterial Pneumonia = 2, Viral Pneumonia = 1, Normal = 0
class XRAYDataset(Dataset):
    
    
    def __init__(self, root_dir, transform=None):
        
        self.transform = transform
        self.files = list()
        
        for dirs in os.listdir(root_dir):
            path = root_dir + dirs + '/'
            for file in os.listdir(path):
                o = dict()
                filename = path + file
                category = 0
                if dirs == 'PNEUMONIA':
                    if file.split('_')[1] == 'virus':
                        category = 1
                    else:
                        category = 2
                o['path'] = filename
                o['category'] = category
                self.files.append(o)
    
    
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, index):
        path = self.files[index]['path']
        category = self.files[index]['category']
        image = Image.open(path).convert('L')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'category': category}
        
    