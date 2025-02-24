import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch
from torch.utils.data import Dataset
from bs4 import BeautifulSoup
import glob
import torch.nn as nn

class ImageData(Dataset):
    def __init__(self, image_folder, xml_path):
        """
        image_folder - путь к директории с фотографиями
        xml_path - путь к фалу с разметками
        """
        
        with open(xml_path, 'r') as f:
            data = f.read()
        Bs_data = BeautifulSoup(data, "xml")
        
        self.mask_raw = Bs_data.find_all('image')
        self.standart_size = (int(self.mask_raw[0].get('height')),int(self.mask_raw[0].get('width'))) 
        
        images_path = glob.glob(f'{image_folder}/*.jpg')
        self.image_folder = image_folder
        
        if len(self.mask_raw) > len(images_path):
            print("Warning! Some images are missing")
        elif len(self.mask_raw) < len(images_path):
            print("Warning! Some images are not labeled")
        
    def __len__(self):
        return len(self.mask_raw)
    
    def rle_to_mask(self, rle, width, height):
        mask = np.zeros(width * height, dtype=np.uint8)
        rle_pairs = rle
        pos = 0
        for i in range(0, len(rle_pairs), 2):
            start = pos
            end = pos + rle_pairs[i]
            mask[start:end] = 1  # Устанавливаем пиксели в белый цвет (255)
            pos = end + rle_pairs[i + 1]  # Пропускаем следующие пиксели
        return mask.reshape((height, width))
    
    def _extract_mask(self, index):
        label_object = self.mask_raw[index]
        image_name = label_object.get('name')
        mask_data = label_object.find('mask')
        
        if mask_data is None:
            raise ValueError(f'image {image_name} does not have label')
        else:
            width = int(mask_data.get('width'))
            height = int(mask_data.get('height'))
            
            left = int(mask_data.get('left'))
            top = int(mask_data.get('top'))
            
            RLE = "0, "+mask_data.get('rle')
            RLE = list(map(int, RLE.split(',')))
            
            mini_mask = self.rle_to_mask(RLE, width, height)
            
            full_mask = np.zeros(self.standart_size)
            full_mask[top:top+height, left:left+width] = mini_mask
            
        return full_mask, image_name
            
    
    def __getitem__(self, index):
        mask, name = self._extract_mask(index)
        
        image = imageio.imread(f'{self.image_folder}/{name}')
        image = image.swapaxes(0,2).swapaxes(1,2)
        
        return image/255.0, mask

    def get_name_label(self, index):
        label_object = self.mask_raw[index]
        image_name = label_object.get('name')
        mask_data = label_object.find('mask')
        label = mask_data.get('label')
        return image_name, label


    @staticmethod
    def draw_appled_mask(image, mask, mask_koef=0.3):
        if type(image) is torch.Tensor:
            image = image.numpy()
        if type(mask) is torch.Tensor:
            mask = mask.numpy()

        cmap = plt.get_cmap('jet')
        rgba_img = cmap(mask)
        rgb_img = np.delete(rgba_img, 3, 2)

        image = image.swapaxes(1,2).swapaxes(0,2)

        plt.imshow(mask_koef*rgb_img+(1.0-mask_koef)*image)


class SC_Model(nn.Module):
    def __init__(self, n_classes = 1):
        super(SC_Model, self).__init__()
        drop = 0.4
        
#       moving down
        self.H1 = SC_Model.set_conv_block(3,64)
        self.P1 = nn.MaxPool2d(2)
        self.D1 = nn.Dropout2d(drop)
        
        self.H2 = SC_Model.set_conv_block(64,128)
        self.P2 = nn.MaxPool2d(2)
        self.D2 = nn.Dropout2d(drop)
        
        self.H3 = SC_Model.set_conv_block(128,256)
        self.P3 = nn.MaxPool2d(2)
        self.D3 = nn.Dropout2d(drop)
        
        self.H4 = SC_Model.set_conv_block(256,512)
        self.P4 = nn.MaxPool2d(2)
        self.D4 = nn.Dropout2d(drop)
        
#       bottom stage
        self.H5 = SC_Model.set_conv_block(512, 1024)
        self.U1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.D5 = nn.Dropout2d(drop)

#       moving up
        self.H6 = SC_Model.set_conv_block(1024,512)
        self.U2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.D6 = nn.Dropout2d(drop)
        
        self.H7 = SC_Model.set_conv_block(512, 256)
        self.U3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.D7 = nn.Dropout2d(drop)
        
        self.H8 = SC_Model.set_conv_block(256, 128)
        self.U4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.D8 = nn.Dropout2d(drop)
        
        self.H9 = SC_Model.set_conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, n_classes, 3, padding = 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, batch):
        c1 = self.H1(batch)
        x = c1
        x = self.P1(x)
        x = self.D1(x)
        
        c2 = self.H2(x)
        x = c2
        x = self.P2(x)
        x = self.D2(x)
        
        c3 = self.H3(x)
        x = c3
        x = self.P3(x)
        x = self.D3(x)
        
        c4 = self.H4(x)
        x = c4
        x = self.P4(x)
        x = self.D4(x)
        
        x = self.H5(x)
        x = self.U1(x)
        x = torch.cat((x, c4), dim=(1))
        x = self.D5(x)
        
        x = self.H6(x)
        x = self.U2(x)
        x = torch.cat((x, c3), dim=(1))
        x = self.D6(x)
        
        x = self.H7(x)
        x = self.U3(x)
        x = torch.cat((x, c2), dim=(1))
        x = self.D7(x)
        
        x = self.H8(x)
        x = self.U4(x)
        x = torch.cat((x, c1), dim=(1))
        x = self.D8(x)
        
        x = self.H9(x)
        
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x
        
    @staticmethod
    def set_conv_block(input_size, output_size, kernel_size = 3, padding = 1):
        block = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size, padding=padding),
            nn.BatchNorm2d(output_size, momentum=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(output_size, output_size, kernel_size, padding=padding),
            nn.BatchNorm2d(output_size, momentum=0.1),
            nn.LeakyReLU()
        )
        
        return block
        