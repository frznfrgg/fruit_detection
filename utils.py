from MLtools import ImageData
import torch
import xml.etree.ElementTree as ET
import numpy as np
from typing import List

import random
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.notebook import tqdm

def add_annotation_to_xml(file_path, id, name, label, RLE, left=0, top=0, 
                          width=640, height=640, mask_width=640, mask_height=640):
    # Парсим XML-файл
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Создаем новый элемент <image>
    image = ET.Element('image', {
        'id': str(id),
        'name': name,
        'width': str(width),
        'height': str(height)
    })
    
    str_RLE = lsit_to_string(RLE)
    # Создаем новый элемент <mask> и добавляем его в <image>
    mask = ET.SubElement(image, 'mask', {
        'label': label,
        'source': "manual",
        'occluded': "0",
        'rle': str_RLE,
        'left': str(left),
        'top': str(top),
        'width': str(mask_width),
        'height': str(mask_height),
        'z_order': "0"
    })

    # Добавляем новый элемент <image> в корневой элемент <annotations>
    root.append(image)

    # Сохраняем изменения в XML-файл
    tree.write(file_path, encoding='utf-8', xml_declaration=True)


def rle_encode_binary(array) -> List[int]:
    # Выравниваем массив в одномерный
    flattened = array.flatten()
    
    # Инициализируем переменные для RLE
    rle = []
    prev_value = flattened[0]
    count = 1
    
    # Проходим по массиву и строим RLE
    for value in flattened[1:]:
        if value == prev_value:
            count += 1
        else:
            rle.append(count)
            prev_value = value
            count = 1
    
    # Добавляем последний элемент
    rle.append(count)
    
    return rle


def lsit_to_string(rle_result):
    s = ""
    for el in rle_result:
        s += str(el) + ', '
        
    return s[:-2]


def aug_dataset(Data: ImageData, n_copys: int=10, xml_path = "annotations.xml",
               images_path="resized_fruits"):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.PILToTensor(),
    ])
#   добавляем новые записи с конца xml списка
    cur_id = len(Data)
    for index, (image, mask) in tqdm(enumerate(Data)):
        orig_name, label = Data.get_name_label(index)
#       кол-во копий делается на одну меньше чтобы вместе с оригиналом вышло по n_copys штук
        for i in range(cur_id,cur_id+n_copys-1):
            seed = random.randint(-1000, 1000)
            name = str(i) + "_" + orig_name

            torch.manual_seed(seed)
            tr_image = transform_train(torch.tensor(image))
#           сохраняем новое изображение в общую папку
            save_image(tr_image.float()/255.0, f'{images_path}/{name}')

            torch.manual_seed(seed)
            tr_mask = transform_train(torch.tensor(mask))

    #       переводим новую маску в строку типа RLE
            RLE = rle_encode_binary(tr_mask/255.0)

    #       добавляем новую маску в xml
            add_annotation_to_xml(xml_path, i, name, label, RLE)
        
        cur_id = cur_id+n_copys-1