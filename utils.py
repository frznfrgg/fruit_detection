import numpy as np
import torch


class RLETransform(torch.nn.Module):
    '''
    transforms CVAT data to proper objects of labels, bboxes and masks
    '''
    def forward(self, Target):
        labels = []
        bboxes = []
        masks = []

        for data in Target.annotations.shapes:
            # лейблы в датасете начинаются с 9. Нулевая метка занята под фон.
            # Поэтому все метки надо сдвинуть на 8 влево
            labels.append(data['label_id']-8)

            bboxes_raw = data['points'][-4:]
            # y1, x1, y2, x2
            bboxes.append([bboxes_raw[1], bboxes_raw[0], bboxes_raw[3], bboxes_raw[2]])

            masks.append(self.rle_to_mask(data['points'][:-4], bboxes[-1]))

        return labels, bboxes, masks


    def rle_to_mask(self, rle, bbox):
        height = int(bbox[2]-bbox[0])+1
        width  = int(bbox[3]-bbox[1])+1
        top = int(bbox[0])
        left = int(bbox[1])

        mask = np.zeros(width * height, dtype=np.uint8)

        rle.insert(0,0.0)
        rle_pairs = np.array(rle, dtype=np.uint32)

        pos = 0
        for i in range(0, len(rle_pairs), 2):
            start = pos
            end = pos + int(rle_pairs[i])
            mask[start:end] = 1  # Устанавливаем пиксели в белый цвет (255)
            pos = end + int(rle_pairs[i + 1])  # Пропускаем следующие пиксели


        return torch.unsqueeze(torch.from_numpy(mask.reshape((height, width))), dim=0)
