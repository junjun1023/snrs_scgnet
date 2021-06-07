import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset

class JSRTset(BaseDataset):

        def __init__(
                self,
                root, 
                augmentation=None, 
                preprocessing=None,
                **kwargs):
                
                self.root = root
                ids = os.listdir(os.path.join(root, "images"))
                ids.sort()
                self.ids = ids

                self.augmentation = augmentation
                self.preprocessing = preprocessing
                
        
        def __getitem__(self, i):

                # read image
                img = cv2.imread(os.path.join(self.root, "images", self.ids[i]), 0)
                msk = cv2.imread(os.path.join(self.root, "masks", self.ids[i]), 0)

                tmp = np.stack([img, msk], axis=-1)

                # apply augmentations
                if self.augmentation:
                        sample = self.augmentation(image=tmp)
                        tmp = sample["image"]

                img, msk = tmp[..., 0], tmp[..., 1]
                cpy = img.copy()
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # apply preprocessing
                if self.preprocessing:
                        img = self.preprocessing(image=img)["image"]
                else:
                        img = img / 255
                        img = img.transpose(2, 0, 1).astype('float32')


                cpy = cpy / 255
                msk = msk / 255

                return img, msk, cpy
                

        def __len__(self):
                return len(self.ids)

        def itemname(self, i):
                return self.ids[i]

