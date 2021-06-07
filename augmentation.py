import albumentations as albu
import cv2
import numpy as np

def image_resize(image, size = 512, inter = cv2.INTER_AREA, **kwargs):
        # initialize the dimensions of the image to be resized and
        # grab the image size

        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if size is None:
                return image

        # check to see if the width is None
        if h > w:
                r = size / float(h)
                dim = (int(w * r), size)
        else:
                r = size / float(w)
                dim = (size, int(h * r))

        # if width is None:
        #         # calculate the ratio of the height and construct the
        #         # dimensions
        #         r = height / float(h)
        #         dim = (int(w * r), height)

        # # otherwise, the height is None
        # else:
        #         # calculate the ratio of the width and construct the
        #         # dimensions
        #         r = width / float(w)
        #         dim = (width, int(h * r))

        resized = []
        for i in range(image.shape[-1]):
                tmp = image[..., i]
                tmp = cv2.resize(tmp, dim, interpolation = inter)
                resized.append(tmp)

        resized = np.stack(resized, axis=-1)
        # resize the image
        # resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized


def random_erasing(x, p=0.5, sl=0.02, sh=0.4, r1=0.3, **kwargs):
        import random
        import math

        W = x.shape[0]
        H = x.shape[1]
        S = W * H
        p1 = random.uniform(0, 1)
        r2 = 1 / r1

        if p1 >= p:
                return x
        else:
                while True:
                        se = random.uniform(sl, sh) * S
                        re = random.uniform(r1, r2)

                        he = math.sqrt(se * re)
                        we = math.sqrt(se / re)

                        xe = random.uniform(0, W)
                        ye = random.uniform(0, H)

                        if (xe+we <= W) and (ye+he <= H):

                                for _x in range(int(xe), int(xe+we)):
                                        for _y in range(int(ye), int(ye+he)):
                                                for c in range(0, 3):
                                                        x[_x, _y, c] = random.uniform(0, 255)

                                return x



def img_augmentation():
        img_transform = [
                albu.InvertImg(),
                # albu.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
                # albu.OneOf(
                #         [
                #                 albu.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                #                 albu.RandomContrast (limit=0.2, always_apply=False, p=0.5),
                #                 albu.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
                #                 albu.RandomBrightness (limit=0.2, always_apply=False, p=0.5),
                #                 albu.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
                #         ],
                #         p=0.5
                # )
                
        ]
        return albu.Compose(img_transform)


def get_training_augmentation():
        train_transform = [
                # ref: https://github.com/albumentations-team/albumentations/issues/640
                albu.Lambda(name="image_resize", image=image_resize, always_apply=True),   
                # albu.RandomCrop(height=512, width=512, p=1),
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
                albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        ]
        return albu.Compose(train_transform)



def get_validation_augmentation():
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
                # ref: https://github.com/albumentations-team/albumentations/issues/640
                albu.Lambda(name="image_resize", image=image_resize, always_apply=True),   
                albu.PadIfNeeded(512, 512, always_apply=True, border_mode=0)
        ]
        return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
        """Construct preprocessing transform
        
        Args:
                preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
                transform: albumentations.Compose
        
        """
        
        _transform = [
                albu.Lambda(image=preprocessing_fn),
                albu.Lambda(image=to_tensor),
        ]
        return albu.Compose(_transform)