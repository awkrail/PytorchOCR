import math
import torch
import random
import cv2
import numpy as np


class ToTensor:
    def __init__(
        self,
    ):
        pass

    def __call__(self, image):
        return torch.from_numpy(image)


class RecResizeImg:
    def __init__(
        self,
        image_shape,
        padding = True,
        interpolation = cv2.INTER_LINEAR,
        **kwargs,
    ):
        self.image_shape = image_shape
        self.padding = padding
        self.interpolation = interpolation

    def __call__(self, image):
        imgC, imgH, imgW = self.image_shape
        h, w, _ = image.shape

        if not self.padding:
            resized_image = cv2.resize(image, (imgW, imgH), interpolation = self.interpolation)
            resized_w = imgW
        else:
            ratio = float(w / h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(image, (resized_w, imgH))

        resized_image = resized_image.astype(np.float32)
        if self.image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255

        resized_image -= 0.5
        resized_image /= 0.5
        padding_image = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_image[:, :, 0:resized_w] = resized_image
        return padding_image


class RecAug:
    def __init__(
        self,
        tia_prob = 0.4,
        crop_prob = 0.4,
        reverse_prob = 0.4,
        noise_prob = 0.4,
        jitter_prob = 0.4,
        blur_prob = 0.4,
        hsv_aug_prob = 0.4,
        **kwargs,
    ):
        self.tia_prob = tia_prob
        self.bda = BaseDataAugmentation(
            crop_prob,
            reverse_prob,
            noise_prob,
            jitter_prob,
            blur_prob,
            hsv_aug_prob 
        )


    def __call__(self, image):
        h, w, _ = image.shape
        if random.random() <= self.tia_prob:
            if h >= 20 and w >= 20:
                image = tia_distort(image, random.randint(3, 6))
                image = tia_stretch(image, random.randint(3, 6))
                image = tia_perspective(image)
        return self.bda(image)


class BaseDataAugmentation:
    def __init__(
        self,
        crop_prob = 0.4,
        reverse_prob = 0.4,
        noise_prob = 0.4,
        jitter_prob = 0.4,
        blur_prob = 0.4,
        hsv_aug_prob = 0.4,
        ksize = 5,
        sigma = 1,
        **kwargs,
    ):
        self.crop_prob = crop_prob
        self.reverse_prob = reverse_prob
        self.noise_prob = noise_prob
        self.jitter_prob = jitter_prob
        self.blur_prob = blur_prob
        self.hsv_aug_prob = hsv_aug_prob

        # GaussianBlur
        self.fil = cv2.getGaussianKernel(ksize=ksize, sigma=sigma, ktype=cv2.CV_32F)


    def __call__(self, image):
        h, w, _ = image.shape

        if random.random() <= self.crop_prob and h >= 20 and w >= 20:
            image = get_crop(image)

        if random.random() <= self.blur_prob:
            image = cv2.sepFilter2D(image, -1, self.fil, self.fil)

        if random.random() <= self.hsv_aug_prob:
            image = hsv_aug(image)

        if random.random() <= self.jitter_prob:
            image = jitter(image)

        if random.random() <= self.noise_prob:
            image = add_gaussian_noise(image)

        if random.random() <= self.reverse_prob:
            image = 255 - image

        return image


"""
Data Augumentation
"""
def get_crop(image, top_min = 1, top_max = 8):
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_image = image.copy()
    if ratio:
        crop_image = crop_image[top_crop:h, :, :]
    else:
        crop_image = crop_image[0:h-top_crop, :, :]
    return crop_image


def flag():
    return 1 if random.random() > 0.5000001 else -1


def hsv_aug(image, multiply = 0.001):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    delta = multiply * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_image


def jitter(image, min_h = 10, min_w = 10, multiply = 0.01):
    h, w, _ = image.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_image = image.copy()
        image[s:, s:, :] = src_image[:h-s, :w-s, :]
    return image


def add_gaussian_noise(image, mean = 0, var = 0.1):
    noise = np.random.normal(mean, var**0.5, image.shape)
    image = image + 0.5 * noise
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    return image
