import random
from functools import wraps
from typing import Callable, Union

import cv2
import numpy as np
from albumentations.core.transforms_interface import BasicTransform
from skimage.exposure import match_histograms
from typing_extensions import Concatenate, ParamSpec

GRAYSCALE_IMAGE_SHAPE = 2
NON_GRAY_IMAGE_SHAPE = 3
RGB_NUM_CHANNELS = 3

P = ParamSpec("P")

NPDTYPE_TO_OPENCV_DTYPE = {
    np.uint8: cv2.CV_8U,
    np.uint16: cv2.CV_16U,
    np.int32: cv2.CV_32S,
    np.float32: cv2.CV_32F,
    np.float64: cv2.CV_64F,
    np.dtype("uint8"): cv2.CV_8U,
    np.dtype("uint16"): cv2.CV_16U,
    np.dtype("int32"): cv2.CV_32S,
    np.dtype("float32"): cv2.CV_32F,
    np.dtype("float64"): cv2.CV_64F,
}


def get_opencv_dtype_from_numpy(value: Union[np.ndarray, int, np.dtype, object]) -> int:
    """Return a corresponding OpenCV dtype for a numpy's dtype
    :param value: Input dtype of numpy array
    :return: Corresponding dtype for OpenCV
    """
    if isinstance(value, np.ndarray):
        value = value.dtype
    return NPDTYPE_TO_OPENCV_DTYPE[value]


def preserve_shape(
        func: Callable[Concatenate[np.ndarray, P], np.ndarray],
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Preserve shape of the image"""

    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        shape = img.shape
        result = func(img, *args, **kwargs)
        return result.reshape(shape)

    return wrapped_function


def normalize_cv2(img: np.ndarray, mean: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    if mean.shape and len(mean) != 4 and mean.shape != img.shape:
        mean = np.array(mean.tolist() + [0] * (4 - len(mean)), dtype=np.float64)
    if not denominator.shape:
        denominator = np.array([denominator.tolist()] * 4, dtype=np.float64)
    elif len(denominator) != 4 and denominator.shape != img.shape:
        denominator = np.array(denominator.tolist() + [1] * (4 - len(denominator)), dtype=np.float64)

    img = np.ascontiguousarray(img.astype("float32"))
    cv2.subtract(img, mean.astype(np.float64), img)
    cv2.multiply(img, denominator.astype(np.float64), img)
    return img


def normalize_numpy(img: np.ndarray, mean: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def normalize(img: np.ndarray, mean: np.ndarray, std: np.ndarray, max_pixel_value: float = 255.0) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    if img.ndim == 3 and img.shape[-1] == 3:
        return normalize_cv2(img, mean, denominator)
    return normalize_numpy(img, mean, denominator)


def copy_paste_data_augmentation(image1, image2, label, min_area_ratio=0.01, max_area_ratio=0.99):
    """
    在变化检测任务中进行复制-粘贴的数据增强（适用于多分类）。

    参数:
    image1 (numpy.ndarray): 输入前时相图像,shape 为 (height, width, channels)。
    image2 (numpy.ndarray): 输入后时相图像,shape 为 (height, width, channels)。
    label (numpy.ndarray): 输入标签,shape 为 (height, width),值为 0 表示背景,1~6 表示前景类别。
    min_area_ratio (float, 可选): 被复制区域占原图面积的最小比例,默认为 0.01。
    max_area_ratio (float, 可选): 被复制区域占原图面积的最大比例,默认为 0.2。

    返回:
    增强后的图像和标签。
    """
    height, width = label.shape

    # 找到所有前景区域的轮廓
    contours, _ = cv2.findContours((label > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 随机选择一个前景区域
    selected_contour = np.random.randint(0, len(contours))
    x, y, w, h = cv2.boundingRect(contours[selected_contour])

    # 计算选定区域的面积占原图的比例
    area_ratio = (w * h) / (height * width)

    # 如果面积比例不在指定范围内,则返回原图
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return image1, image2, label
    elif (w == width) or (h == height):
        return image1, image2, label

    # 复制选定区域的图像和标签
    roi_image1 = image1[y:y + h, x:x + w].copy()
    roi_image2 = image2[y:y + h, x:x + w].copy()
    roi_label = label[y:y + h, x:x + w].copy()

    # 随机选择一个粘贴位置
    paste_x = np.random.randint(0, width - w)
    paste_y = np.random.randint(0, height - h)

    new_label = label.copy()
    new_label[paste_y:paste_y + h, paste_x:paste_x + w][roi_label > 0] = roi_label[roi_label > 0]

    # 粘贴图像
    new_image1 = image1.copy()
    new_image1[paste_y:paste_y + h, paste_x:paste_x + w, :] = roi_image1

    new_image2 = image2.copy()
    new_image2[paste_y:paste_y + h, paste_x:paste_x + w, :] = roi_image2

    return new_image1, new_image2, new_label


@preserve_shape
def apply_histogram(img: np.ndarray, reference_image: np.ndarray, blend_ratio: float) -> np.ndarray:
    # Resize reference image only if necessary
    if img.shape[:2] != reference_image.shape[:2]:
        reference_image = cv2.resize(reference_image, dsize=(img.shape[1], img.shape[0]))

    img, reference_image = np.squeeze(img), np.squeeze(reference_image)

    # Determine if the images are multi-channel based on a predefined condition or shape analysis
    is_multichannel = img.ndim == NON_GRAY_IMAGE_SHAPE and img.shape[2] == RGB_NUM_CHANNELS

    # Match histograms between the images
    matched = match_histograms(img, reference_image, channel_axis=2 if is_multichannel else None)

    # Blend the original image and the matched image
    return cv2.addWeighted(matched, blend_ratio, img, 1 - blend_ratio, 0, dtype=get_opencv_dtype_from_numpy(img.dtype))


class ExchangeTime(BasicTransform):
    """Exchange images of different times.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
            self,
            always_apply=False,
            p=0.5,
    ):
        super(ExchangeTime, self).__init__(always_apply, p)

    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            kwargs['image'], kwargs['image_2'] = kwargs['image_2'], kwargs['image']

        return kwargs


class StyleTransfer(BasicTransform):
    """Style Transfer images of different times.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
        queue (str): The order in which style transfer is executed, including "A2B", "B2A", "Both".
        max_value (int): The maximum value of the image.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
            self,
            always_apply=False,
            queue='Both',
            max_value=255,
            p=0.5,
    ):
        super(StyleTransfer, self).__init__(always_apply, p)
        assert queue in ["A2B", "B2A", "Both"], "queue must be A2B or B2A or Both!"
        self.queue = queue
        self.max_value = max_value

    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if self.queue == "A2B":
            if (random.random() < self.p) or self.always_apply or force_apply:
                kwargs['image'] = self.__style_transfer__(kwargs['image'], kwargs['image_2'], self.max_value)
        elif self.queue == "B2A":
            if (random.random() < self.p) or self.always_apply or force_apply:
                kwargs['image_2'] = self.__style_transfer__(kwargs['image_2'], kwargs['image'], self.max_value)
        else:
            prob = self.p / 2
            if random.random() < prob:
                kwargs['image'] = self.__style_transfer__(kwargs['image'], kwargs['image_2'], self.max_value)
            elif prob < random.random() < self.p:
                kwargs['image_2'] = self.__style_transfer__(kwargs['image_2'], kwargs['image'], self.max_value)
            else:
                pass
        return kwargs

    @staticmethod
    def __style_transfer__(source_image, target_image, max_value):
        h, w, c = source_image.shape
        data_type = source_image.dtype
        out = []
        for i in range(c):
            source_image_f = np.fft.fft2(source_image[:, :, i])
            source_image_fshift = np.fft.fftshift(source_image_f)
            target_image_f = np.fft.fft2(target_image[:, :, i])
            target_image_fshift = np.fft.fftshift(target_image_f)

            change_length = 1
            source_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
            int(h / 2) - change_length:int(h / 2) + change_length] = \
                target_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
                int(h / 2) - change_length:int(h / 2) + change_length]

            source_image_ifshift = np.fft.ifftshift(source_image_fshift)
            source_image_if = np.fft.ifft2(source_image_ifshift)
            source_image_if = np.abs(source_image_if)

            source_image_if[source_image_if > max_value] = np.max(source_image[:, :, i])
            out.append(source_image_if)
        out = np.array(out).swapaxes(1, 0).swapaxes(1, 2)
        out = out.astype(data_type)
        return out


class HistogramMatching4CD(BasicTransform):
    def __init__(
            self,
            always_apply=False,
            queue='Both',
            blend_ratio=(0.5, 1.0),
            p=0.5,
    ):
        super(HistogramMatching4CD, self).__init__(always_apply, p)
        assert queue in ["A2B", "B2A", "Both"], "queue must be A2B or B2A or Both!"
        self.queue = queue
        self.blend_ratio = blend_ratio

    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if self.queue == "A2B":
            if (random.random() < self.p) or self.always_apply or force_apply:
                blend_ratio = random.uniform(self.blend_ratio[0], self.blend_ratio[1])
                kwargs['image'] = apply_histogram(kwargs['image'], kwargs['image_2'], blend_ratio)
        elif self.queue == "B2A":
            if (random.random() < self.p) or self.always_apply or force_apply:
                blend_ratio = random.uniform(self.blend_ratio[0], self.blend_ratio[1])
                kwargs['image_2'] = apply_histogram(kwargs['image_2'], kwargs['image'], blend_ratio)
        else:
            prob = self.p / 2
            blend_ratio = random.uniform(self.blend_ratio[0], self.blend_ratio[1])
            if random.random() < prob:
                kwargs['image'] = apply_histogram(kwargs['image'], kwargs['image_2'], blend_ratio)
            elif prob < random.random() < self.p:
                kwargs['image_2'] = apply_histogram(kwargs['image_2'], kwargs['image'], blend_ratio)
            else:
                pass
        return kwargs


class Normalize4CD(BasicTransform):
    def __init__(
            self,
            always_apply=False,
            meanA=(0.485, 0.456, 0.406),
            stdA=(0.229, 0.224, 0.225),
            meanB=(0.485, 0.456, 0.406),
            stdB=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
    ):
        super(Normalize4CD, self).__init__(always_apply, p)
        self.meanA = meanA
        self.stdA = stdA
        self.meanB = meanB
        self.stdB = stdB
        self.max_pixel_value = max_pixel_value

    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        kwargs['image'] = normalize(kwargs['image'], self.meanA, self.stdA, self.max_pixel_value)
        kwargs['image_2'] = normalize(kwargs['image_2'], self.meanB, self.stdB, self.max_pixel_value)
        return kwargs


class CopyPaste(BasicTransform):
    def __init__(
            self,
            always_apply=False,
            min_area_ratio=0.01,
            max_area_ratio=0.99,
            p=0.5,
    ):
        super(CopyPaste, self).__init__(always_apply, p)
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs
        if (random.random() < self.p) or self.always_apply or force_apply:
            kwargs['image'], kwargs['image_2'], kwargs['mask'] = copy_paste_data_augmentation(kwargs['image'],
                                                                                              kwargs['image_2'],
                                                                                              kwargs['mask'],
                                                                                              self.min_area_ratio,
                                                                                              self.max_area_ratio)
        return kwargs


class ISPRSLabelResize(BasicTransform):
    def __init__(
            self,
            always_apply=False,
            p=1.0,
    ):
        super(ISPRSLabelResize, self).__init__(always_apply, p)

    def __call__(self, **kwargs):
        h, w, c = kwargs['image'].shape
        kwargs['mask'] = cv2.resize(kwargs['mask'], (h, w), interpolation=cv2.INTER_NEAREST)
        # if kwargs['mask_2'] is not None:
        #     kwargs['mask_2'] = cv2.resize(kwargs['mask_2'], (h, w), interpolation=cv2.INTER_NEAREST)
        return kwargs


class ISPRSNormalize(BasicTransform):
    def __init__(
            self,
            always_apply=False,
            p=1.0,
    ):
        super(ISPRSNormalize, self).__init__(always_apply, p)

    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        new_img = []
        for i in range(kwargs['image'].shape[-1]):
            data = kwargs['image'][:, :, i]
            threshold1 = np.percentile(data, 96)
            threshold2 = np.percentile(data, 4)
            data = np.clip(data, threshold2, threshold1)
            # data = (data - data.min()) / (data.max() - data.min())
            data = data / data.max()
            new_img.append(data)
        kwargs['image'] = np.array(new_img).transpose((1, 2, 0)).astype(np.float32)

        new_img = []
        for i in range(kwargs['image'].shape[-1]):
            data = kwargs['image'][:, :, i]
            threshold1 = np.percentile(data, 96)
            threshold2 = np.percentile(data, 4)
            data = np.clip(data, threshold2, threshold1)
            # data = (data - data.min()) / (data.max() - data.min())
            data = data / data.max()
            new_img.append(data)
        kwargs['image_2'] = np.array(new_img).transpose((1, 2, 0)).astype(np.float32)
        return kwargs
