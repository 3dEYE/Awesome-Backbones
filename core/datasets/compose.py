# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
import copy

from .loading import LoadImageFromFile
from .transforms import CenterCrop, Normalize, RandomCrop, RandomErasing, RandomFlip, RandomResizedCrop, Resize
from .formatting import Collect, ImageToTensor, ToNumpy, ToPIL, ToTensor, Transpose, to_tensor


class Compose(object):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform_cfg = copy.deepcopy(transform)
                transform = eval(transform_cfg.pop('type'))(**transform_cfg)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string
