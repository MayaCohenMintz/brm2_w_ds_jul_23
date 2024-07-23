from typing import Any, Tuple

import numpy as np
from albumentations import ImageOnlyTransform


class FixEmptyImage(ImageOnlyTransform):
    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if img.max() == img.min():
            img[0, 0] = img.min() + 1
        return img

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ()
