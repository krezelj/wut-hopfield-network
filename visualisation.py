import numpy as np
import numpy.typing as npt
from PIL import Image


def draw_bitmap(X: npt.NDArray):
    bitmap = np.where(X < 0, 0, 255).astype(np.uint8)
    return Image.fromarray(bitmap, mode='L')
