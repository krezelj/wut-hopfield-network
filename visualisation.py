import numpy as np
import numpy.typing as npt
from PIL import Image


def draw_bitmap(X: npt.NDArray, scale: int = 10):
    shape = X.shape
    bitmap = np.where(X < 0, 0, 255).astype(np.uint8)
    img = Image.fromarray(bitmap, mode='L')
    return img.resize((shape[1] * scale, shape[0] * scale), resample=Image.NEAREST)

def draw_dataset(X: npt.NDArray, scale: int = 10):
    X = np.hstack(X)
    return draw_bitmap(X, scale)

def draw_comparison(X: npt.NDArray, Y: npt.NDArray, scale: int = 10):
    img1 = draw_dataset(X, scale)
    img2 = draw_dataset(Y, scale)

    new_height = img1.height + img2.height
    combined = Image.new("RGB", (img1.width, new_height))

    combined.paste(img1, (0, 0))
    combined.paste(img2, (0, img1.height))

    return combined
