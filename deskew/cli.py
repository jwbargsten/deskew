import argparse
import sys

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
import logging

from deskew import determine_skew

log_dfmt = "%Y-%m-%d %H:%M:%S"
log_fmt = "[%(asctime)s] [%(levelname)s] %(name)s.%(funcName)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=log_dfmt)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", default=None, help="Output file name")
    parser.add_argument("--sigma", default=3.0, help="The use sigma")
    parser.add_argument("--num-peaks", default=20, help="The used num peaks")
    parser.add_argument("--background", help="The used background color")
    parser.add_argument(default=None, dest="input", help="Input file name")
    options = parser.parse_args()

    img = io.imread(options.input)
    grayscale = rgb2gray(img)
    try:
        angle = determine_skew(grayscale, sigma=options.sigma, num_peaks=options.num_peaks)
    except:
        e = sys.exc_info()[0]
        logger.error(f"could not estimate angle: {e}")
        angle = 0

    if options.output is None:
        print(angle)
    else:
        if options.background:
            try:
                background = [int(c) for c in options.background.split(",")]
            except:  # pylint: disable=bare-except
                logger.error("Wrong background color, should be r,g,b")
                sys.exit(1)
            rotated = rotate(img, angle, resize=True, cval=-1) * 255
            pos = np.where(rotated == -255)
            rotated[pos[0], pos[1], :] = background
        else:
            rotated = rotate(img, angle, resize=True) * 255
        io.imsave(options.output, rotated.astype(np.uint8))


if __name__ == "__main__":
    main()
