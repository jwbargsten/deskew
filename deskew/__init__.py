import numpy.typing as npt

import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

def determine_skew(img, sigma = 3.0, num_peaks = 20):
    edges = canny(img, sigma=sigma)
    # assume that pages are already in the right orientation (portrait/landscape)
    # therefore, only change +-30 deg (pi/6). The get the right peaks,
    # we need to rotate theta also by 90 deg (p/2), because lines of text create the
    # peaks on horizontal line
    theta = np.linspace(np.pi/2-np.pi/6, np.pi/2+np.pi/6, 180)
    hspace, angles, distances = hough_line(edges, theta=theta)

    _, peaks, _ = hough_line_peaks(hspace, angles, distances, num_peaks=num_peaks)

    unique_peaks, count = np.unique(peaks, return_counts=True)
    pidx = np.argmax(count)
    the_peak_deg = np.rad2deg(unique_peaks[pidx] - np.pi/2)

    return the_peak_deg
