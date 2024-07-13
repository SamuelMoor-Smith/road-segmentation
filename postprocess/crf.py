import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils


MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3


def dense_crf(img, mask):
    """
    Applies DenseCRF to the provided image and single-channel mask.

    Parameters:
    img (numpy.ndarray): Input image with shape (H, W, 3).
    mask (numpy.ndarray): Single-channel mask with shape (H, W).

    Returns:
    numpy.ndarray: Refined single-channel mask with shape (H, W).
    """
    h, w = mask.shape

    # Create unary potentials (log-probabilities) from the binary mask
    output_probs = np.stack([1 - mask, mask], axis=0)  # Shape: (2, H, W)
    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, 2)  # We have 2 classes: road and non-road
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((2, h, w))

    # Take the argmax to get the final labels (0 or 1)
    refined_mask = np.argmax(Q, axis=0)

    return refined_mask

