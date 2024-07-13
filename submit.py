import re
from data.dataset import load_all_from_path
from glob import glob
import numpy as np
import cv2
from utils import np_to_tensor
from postprocess import crf

PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road


def create_predictions(model, device):
    test_path = 'data/test/images'
    test_filenames = (glob(test_path + '/*.png'))
    orig_test_images = load_all_from_path(test_path)
    batch_size = orig_test_images.shape[0]
    size = orig_test_images.shape[1:3]
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in orig_test_images], 0)
    test_images = test_images[:, :, :, :3]
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape
    # now compute labels

    crf_refined = []
    for i, image in enumerate(orig_test_images):
        image = (image * 255).astype(np.uint8)
        image = image[:, :, :3]
        pred = test_pred[i]
        res = crf.dense_crf(image, pred)
        crf_refined.append(res)

    test_pred = np.stack(crf_refined)

    test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
    return test_pred, test_filenames


def create_submission(submission_filename, model, device):
    test_pred, test_filenames = create_predictions(model, device)
    with open('./submissions/' + submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), test_pred):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j*PATCH_SIZE, i*PATCH_SIZE, int(patch_array[i, j])))


def crop_image_and_mask(x):
    assert x.shape == (3, 400, 400)

    positions = [(0, 0), (0, 400-224), (400-224, 0), (400-224, 400-224)]

    crops = []

    for y_pos, x_pos in positions:
        x_cropped = x[:, y_pos:y_pos+224, x_pos:x_pos+224]
        assert x_cropped.shape == (3, 224, 224)
        crops.append(x_cropped)

    return crops


def create_predictions_224_patches(model, device):
    test_path = 'data/test/images'
    test_filenames = (glob(test_path + '/*.png'))
    test_images = load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in test_images], 0)
    test_images = test_images[:, :, :, :3]
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    test_pred = []
    for t in test_images.unsqueeze(1):
        # What we do here is to crop the image into 4 patches, predict each patch and then average the predictions
        # where the patches overlap
        crops = crop_image_and_mask(t)
        test_pred = [model(c).detach().cpu().numpy() for c in crops]
        final_mask = np.zeros((400, 400))
        count_mask = np.zeros((400, 400))
        for patch, (y_pos, x_pos) in zip(test_pred, [(0, 0), (0, 400-224), (400-224, 0), (400-224, 400-224)]):
            final_mask[y_pos:y_pos+224, x_pos:x_pos+224] += patch
            count_mask[y_pos:y_pos+224, x_pos:x_pos+224] += 1
        final_mask /= count_mask
        test_pred.append(final_mask)
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape
    # now compute labels
    test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
    return test_pred, test_filenames


def create_submission_224_patches(submission_filename, model, device):
    test_pred, test_filenames = create_predictions(model, device)
    with open('./submissions/' + submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), test_pred):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j*PATCH_SIZE, i*PATCH_SIZE, int(patch_array[i, j])))