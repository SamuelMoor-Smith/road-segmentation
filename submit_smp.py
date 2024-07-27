import re
import PIL
import torch
import os
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
from PIL import Image
from torch.utils.data import DataLoader
from preprocess.augment import smp_get_preprocessing
from data.dataset import TestDataset


def patch_to_label(patch):
    patch = patch.astype(np.float64) / 255
    df = np.mean(patch)
    if df > 0.25:
        return 1
    else:
        return 0


def compute_true_count(i, k, l, patched_preds):
    top = patched_preds[i][k-1][l]
    bot = patched_preds[i][k+1][l]
    left = patched_preds[i][k][l-1]
    right = patched_preds[i][k][l+1]
    tl = patched_preds[i][k-1][l-1]
    tr = patched_preds[i][k-1][l+1]
    bl = patched_preds[i][k+1][l-1]
    br = patched_preds[i][k+1][l+1]
    variables = [top, bot, left, right, tl, tr, br, bl]
    true_count = sum(variables)
    return true_count

def post_process_patches(patched_preds):
    postprocessed_patches = np.empty_like(patched_preds)
    for i in range(len(patched_preds)):
        for k in range(25):
            for l in range(25):
                if k > 1 and k < 23 and l > 1 and l < 23:
                    true_count = compute_true_count(i, k, l, patched_preds)
                    if patched_preds[i][k][l] == 0:
                        if true_count == 8:
                            postprocessed_patches[i][k][l] = 0.7
                        else:
                            postprocessed_patches[i][k][l] = patched_preds[i][k][l]
                    else:
                        if true_count == 0:
                            postprocessed_patches[i][k][l] = 0.3
                        elif true_count == 1:
                            nei_counts = []
                            for nei in [(k - 1, l), (k - 1, l - 1), (k - 1, l + 1), (k, l + 1), (k, l - 1), (k + 1, l),
                                        (k + 1, l - 1), (k + 1, l + 1)]:
                                nei_counts.append(compute_true_count(i, nei[0], nei[1], patched_preds))
                                if all(count <= 1 for count in nei_counts):
                                    postprocessed_patches[i][k][l] = 0.7
                                else:
                                    postprocessed_patches[i][k][l] = patched_preds[i][k][l]
                        else:
                            postprocessed_patches[i][k][l] = patched_preds[i][k][l]
                else:
                    postprocessed_patches[i][k][l] = patched_preds[i][k][l]

def mask_to_submission_strings(image_filename, im_arr, mask_dir=None, full_mask_dir=None):
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im_arr = (im_arr.reshape(400, 400) * 255.0).astype(np.uint8)
    patch_size = 16
    mask = np.zeros_like(im_arr)
    for j in range(0, im_arr.shape[1], patch_size):
        for i in range(0, im_arr.shape[0], patch_size):
            patch = im_arr[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            mask[i:i + patch_size, j:j + patch_size] = int(label * 255)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

    if mask_dir:
        save_mask_as_img(mask, os.path.join(mask_dir, "mask_" + image_filename))
    if full_mask_dir:
        save_mask_as_img(im_arr, os.path.join(full_mask_dir, "mask_" + image_filename))


def save_mask_as_img(img_arr, mask_filename):
    img = PIL.Image.fromarray(img_arr)
    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    img.save(mask_filename)


def masks_to_submission(submission_filename, image_filenames, predictions, full_mask_dir=None, mask_dir=None):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, pred in zip(image_filenames, predictions):
            f.writelines('{}\n'.format(s) for s in
                         mask_to_submission_strings(image_filename=fn, im_arr=pred, mask_dir=mask_dir,
                                                    full_mask_dir=full_mask_dir))


def make_submission(model, config, test_dir, submission_dir):
    backbone = config['backbone']
    device = config['device']
    resize = config['resize']
    batch_size = config['batch_size']
    preprocessing_fn = smp.encoders.get_preprocessing_fn(backbone, 'imagenet')
    preprocessing_fn = smp_get_preprocessing(preprocessing_fn)

    model.to(device=device)
    model.eval()

    test_dataset = TestDataset(
        data_dir=test_dir,
        transforms='validation',
        preprocess=preprocessing_fn,
        resize=resize)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True,
                             worker_init_fn=42,
                             )

    preds_lst = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            preds = model.predict(images)

            predicted_masks = torch.sigmoid(preds).permute(0, 2, 3, 1).cpu().numpy()  # [batch_size,height,width,channels]
            images = images.permute(0, 2, 3, 1).cpu().numpy()

            crop = A.Compose([
                A.CenterCrop(height=400, width=400),
            ])

            for image, mask in zip(images, predicted_masks):
                cropped = crop(image=image, mask=mask)
                mask = cropped["mask"]
                preds_lst.append(mask)

    masks_to_submission(submission_filename=submission_dir,
                        full_mask_dir="full_masks/",
                        mask_dir="patched_masks/",
                        image_filenames=test_dataset.filenames,
                        predictions=preds_lst)


if __name__ == "__main__":
    # For debugging purposes
    device = 'cpu'
    # Load the state dictionary into the model
    smp_config = {
        'decoder_channels': [256, 128, 64, 32, 16],
        'backbone': 'efficientnet-b7',
        'epochs': 150,
        'use_epfl': False,
        'use_deepglobe': False,
        'augmentation_factor': 1,
        'transformation': 'minimal',
        'resize': 416,
        'validation_size': 0.15,
        'seed': 42,
        'batch_size': 4,
        'lr': 0.0005,
        'device': device
    }

    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b7',
        encoder_weights='imagenet',
        decoder_channels=[256, 128, 64, 32, 16],
        decoder_attention_type=None,
        classes=1,
        activation=None,
    )

    state_dict = torch.load('model_checkpoints/UNetpp_B7_0.927.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    submission_dir = 'submissions/UNetpp_0.927_Basic_PostP.csv'
    make_submission(model, smp_config, 'data', submission_dir)
