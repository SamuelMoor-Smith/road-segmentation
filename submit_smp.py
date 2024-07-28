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
from tqdm import tqdm
import argparse


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
                if k > 0 and k < 24 and l > 0 and l < 24:
                    true_count = compute_true_count(i, k, l, patched_preds)
                    if patched_preds[i][k][l] == 0:
                        if true_count == 8:
                            postprocessed_patches[i][k][l] = 0.7
                        else:
                            postprocessed_patches[i][k][l] = patched_preds[i][k][l]
                    else:
                        if true_count == 0:
                            postprocessed_patches[i][k][l] = 0.3
                        else:
                            postprocessed_patches[i][k][l] = patched_preds[i][k][l]
                else:
                    postprocessed_patches[i][k][l] = patched_preds[i][k][l]
    return postprocessed_patches


def mask_to_submission_strings(image_filename, im_arr, mask_dir=None, full_mask_dir=None):
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im_arr = (im_arr.reshape(400, 400) * 255.0).astype(np.uint8)
    patch_preds = np.zeros((25, 25))
    for j in range(0, im_arr.shape[1], 16):
        for i in range(0, im_arr.shape[0], 16):
            patch = im_arr[i:i + 16, j:j + 16]
            label = patch_to_label(patch)
            patch_preds[i // 16, j // 16] = label
            #yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))
    """if mask_dir:
        save_mask_as_img(mask, os.path.join(mask_dir, "mask_" + image_filename))
    if full_mask_dir:
        save_mask_as_img(im_arr, os.path.join(full_mask_dir, "mask_" + image_filename))"""

    for j in range(patch_preds.shape[0]):
        for i in range(patch_preds.shape[1]):
            yield "{:03d}_{}_{},{}".format(img_number, j*16, i*16, int(patch_preds[i, j]))



def save_mask_as_img(img_arr, mask_filename):
    img = PIL.Image.fromarray(img_arr)
    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    img.save(mask_filename)


def masks_to_submission(submission_filename, image_filenames, predictions, full_mask_dir=None, mask_dir=None):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, pred in zip(image_filenames, predictions):
            f.writelines('{}\n'.format(s) for s in
                         mask_to_submission_strings(image_filename=fn, im_arr=pred, mask_dir=mask_dir, full_mask_dir=full_mask_dir))


def make_submission(models, backbones, device, test_dir, submission_dir):
    if len(models) > 1:
        preds, fn = get_ensemble_preds(models, backbones, device, test_dir)
    else:
        preds, fn = get_preds(models[0], backbones[0], test_dir, submission_dir)

    masks_to_submission(submission_filename=submission_dir,
                        full_mask_dir="full_masks/",
                        mask_dir="patched_masks/",
                        image_filenames=fn,
                        predictions=preds)


def get_ensemble_preds(models, backbones, device, test_dir):
    all_preds = []
    for model, backbone in zip(models, backbones):
        test_preds, fn = get_preds(model, backbone, device, test_dir)
        all_preds.append(test_preds)

    ensemble_preds = np.mean(all_preds, axis=0)
    return ensemble_preds, fn


def get_preds(model, backbone, device, test_dir):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(backbone, 'imagenet')
    preprocessing_fn = smp_get_preprocessing(preprocessing_fn)

    model.to(device=device)
    model.eval()

    test_dataset = TestDataset(
        data_dir=test_dir,
        transforms='validation',
        preprocess=preprocessing_fn,
        resize=416)

    test_loader = DataLoader(test_dataset,
                             batch_size=4,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True,
                             worker_init_fn=42,
                             )

    test_pred = []

    with torch.no_grad():
        for images in tqdm(test_loader):
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
                test_pred.append(mask)
    return test_pred, test_dataset.filenames


if __name__ == "__main__":
    model1 = smp.UnetPlusPlus(
        encoder_name='efficientnet-b7',
        encoder_weights='imagenet',
        decoder_channels=[256, 128, 64, 32, 16],
        decoder_attention_type=None,
        classes=1,
        activation=None,
    )

    state_dict = torch.load('model_checkpoints/UNetpp_B7_Final.pt', map_location=torch.device('cpu'))
    model1.load_state_dict(state_dict)

    model2 = smp.DeepLabV3Plus(
                encoder_name='timm-regnetx_160',
                encoder_weights='imagenet',
                classes=1,
                activation=None,
            )

    state_dict = torch.load('model_checkpoints/timm-regnetx_160-final-batch_deeplabv3plus.pt', map_location=torch.device('cpu'))
    model2.load_state_dict(state_dict)

    model3 = smp.PSPNet(
                encoder_name='timm-resnest200e',
                encoder_weights='imagenet',
                classes=1,
                activation=None,
            )

    state_dict = torch.load('model_checkpoints/timm-resnest200e_pspnet.pt', map_location=torch.device('cpu'))
    model3.load_state_dict(state_dict)

    models = [model1, model2, model3]
    backbones = ['efficientnet-b7', 'timm-regnetx_160', 'timm-resnest200e']

    submission_dir = 'submissions/Ensemble.csv'
    make_submission(models, backbones, 'cpu', 'data', submission_dir)
