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


def mask_to_submission_strings(image_filename, im_arr, mask_dir=None,full_mask_dir=None):
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im_arr = (im_arr.reshape(400,400)*255.0).astype(np.uint8)
    patch_size = 16
    mask = np.zeros_like(im_arr)
    for j in range(0, im_arr.shape[1], patch_size):
        for i in range(0, im_arr.shape[0], patch_size):
            patch = im_arr[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            mask[i:i+patch_size, j:j+patch_size] = int(label*255)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

    if mask_dir:
        save_mask_as_img(mask, os.path.join(mask_dir, "mask_" + image_filename))
    if full_mask_dir:
        save_mask_as_img(im_arr, os.path.join(full_mask_dir, "mask_" + image_filename))


def save_mask_as_img(img_arr, mask_filename):
    img = PIL.Image.fromarray(img_arr)
    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    img.save(mask_filename)


def masks_to_submission(submission_filename, image_filenames,predictions,full_mask_dir=None, mask_dir=None):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn,pred in zip(image_filenames,predictions):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image_filename=fn,im_arr=pred, mask_dir=mask_dir,full_mask_dir=full_mask_dir))


def main(model, device, backbone, test_dir):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(backbone, 'imagenet')
    preprocessing_fn = smp_get_preprocessing(preprocessing_fn)

    model.to(device=device)
    model.eval()

    test_dataset = TestDataset(
        data_dir=test_dir,
        transforms='minimal',
        preprocess=preprocessing_fn,
        resize=416)

    test_loader = DataLoader(test_dataset,
                             batch_size=4,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True,
                             worker_init_fn=42,
                             )

    imgs_lst = []
    preds_lst = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            preds = model.predict(images)
            predicted_masks = preds.permute(0, 2, 3, 1).cpu().numpy() #permutes similar to reshape, such that [batch_size,height,width,channels]
            images = images.permute(0, 2, 3, 1).cpu().numpy() #permutes similar to reshape, such that [batch_size,height,width,channels]

            # Unet needed 416,416 we need back 400,400
            crop = A.Compose([
                A.CenterCrop(height=400, width=400),
            ])

            for image, mask in zip(images, predicted_masks):
                cropped = crop(image=image, mask=mask)
                image = cropped["image"]
                mask = cropped["mask"]
                imgs_lst.append(image)
                preds_lst.append(mask)

    masks_to_submission(submission_filename="submissions/dummy_submission.csv",
                        full_mask_dir="full_masks/",
                        mask_dir="patched_masks/",
                        image_filenames=test_dataset.filenames,
                        predictions=preds_lst)


if __name__ == "__main__":

    model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b5',
            encoder_weights='imagenet',
            decoder_channels=[256, 128, 64, 32, 16],
            decoder_attention_type=None,
            classes=1,
            activation='sigmoid',
        )

    device = 'cpu'
    state_dict = torch.load('model_checkpoints/Unetpp.pt', map_location=torch.device('cpu'))

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    backbone = 'efficientnet-b5'
    test_dir = 'data'
    main(model, device, backbone, test_dir)