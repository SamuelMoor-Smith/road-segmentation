import re
from test import create_predictions

PATCH_SIZE = 16  # pixels per side of square patches


def create_submission(submission_filename, model, device, apply_crf=False, used_224=False):
    test_pred, test_filenames = create_predictions(model, device, apply_crf=apply_crf, used_224=used_224)
    with open('./submissions/' + submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), test_pred):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j*PATCH_SIZE, i*PATCH_SIZE, int(patch_array[i, j])))
