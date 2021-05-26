import os
import random
import csv
import numpy as np
import tifffile
import imageio
from skimage.segmentation import find_boundaries


def generate_labels():
    masks_dir = 'data/BBBC010/masks'
    images_dir = 'data/BBBC010/images'
    labels_dir = 'data/BBBC010/labels'
    os.makedirs('data/BBBC010/labels', exist_ok=True)

    image_files = os.listdir(images_dir)
    image_files.sort()

    mask_files = os.listdir(masks_dir)
    mask_files.sort()
    mask_files = mask_files[1:]

    images = []
    gfp_masks = {}
    for i in range(0, len(image_files), 2):
        gfp_img = tifffile.imread(
            os.path.join(images_dir, image_files[i])
        )
        raw_img = tifffile.imread(
            os.path.join(images_dir, image_files[i+1])
        )
        akr = image_files[i][33:36]
        mask_names = [f for f in mask_files if akr in f]

        for mask_name in mask_names:
            mask = imageio.imread(
                os.path.join(masks_dir, mask_name)
            ).astype(np.int)
            gfp_mask = np.zeros_like(gfp_img)
            coords = np.where(mask == 255)
            gfp_mask[coords] = gfp_img[coords]
            num_pixels = len(coords[0])
            average_intensity = gfp_mask.sum() / num_pixels
            max_intensity = gfp_mask.max()
            label = 1 if average_intensity > 500 else 0
            path_to_file = os.path.join(
                labels_dir, mask_name[:6] + '_label.csv'
            )
            with open(path_to_file, mode='w') as csvfile:
                fields = ['label', 'average_intensity', 'max_intensity']
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {'label': label,
                     'average_intensity': average_intensity,
                     'max_intensity': max_intensity}
                )
                gfp_masks.update(
                    {mask_name: {'average_intensity': average_intensity,
                                 'max_intensity': max_intensity}}
                )


def load_data(masks_dir, labels_dir):
    mask_files = os.listdir(masks_dir)
    mask_files.sort()
    mask_files = mask_files[1:]

    label_files = os.listdir(labels_dir)
    label_files.sort()

    data = []
    for mask_name in mask_files:
        mask = imageio.imread(
            os.path.join(masks_dir, mask_name)
        )
        points = mask_to_points(mask)
        akr = mask_name[:6]
        label_name = [f for f in label_files if akr in f]
        assert len(label_name) == 1
        path_to_label = os.path.join(labels_dir, label_name[0])
        with open(path_to_label) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(csvreader):
                if i == 0:
                    continue
                else:
                    label = int(row[0])
        data += [{'points': points, 'label': label}]
    random.shuffle(data)
    return data


def mask_to_points(mask):
    boundary = find_boundaries(mask)
    points = np.where(boundary)
    points = np.concatenate(
        (points[0][None], points[1][None])
    ).T
    return points


class Center:
    """Centers node positions around the origin."""
    def __call__(self, tensor):
        tensor = tensor - tensor.mean(axis=-2, keepdims=True)
        return tensor


class RandomCoordsFlip:
    """
    This transform is used to flip sparse coords.
    """
    def __call__(self, tensor):
        axes = set(range(2)) - set({random.choice([0, 1])})
        for curr_ax in axes:
            coord_max = np.max(tensor[:, curr_ax])
            tensor[:, curr_ax] = coord_max - tensor[:, curr_ax]
        return tensor

