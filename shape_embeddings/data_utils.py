import os
import csv
import numpy as np
import tifffile
import napari
import imageio
from skimage.measure import label
from scipy.ndimage.measurements import label


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
