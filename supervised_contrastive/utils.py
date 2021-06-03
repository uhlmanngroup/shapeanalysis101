import os
import numpy as np
import h5py
import tifffile
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.segmentation.boundaries import find_boundaries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


def load_data(wt_dir, lmna_dir):
    def _load_data(_dir):
        raw_files = os.listdir(
            os.path.join(_dir, 'raw_images')
        )
        raw_files.sort()
        data_dict = {}
        for f in raw_files:
            img = tifffile.imread(os.path.join(_dir, 'raw_images', f))
            mask_file = f[:7] + '_Object Identities.h5'
            with h5py.File(
                    os.path.join(_dir, 'instances', mask_file), 'r'
            ) as h5file:
                mask = h5file['exported_data'][:, :, 0]
            data_dict.update({f[:7] : {'raw': img, 'mask': mask}})
        return data_dict

    wt = _load_data(wt_dir)
    lmna = _load_data(lmna_dir)

    return wt, lmna


def visualize_data():
    wt_cell = tifffile.imread('data/MEF_LMNA/wildtype/raw_images/xy001c1.tif')
    wt_nuc = tifffile.imread('data/MEF_LMNA/wildtype/raw_images/xy001c2.tif')
    lmna_cell = tifffile.imread('data/MEF_LMNA/lmna_deficient/raw_images/xy001c1.tif')
    lmna_nuc = tifffile.imread('data/MEF_LMNA/lmna_deficient/raw_images/xy001c2.tif')
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(wt_cell, cmap='Greens')
    ax1.imshow(wt_nuc, cmap='Reds', alpha=0.5)
    ax1.axis('off')
    ax1.set_title('wild-type')
    ax2.imshow(lmna_cell, cmap='Greens')
    ax2.imshow(lmna_nuc, cmap='Reds', alpha=0.5)
    ax2.axis('off')
    ax2.set_title('lamin deficient')
    plt.show()


def show_image(path_to_cells, path_to_nuclei):
    cell_img = tifffile.imread(path_to_cells)
    nuc_img = tifffile.imread(path_to_nuclei)
    plt.axis('off')
    plt.imshow(cell_img, cmap='Greens')
    plt.imshow(nuc_img, cmap='Reds', alpha=0.5)
    plt.show()


def segmentation_to_pointcloud(masks):
    data = {}
    for k, mask in masks.items():
        boundaries = find_boundaries(mask)
        contours = find_contours(boundaries)
        for i, contour in enumerate(contours):
            _cs = [c for c in contours if len(c) > 300]
            cs = []
            for c in _cs:
               idx = np.random.choice(len(c), 120, replace=False)
               cs += [c[idx]]

            cs = np.concatenate(cs)
            with napari.gui_qt():
                viewer = napari.Viewer()
                viewer.add_labels(mask)
                viewer.add_points(cs, size=3.0)
            if len(contour) > 120:
                label = 'wildtype' if 'wildtype' in k else 'lmna'
                indices = np.random.choice(len(contour), 120, replace=False)
                contour = contour[indices]
                key = k + f'_{i}'
                data.update({key : contour})
    np.save('data/MEF_LMNA/mef_data.npy', data)
    data = np.load('data/MEF_LMNA/mef_data.npy', allow_pickle=True)


def run_logistic_regression(features, labels):
    skf = StratifiedKFold(n_splits=5)
    scores = []
    for train_idx, test_idx in skf.split(features, labels):
        logistic_regr = LogisticRegression(C=1, multi_class='auto', solver='lbfgs')
        logistic_regr.fit(features[train_idx], labels[train_idx])
        score = logistic_regr.score(features[test_idx], labels[test_idx])
        scores.append(score)

        preds = logistic_regr.predict(features[test_idx])
        conf_matrix = metrics.confusion_matrix(labels[test_idx], preds)
    return np.array(scores).mean()


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

class RandomNoise:
    """
    Isotropic additive gaussian noise
    """

    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, tensor):
        noise = self.sigma * np.random.randn(*tensor.shape)
        noise = noise.clip(-self.clip, self.clip)
        tensor = tensor + noise
        return tensor



if __name__ == '__main__':
    import napari

    wt_dir = 'data/MEF_LMNA/wildtype/'
    lmna_dir = 'data/MEF_LMNA/lmna_deficient/'

    wt, lmna = load_data(wt_dir, lmna_dir)

    images = []
    masks = {}

    for i, d in enumerate([wt, lmna]):
        for k, v in d.items():
            if 'c2' in k:
                continue
            images += [v['raw']]
            if i == 0:
                key = 'wildtype_' + k
            else:
                key = 'lmna_' + k
            masks.update(
                {key : v['mask']}
            )

    # masks = np.stack(masks)
    # images = np.stack(images)

    segmentation_to_pointcloud(masks)
    quit()

    # with napari.gui_qt():
    #     viewer = napari.view_labels(wt_mask, blending='additive')
    #     viewer.add_image(wt_img, blending='additive')
    #     viewer.add_labels(lmna_mask, blending='additive')
    #     viewer.add_image(lmna_img, blending='additive')
