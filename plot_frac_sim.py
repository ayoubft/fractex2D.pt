import itertools
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from skimage import io
from skimage.measure import label
from skimage.morphology import skeletonize
from skimage.segmentation import expand_labels
from skimage.util import view_as_windows

model_path = ('/users/afatihi/work-detect/fractex2D.pt/outputs_BM_/unet-huber-rmspro-0.1/2025-08-07_09-28')
# model_path = ('/users/afatihi/work-detect/fractex2D.pt/multirun_BM_/segformer/2025-08-30_16-50/model=sm_segformer')

model_name = 'test'


def convert_path(old_path, folder, ext='.tif'):
    dir_path, filename = os.path.split(old_path)
    base, _ = os.path.splitext(filename)
    new_filename = base + ext
    new_dir = os.path.join(dir_path, "test", folder)
    os.makedirs(new_dir, exist_ok=True)
    return os.path.join(new_dir, new_filename)


def remove_junctions(skel):
    skel = skel.astype(np.uint8)
    mask = np.zeros_like(skel)
    windows = view_as_windows(skel, (3, 3))
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            if windows[i, j].sum() > 4:  # junction or thick crossing
                mask[i:i+3, j:j+3] = 1
    return skel * (1 - mask)


def fracture_similarity(pred_mask, true_mask, save_path="fracture_comparison.png"):
    # skeletonize both masks
    pred_skel = skeletonize((pred_mask > 0.1).cpu().numpy())
    true_skel = skeletonize((true_mask > 0.1).cpu().numpy())

    # remove junctions to get clean fracture lines
    pred_clean = remove_junctions(pred_skel)
    true_clean = remove_junctions(true_skel)

    # label connected segments
    pred_labeled = label(pred_clean)
    true_labeled = label(true_clean)

    # compute fracture segment lengths
    pred_lengths = np.bincount(pred_labeled.ravel())[1:]
    true_lengths = np.bincount(true_labeled.ravel())[1:]

    # build comparable histograms
    bins = np.linspace(0, 500, 27)
    pred_hist, _ = np.histogram(pred_lengths, bins=bins)
    true_hist, _ = np.histogram(true_lengths, bins=bins)

    # chi-square distance between histograms
    pred_hist = pred_hist + 1e-6
    true_hist = true_hist + 1e-6
    chi_dist = 0.5 * np.sum(((pred_hist - true_hist) ** 2) / (pred_hist + true_hist))

    if chi_dist < 3 and pred_skel.sum() > 300 and true_skel.sum() > 300:

        # io.imsave(convert_path(save_path, 'image', '.png'), (image*255).astype(np.uint8))
        # io.imsave(convert_path(save_path, 'dem'), dem)
        # io.imsave(convert_path(save_path, 'gt'), true_mask.cpu().numpy())
        # io.imsave(convert_path(save_path, ), pred_mask.cpu().numpy())

        fig, axs = plt.subplots(4, 2, figsize=(10, 20))
        # Skeletons
        axs[0, 0].imshow(pred_skel, cmap='gray_r')
        axs[0, 0].set_title("Pred skeleton")
        axs[0, 1].imshow(true_skel, cmap='gray_r')
        axs[0, 1].set_title("True skeleton")

        # Cleaned (no junctions)
        axs[1, 0].imshow(pred_clean, cmap='gray_r')
        axs[1, 0].set_title("No junctions / Pred skeleton")
        axs[1, 1].imshow(true_clean, cmap='gray_r')
        axs[1, 1].set_title("No junctions / True skeleton")

        # Segmented fractures
        axs[2, 0].imshow(expand_labels(pred_labeled, 8), cmap='tab20_r')
        axs[2, 0].set_title("Pred segments")
        axs[2, 1].imshow(expand_labels(true_labeled, 8), cmap='tab20_r')
        axs[2, 1].set_title("True segments")

        # Length histograms
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # center positions of bins

        axs[3, 0].bar(bin_centers, pred_hist, width=np.diff(bins), align='center')
        axs[3, 0].set_title("Pred Length Distribution")
        axs[3, 0].set_xlabel("Fracture Length")
        axs[3, 0].set_ylabel("Count")
        axs[3, 0].set_xticks(bins[::2])
        axs[3, 0].set_xlim(0, 500)
        axs[3, 0].tick_params(axis='x', rotation=-45)

        axs[3, 1].bar(bin_centers, true_hist, width=np.diff(bins), align='center')
        axs[3, 1].set_title(f'True Length Dist | $\chi$ distance: {chi_dist:.2f}')
        axs[3, 1].set_xlabel("Fracture Length")
        axs[3, 1].set_ylabel("Count")
        axs[3, 1].set_xticks(bins[::2])
        axs[3, 1].set_xlim(0, 500)
        axs[3, 1].tick_params(axis='x', rotation=-45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close(fig)

    return chi_dist


@hydra.main(config_name="config.yaml",
            config_path=os.path.join(model_path, '.hydra'),
            version_base=None)
def main(cfg: DictConfig):

    datasets = ['samsu19', 'matteo21', 'ovaskainen23']

    for dataset in datasets:

        cfg.dataset.datasets = dataset
        cfg.batch_size = 1

        use_cuda = not cfg.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        save_path = model_path
        ##
        worst_patches_folder_name = f'frac_sim_plots/{cfg.dataset.datasets}'
        ##
        eval_path = os.path.join(save_path, worst_patches_folder_name)
        os.makedirs(eval_path, exist_ok=True)

        trainloader, valloader, testloader = instantiate(cfg.dataset)

        model = instantiate(cfg.model)
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'),
                              weights_only=True,
                              map_location=torch.device('cpu')))
        model = model.to(device)
        model.eval()

        for i, (image, label_) in enumerate(testloader):
            image = image.to(device)
            label_ = label_.to(device)

            out = model(image)
            predicted_clf = (out > cfg.threshold).float()

            fracture_similarity(
                predicted_clf.squeeze(0, 1), label_.squeeze(0, 1),
                os.path.join(save_path, f'{worst_patches_folder_name}/{i}.png')
                )
            # if i == 200:
            #     break


if __name__ == "__main__":
    main()
