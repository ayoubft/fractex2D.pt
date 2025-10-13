import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

model_path = ('//users/afatihi/work-detect/fractex2D.pt/outputs_BM_/unet-huber-rmspro-0.1/2025-08-07_09-28')

model_name = 'test'


@hydra.main(config_name="config.yaml",
            config_path=os.path.join(model_path, '.hydra'),
            version_base=None)
def main(cfg: DictConfig):

    datasets = ['ovaskainen23']  #['samsu19', 'matteo21', 'ovaskainen23']
    aims = ['fp', 'fn']
    J = 100

    for what, dataset in itertools.product(aims, datasets):

        cfg.dataset.datasets = dataset
        cfg.batch_size = 1

        use_cuda = not cfg.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        save_path = model_path
        ##
        worst_patches_folder_name = f'worst_preds_1/wpatches_{what}_{cfg.dataset.datasets}'
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

        diffs_paths = {}

        for i, (image, label) in enumerate(testloader):
            image = image.to(device)
            label = label.to(device)

            out = model(image)

            predicted_clf = (out > cfg.threshold).float()
            label_clf = (label > 0.).float()

            # diff = torch.sum(torch.abs( - label_clf))

            tn, fp, fn, tp = confusion_matrix(
                label_clf.to('cpu').flatten(),
                predicted_clf.to('cpu').flatten(),
                labels=[0, 1]).ravel()

            if what == 'fp':
                diff = fp
            if what == 'fn':
                diff = fn

            diffs_paths[i] = {
                'diff': diff,
                'path': f'{save_path}/{worst_patches_folder_name}/{i}.png'
            }

            # print(diffs_paths)

            #############
            image = image.squeeze(0).permute(1, 2, 0)[:, :, :3]
            label = label.squeeze(0).permute(1, 2, 0)
            out = out.squeeze(0).permute(1, 2, 0)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(image.to('cpu').numpy())
            # axes[0].set_title('Image')
            axes[0].set_yticks([])
            axes[0].set_xticks([])
            # axes[0].axis('off')
            # axes[0].grid([])

            axes[1].imshow(label.to('cpu').max() - label.to('cpu'), cmap='gray')
            # axes[1].set_title('Ground truth')
            axes[1].set_yticks([])
            axes[1].set_xticks([])

            axes[2].imshow(out.detach().to('cpu').max() - out.detach().to('cpu'), cmap='gray')
            # axes[2].set_title('Prediction')
            axes[2].set_yticks([])
            axes[2].set_xticks([])

            fig.tight_layout()
            fig.savefig(os.path.join(
                save_path, f'{worst_patches_folder_name}/{i}.png'))
            plt.close()
            #################

            if i > J:
                sorted_items = sorted(
                    diffs_paths.items(),
                    key=lambda item: item[1]['diff'], reverse=True)
                top_10_items = dict(sorted_items[:J])

                for key, value in diffs_paths.items():
                    top_10_paths = [i['path'] for i in top_10_items.values()]
                    if value['path'] not in top_10_paths:
                        try:
                            os.remove(value['path'])  # Actual file deletion
                            print(f"Deleted: {value['path']}")
                        except FileNotFoundError:
                            print(f"File not found: {value['path']}")
                        except Exception as e:
                            print(f"Error deleting {value['path']}: {e}")

                diffs_paths.clear()
                diffs_paths.update(top_10_items)

            print(f'Sample {i}: OK!')

        print('Done!')


if __name__ == "__main__":
    main()
