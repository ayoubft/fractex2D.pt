import torch

from src.dataset_benchm import MATTEO, OVAS, SAMSU

EXPAND, DILATE = True, True

datasets = ['OVAS', 'MATTEO', 'SAMSU']
subsets = ['train', 'valid', 'test']
results = {}

for ds_name in datasets:
    ds_class = globals()[ds_name]  # assumes the dataset classes are imported
    results[ds_name] = {}
    for subset in subsets:
        dataset = ds_class(subset=subset, expand=EXPAND, dilate=DILATE)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

        ones = 0
        total = 0
        for _, maks in loader:
            ones += (maks == 1).sum().item()
            total += maks.numel()

        results[ds_name][subset] = {'ones': ones, 'total': total}

print('with pre-processing')
print(results)
