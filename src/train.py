import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.morphology import label, skeletonize
from skimage.util import view_as_windows
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAUROC, BinaryCohenKappa, BinaryF1Score,
    BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinarySpecificity
)
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.segmentation import DiceScore
from tqdm.auto import tqdm


def remove_junctions(skel: np.ndarray) -> np.ndarray:
    """Remove junction points from a binary skeleton."""
    skel = skel.astype(np.uint8)
    mask = np.zeros_like(skel)
    windows = view_as_windows(skel, (3, 3))
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            if windows[i, j].sum() > 4:
                mask[i:i+3, j:j+3] = 1
    return skel * (1 - mask)


def fracture_similarity(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    """Compute similarity score between predicted and true fracture masks."""
    pred_skel = skeletonize((pred_mask > 0.1).cpu().numpy())
    true_skel = skeletonize((true_mask > 0.1).cpu().numpy())
    pred_clean = remove_junctions(pred_skel)
    true_clean = remove_junctions(true_skel)
    pred_labeled = label(pred_clean)
    true_labeled = label(true_clean)
    pred_lengths = np.bincount(pred_labeled.ravel())[1:]
    true_lengths = np.bincount(true_labeled.ravel())[1:]
    bins = np.linspace(0, 260, 20)
    pred_hist, _ = np.histogram(pred_lengths, bins=bins)
    true_hist, _ = np.histogram(true_lengths, bins=bins)
    pred_hist = pred_hist + 1e-6
    true_hist = true_hist + 1e-6
    chi_dist = 0.5 * np.sum((pred_hist - true_hist)**2 / (pred_hist + true_hist))
    return chi_dist


def train_loop(model, optimizer, criterion, train_loader, device='cpu', mdl=None):
    """Train the model for one epoch."""
    running_loss = 0
    model = model.to(device)
    model.train()
    pbar = tqdm(train_loader, desc="Iterating over train data")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        out = model(images)['out'] if mdl == 'fcn_resnet101' else model(images)
        loss = criterion(out, labels)
        running_loss += loss.item() * images.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    running_loss /= len(train_loader.sampler)
    return running_loss


def eval_loop(model, scheduler, criterion, eval_loader, threshold=0.5, device='cpu',
              mdl=None, ignore_index=None):
    """Evaluate the model on a validation or test dataset."""
    running_loss = 0
    model.eval()
    if ignore_index not in [0, 1]:
        ignore_index = None

    with torch.no_grad():
        # Metrics
        acc_metric = BinaryAccuracy(ignore_index=ignore_index).to(device)
        f1_metric = BinaryF1Score(ignore_index=ignore_index).to(device)
        prec_metric = BinaryPrecision(ignore_index=ignore_index).to(device)
        rec_metric = BinaryRecall(ignore_index=ignore_index).to(device)
        spec_metric = BinarySpecificity(ignore_index=ignore_index).to(device)
        auroc_metric = BinaryAUROC(ignore_index=ignore_index).to(device)
        iou_metric = BinaryJaccardIndex(ignore_index=ignore_index).to(device)
        dice_metric = DiceScore(num_classes=1, average="micro",
                                aggregation_level='global').to(device)
        ck_metric = BinaryCohenKappa().to(device)
        mse_metric = MeanSquaredError().to(device)
        ae_metric = MeanAbsoluteError().to(device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure().to(device)
        fracture_sim_scores = []

        pbar = tqdm(eval_loader, desc='Iterating over evaluation/test data')
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)['out'] if mdl == 'fcn_resnet101' else model(imgs)
            loss = criterion(out, labels)
            running_loss += loss.item() * imgs.shape[0]

            predicted = out
            if mdl == 'Segformer':
                predicted[predicted > 0.99] = 0.
            predicted_clf = (out > threshold).float()
            labels_clf = (labels > 0.).float()
            labels = labels.float()

            # Compute metrics
            acc_metric(predicted_clf, labels_clf)
            f1_metric(predicted_clf, labels_clf)
            prec_metric(predicted_clf, labels_clf)
            rec_metric(predicted_clf, labels_clf)
            spec_metric(predicted_clf, labels_clf)
            if labels_clf.numel() > 0 and labels_clf.min() != labels_clf.max():
                auroc_metric(predicted_clf, labels_clf)
            dice_metric(predicted_clf, labels_clf)
            iou_metric(predicted_clf, labels_clf)
            ck_metric(predicted_clf, labels_clf)
            mse_metric(predicted, labels)
            psnr_metric(predicted, labels)
            ssim_metric(predicted, labels)
            ae_metric(predicted, labels)

            for i in range(imgs.shape[0]):
                pred_mask = predicted_clf[i, 0].detach().cpu()
                true_mask = labels_clf[i, 0].detach().cpu()
                fracture_sim_scores.append(fracture_similarity(pred_mask, true_mask))

        avg_fracture_sim = float(np.mean(fracture_sim_scores)) if fracture_sim_scores else float('nan')

    return {
        'mse': mse_metric.compute().item(),
        'psnr': psnr_metric.compute().item(),
        'ssim': ssim_metric.compute().item(),
        'ae': ae_metric.compute().item(),
        'acc': acc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'prec': prec_metric.compute().item(),
        'rec': rec_metric.compute().item(),
        'spec': spec_metric.compute().item(),
        'dice': dice_metric.compute().item(),
        'iou': iou_metric.compute().item(),
        'ck': ck_metric.compute().item(),
        'roc_auc': auroc_metric.compute().item(),
        'loss': running_loss / len(eval_loader.sampler),
        'frac_sim': avg_fracture_sim,
    }


def eval_single(gt, pred, threshold=0.5, device="cpu", ignore_index=None):
    """Evaluate metrics for a single prediction and ground truth pair."""
    gt = torch.from_numpy(gt).to(device).float().unsqueeze(0).unsqueeze(0)
    pred = torch.from_numpy(pred).to(device).float().unsqueeze(0).unsqueeze(0)

    pred_clf = (pred > threshold).long()
    gt_clf = (gt > 0).long()
    if ignore_index not in [0, 1]:
        ignore_index = None

    # Metrics
    acc_metric = BinaryAccuracy(ignore_index=ignore_index).to(device)
    f1_metric = BinaryF1Score(ignore_index=ignore_index).to(device)
    prec_metric = BinaryPrecision(ignore_index=ignore_index).to(device)
    rec_metric = BinaryRecall(ignore_index=ignore_index).to(device)
    spec_metric = BinarySpecificity(ignore_index=ignore_index).to(device)
    auroc_metric = BinaryAUROC(ignore_index=ignore_index).to(device)
    iou_metric = BinaryJaccardIndex(ignore_index=ignore_index).to(device)
    dice_metric = DiceScore(num_classes=1, average="micro").to(device)
    ck_metric = BinaryCohenKappa().to(device)
    mse_metric = MeanSquaredError().to(device)
    ae_metric = MeanAbsoluteError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    # Compute metrics
    acc_metric(pred_clf, gt_clf)
    f1_metric(pred_clf, gt_clf)
    prec_metric(pred_clf, gt_clf)
    rec_metric(pred_clf, gt_clf)
    spec_metric(pred_clf, gt_clf)
    if gt_clf.numel() > 0 and gt_clf.min() != gt_clf.max():
        auroc_metric(pred, gt_clf.int())
    dice_metric(pred_clf, gt_clf)
    iou_metric(pred_clf, gt_clf)
    ck_metric(pred_clf, gt_clf)
    mse_metric(pred, gt)
    psnr_metric(pred, gt)
    ssim_metric(pred, gt)
    ae_metric(pred, gt)

    return {
        'mse': mse_metric.compute().item(),
        'psnr': psnr_metric.compute().item(),
        'ssim': ssim_metric.compute().item(),
        'ae': ae_metric.compute().item(),
        'acc': acc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'prec': prec_metric.compute().item(),
        'rec': rec_metric.compute().item(),
        'spec': spec_metric.compute().item(),
        'dice': dice_metric.compute().item(),
        'iou': iou_metric.compute().item(),
        'ck': ck_metric.compute().item(),
        'roc_auc': auroc_metric.compute().item(),
    }


def save_metrics(metrics: dict, kind: str, writer, epoch: int):
    """Log metrics to a TensorBoard writer."""
    writer.add_scalar(f"Loss/{kind}", metrics['loss'], epoch)
    writer.add_scalar(f"ACC/{kind}", metrics['acc'], epoch)
    writer.add_scalar(f"F1/{kind}", metrics['f1'], epoch)
    writer.add_scalar(f"PREC/{kind}", metrics['prec'], epoch)
    writer.add_scalar(f"REC/{kind}", metrics['rec'], epoch)
    writer.add_scalar(f"ROC_AUC/{kind}", metrics['roc_auc'], epoch)
    writer.add_scalar(f"MSE/{kind}", metrics['mse'], epoch)
    writer.add_scalar(f"PSNR/{kind}", metrics['psnr'], epoch)
    writer.add_scalar(f"SSIM/{kind}", metrics['ssim'], epoch)
    writer.add_scalar(f"SPEC/{kind}", metrics['spec'], epoch)
    writer.add_scalar(f"DICE/{kind}", metrics['dice'], epoch)
    writer.add_scalar(f"AE/{kind}", metrics['ae'], epoch)
    writer.add_scalar(f"IoU/{kind}", metrics['iou'], epoch)
    writer.flush()
