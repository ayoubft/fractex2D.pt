import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.morphology import label, skeletonize
from skimage.util import view_as_windows
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC,
                                         BinaryCohenKappa, BinaryF1Score,
                                         BinaryJaccardIndex, BinaryPrecision,
                                         BinaryRecall, BinarySpecificity)
from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)
from torchmetrics.segmentation import DiceScore, HausdorffDistance, MeanIoU
from tqdm.auto import tqdm


def remove_junctions(skel):
    skel = skel.astype(np.uint8)
    mask = np.zeros_like(skel)
    windows = view_as_windows(skel, (3, 3))
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            if windows[i, j].sum() > 4:
                mask[i:i+3, j:j+3] = 1
    return skel * (1 - mask)


def fracture_similarity(pred_mask, true_mask):
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
    bins = np.linspace(0, 260, 20)
    pred_hist, _ = np.histogram(pred_lengths, bins=bins)
    true_hist, _ = np.histogram(true_lengths, bins=bins)

    # chi-square distance between histograms
    pred_hist = pred_hist + 1e-6
    true_hist = true_hist + 1e-6
    chi_dist = 0.5 * np.sum(((pred_hist - true_hist) ** 2) / (pred_hist + true_hist))
    return chi_dist


def train_loop(model, optimizer, criterion, train_loader, device='cpu',
               mdl=None):
    running_loss = 0
    model = model.to(device)
    model.train()
    pbar = tqdm(train_loader, desc="Iterating over train data")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # forward
        if mdl == 'fcn_resnet101':
            out = model(images)['out']
        else:
            out = model(images)

        loss = criterion(out, labels)
        running_loss += loss.item()*images.shape[0]

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    running_loss /= len(train_loader.sampler)
    return running_loss


def eval_loop(model, scheduler, criterion, eval_loader,
              threshold=.5, device='cpu', mdl=None, ignore_index=None):
    running_loss = 0
    model.eval()

    if ignore_index not in [0, 1]:
        ignore_index = None
    with torch.no_grad():
        acc_metric = BinaryAccuracy(ignore_index=ignore_index).to(device)
        f1_metric = BinaryF1Score(ignore_index=ignore_index).to(device)
        prec_metric = BinaryPrecision(ignore_index=ignore_index).to(device)
        rec_metric = BinaryRecall(ignore_index=ignore_index).to(device)
        spec_metric = BinarySpecificity(ignore_index=ignore_index).to(device)
        auroc_metric = BinaryAUROC(ignore_index=ignore_index).to(device)
        iou_metric = BinaryJaccardIndex(ignore_index=ignore_index).to(device)
        dice_metric = DiceScore(num_classes=1, average="micro",
                                aggregation_level='global').to(device)
        # hd_metric = HausdorffDistance(num_classes=1).to(device)
        ck_metric = BinaryCohenKappa().to(device)
        mse_metric = MeanSquaredError().to(device)
        ae_metric = MeanAbsoluteError().to(device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure().to(device)
        fracture_sim_scores = []

        pbar = tqdm(eval_loader, desc='Iterating over evaluation/test data')
        # metric_group = MetricGroup(num_classes=2, ignore_index=ignore_index)
        for imgs, labels in pbar:
            # pass to device
            imgs = imgs.to(device)
            labels = labels.to(device)

            # forward
            if mdl == 'fcn_resnet101':
                out = model(imgs)['out']
            else:
                out = model(imgs)
            loss = criterion(out, labels)
            running_loss += loss.item()*imgs.shape[0]

            predicted = out
            if mdl == 'Segformer':
                predicted[predicted > .99] = 0.
            predicted_clf = (out > threshold).float()
            labels_clf = (labels > 0.).float()
            labels = labels.float()

            # metric_group.add(predicted, labels_clf.long())

            # TO MAKE ALL ONES or ZEROS PREDICTIONS
            # predicted_clf = torch.zeros(labels_clf.shape).to(device)
            # print('**************************ALL zeros**************************')
            # predicted = predicted_clf

            acc_metric(predicted_clf, labels_clf)
            f1_metric(predicted_clf, labels_clf)
            prec_metric(predicted_clf, labels_clf)
            rec_metric(predicted_clf, labels_clf)
            spec_metric(predicted_clf, labels_clf)
            if labels_clf.numel() > 0 and labels_clf.min() != labels_clf.max():
                auroc_metric(predicted_clf, labels_clf)
            dice_metric(predicted_clf, labels_clf)
            iou_metric(predicted_clf, labels_clf)
            # hd_metric(predicted_clf, labels_clf)
            ck_metric(predicted_clf, labels_clf)
            mse_metric(predicted, labels)
            psnr_metric(predicted, labels)
            ssim_metric(predicted, labels)
            ae_metric(predicted, labels)
            for i in range(imgs.shape[0]):
                pred_mask = predicted_clf[i, 0].detach().cpu()
                true_mask = labels_clf[i, 0].detach().cpu()
                sim_score = fracture_similarity(pred_mask, true_mask)
                fracture_sim_scores.append(sim_score)

        avg_fracture_sim = float(np.mean(fracture_sim_scores)) if len(fracture_sim_scores) > 0 else float('nan')

    # results = metric_group.value()
    # print(results)
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
        # 'hd': hd_metric.compute().item(),
        'ck': ck_metric.compute().item(),
        'roc_auc': auroc_metric.compute().item(),
        'loss': running_loss / len(eval_loader.sampler),
        'frac_sim': avg_fracture_sim,
    }


def eval_single(gt, pred, threshold=0.5, device="cpu", ignore_index=None):
    """
    gt:   torch.Tensor (B,C,H,W) or (H,W), ground truth binary mask
    pred: torch.Tensor (B,C,H,W) or (H,W), predicted probabilities or logits
    """
    gt = torch.from_numpy(gt).to(device).float().unsqueeze(0).unsqueeze(0)
    pred = torch.from_numpy(pred).to(device).float().unsqueeze(0).unsqueeze(0)

    # threshold
    pred_clf = (pred > threshold).long()
    gt_clf = (gt > 0).long()

    if ignore_index not in [0, 1]:
        ignore_index = None

    # metrics
    acc_metric = BinaryAccuracy(ignore_index=ignore_index).to(device)
    f1_metric = BinaryF1Score(ignore_index=ignore_index).to(device)
    prec_metric = BinaryPrecision(ignore_index=ignore_index).to(device)
    rec_metric = BinaryRecall(ignore_index=ignore_index).to(device)
    spec_metric = BinarySpecificity(ignore_index=ignore_index).to(device)
    auroc_metric = BinaryAUROC(ignore_index=ignore_index).to(device)
    iou_metric = BinaryJaccardIndex(ignore_index=ignore_index).to(device)
    dice_metric = DiceScore(num_classes=1, average="micro").to(device)
    # hd_metric = HausdorffDistance(num_classes=1).to(device)
    ck_metric = BinaryCohenKappa().to(device)
    mse_metric = MeanSquaredError().to(device)
    ae_metric = MeanAbsoluteError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    # update
    acc_metric(pred_clf, gt_clf)
    f1_metric(pred_clf, gt_clf)
    prec_metric(pred_clf, gt_clf)
    rec_metric(pred_clf, gt_clf)
    spec_metric(pred_clf, gt_clf)
    if gt_clf.numel() > 0 and gt_clf.min() != gt_clf.max():
        auroc_metric(pred, gt_clf.int())  # use probs for AUROC
    dice_metric(pred_clf, gt_clf)
    iou_metric(pred_clf, gt_clf)
    # hd_metric(pred_clf, gt_clf)
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
        # 'hd': hd_metric.compute().item(),
        'ck': ck_metric.compute().item(),
        'roc_auc': auroc_metric.compute().item(),
    }


def save_metrics(metrics, kind, writer, epoch):
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
    # writer.add_scalar(f"SENS/{kind}", metrics['sens'], epoch)
    writer.add_scalar(f"DICE/{kind}", metrics['dice'], epoch)
    writer.add_scalar(f"AE/{kind}", metrics['ae'], epoch)
    writer.add_scalar(f"IoU/{kind}", metrics['iou'], epoch)
    writer.flush()
