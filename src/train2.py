import torch
from tqdm.auto import tqdm

from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall,
    BinarySpecificity, BinaryAUROC, BinaryJaccardIndex, BinaryCohenKappa
)
from torchmetrics.segmentation import DiceScore, MeanIoU, HausdorffDistance
from torchmetrics.image import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
)
from torchmetrics import MeanSquaredError, MeanAbsoluteError


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
                print('WE are doing it')
                predicted[predicted > .99] = 0.
            predicted_clf = (out > threshold).float()
            labels_clf = (labels > 0.).float()
            labels = labels.float()

            # metric_group.add(predicted, labels_clf.long())

            # TO MAKE ALL ONES or ZEROS PREDICTIONS
            # predicted_clf = torch.zeros(labels_clf.shape).to(device)
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
