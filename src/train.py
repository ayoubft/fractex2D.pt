import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             jaccard_score, precision_score, recall_score,
                             roc_auc_score)
from tqdm.auto import tqdm

from .metrics import AE, MSE, PSNR, SSIM, iou_nobg


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
              threshold=.5, device='cpu', mdl=None):
    running_loss = 0
    model.eval()

    with torch.no_grad():
        mses, psnrs, ssims, aes, ious_nbg = [], [], [], [], []
        accs, f1s, roc_aucs, precs, recs = [], [], [], [], []
        senss, specs, dices = [], [], []
        pbar = tqdm(eval_loader, desc='Iterating over evaluation data')
        # metric_group = MetricGroup(num_classes=2, ignore_index=0)
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

            # calculate predictions using output
            predicted = out
            predicted_clf = (out > threshold).float().cpu()
            labels_clf = (labels > 0.).float().cpu()

            predicted = predicted.cpu()
            labels = labels.cpu()

            # metric_group.add(predicted, labels_clf.long())

            tn, fp, fn, tp = confusion_matrix(
                labels_clf.view(-1), predicted_clf.view(-1),
                labels=[0, 1]).ravel()
            senss.append(tp / (tp + fn + 1e-8))
            specs.append(tn / (tn + fp + 1e-8))
            dices.append(jaccard_score(
                labels_clf.view(-1), predicted_clf.view(-1)))

            mses.append(MSE()(predicted, labels).item())
            psnrs.append(PSNR()(predicted, labels).item())
            ssims.append(SSIM()(predicted, labels).item())
            aes.append(AE()(predicted, labels).item())

            accs.append(accuracy_score(
                labels_clf.view(-1), predicted_clf.view(-1)))
            f1s.append(f1_score(labels_clf.view(-1), predicted_clf.view(-1)))
            precs.append(precision_score(
                labels_clf.view(-1), predicted_clf.view(-1)))
            recs.append(recall_score(
                labels_clf.view(-1), predicted_clf.view(-1)))
            try:
                roc_aucs.append(
                    roc_auc_score(labels_clf.view(-1), predicted_clf.view(-1)))
            except ValueError:
                roc_aucs.append(-99)
            ious_nbg.append(iou_nobg(predicted_clf, labels_clf).item())

    # results = metric_group.value()
    # print(results)

    mse = sum(mses)/len(mses)
    psnr = sum(psnrs)/len(psnrs)
    ssim = sum(ssims)/len(ssims)
    ae = sum(aes)/len(aes)

    sens = sum(senss)/len(senss)
    spec = sum(specs)/len(specs)
    dice = sum(dices)/len(dices)

    acc = sum(accs)/len(accs)
    f1 = sum(f1s)/len(f1s)
    prec = sum(precs)/len(precs)
    rec = sum(recs)/len(recs)
    roc_auc = sum(roc_aucs)/len(roc_aucs)
    iou_nbg = sum(ious_nbg)/len(ious_nbg)
    running_loss /= len(eval_loader.sampler)

    if False:
        scheduler.step(mse)

    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'ae': ae,
        'acc': acc,
        'f1': f1,
        'prec': prec,
        'rec': rec,
        'spec': spec,
        'sens': sens,
        'dice': dice,
        'roc_auc': roc_auc,
        'iou_nbg': iou_nbg,
        'loss': running_loss,
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
    writer.add_scalar(f"SENS/{kind}", metrics['sens'], epoch)
    writer.add_scalar(f"DICE/{kind}", metrics['dice'], epoch)
    writer.add_scalar(f"AE/{kind}", metrics['ae'], epoch)
    writer.add_scalar(f"IoU_nbg/{kind}", metrics['iou_nbg'], epoch)
    writer.flush()
