import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from tqdm.auto import tqdm

from .metrics import AE, MSE, PSNR, SSIM


def train_loop(model, optimizer, criterion, train_loader, device='cpu',
               mdl=None):
    running_loss = 0
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
              threshold=False, device='cpu', mdl=None):
    running_loss = 0
    model.eval()

    with torch.no_grad():
        mses, psnrs, ssims, aes = [], [], [], []
        accs, f1s, roc_aucs, precs, recs = [], [], [], [], []
        pbar = tqdm(eval_loader, desc='Iterating over evaluation data')
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
            predicted_clf = (out > threshold).float()
            labels_clf = (labels > threshold).float()

            predicted = predicted.cpu()
            predicted_clf = predicted_clf.cpu()
            labels = labels.cpu()
            labels_clf = labels_clf.cpu()

            mses.append(MSE()(predicted, labels).item())
            psnrs.append(PSNR()(predicted, labels).item())
            ssims.append(SSIM()(predicted, labels).item())
            aes.append(AE()(predicted, labels).item())

            accs.append(accuracy_score(labels_clf.view(-1), predicted_clf.view(-1)))
            f1s.append(f1_score(labels_clf.view(-1), predicted_clf.view(-1)))
            precs.append(precision_score(labels_clf.view(-1), predicted_clf.view(-1)))
            recs.append(recall_score(labels_clf.view(-1), predicted_clf.view(-1)))
            roc_aucs.append(roc_auc_score(labels_clf.view(-1), predicted_clf.view(-1)))

    mse = sum(mses)/len(mses)
    psnr = sum(psnrs)/len(psnrs)
    ssim = sum(ssims)/len(ssims)
    ae = sum(aes)/len(aes)

    acc = sum(accs)/len(accs)
    f1 = sum(f1s)/len(f1s)
    prec = sum(precs)/len(precs)
    rec = sum(recs)/len(recs)
    roc_auc = sum(roc_aucs)/len(roc_aucs)
    running_loss /= len(eval_loader.sampler)

    # scheduler.step(mse)

    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'ae': ae,
        'acc': acc,
        'f1': f1,
        'prec': prec,
        'rec': rec,
        'roc_auc': roc_auc,
        'loss': running_loss,
        }
