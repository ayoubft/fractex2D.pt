import torch
from .metrics import MSE, PSNR, SSIM, AE
from tqdm.auto import tqdm


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
            # FIXME: this is not proper way to do it !
            if not threshold:
                predicted = out
            else:
                predicted = (out > threshold).float()

            predicted = predicted.cpu()  # .numpy()
            labels = labels.cpu()  # .numpy()

            mses.append(MSE()(predicted, labels).item())
            psnrs.append(PSNR()(predicted, labels).item())
            ssims.append(SSIM()(predicted, labels).item())
            aes.append(AE()(predicted, labels).item())

    mse = sum(mses)/len(mses)
    psnr = sum(psnrs)/len(psnrs)
    ssim = sum(ssims)/len(ssims)
    ae = sum(aes)/len(aes)
    running_loss /= len(eval_loader.sampler)

    scheduler.step(mse)

    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'ae': ae,
        'loss': running_loss}
