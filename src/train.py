import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


def train_loop(model, optimizer, criterion, train_loader, device='cpu'):
    running_loss = 0
    model.train()
    pbar = tqdm(train_loader, desc="Iterating over train data")

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # forward
        out = model(images)
        loss = criterion(out, masks)
        running_loss += loss.item()*images.shape[0]

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    running_loss /= len(train_loader.sampler)
    return running_loss


def eval_loop(model, criterion, eval_loader, threshold=False, device='cpu'):
    running_loss = 0
    model.eval()

    with torch.no_grad():
        accuracy, f1_scores = [], []
        pbar = tqdm(eval_loader, desc='Iterating over evaluation data')
        for imgs, masks in pbar:
            # pass to device
            imgs = imgs.to(device)
            masks = masks.to(device)

            # forward
            out = model(imgs)
            loss = criterion(out, masks)
            running_loss += loss.item()*imgs.shape[0]

            # calculate predictions using output
            # FIXME: this is not proper way to do it !
            if not threshold:
                predicted = out
            else:
                predicted = (out > threshold).float()

            predicted = predicted.view(-1).cpu().numpy()
            labels = masks.view(-1).cpu().numpy()
            accuracy.append(accuracy_score(labels, predicted))
            f1_scores.append(f1_score(labels, predicted))

    acc = sum(accuracy)/len(accuracy)
    f1 = sum(f1_scores)/len(f1_scores)
    running_loss /= len(eval_loader.sampler)
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'loss': running_loss}
