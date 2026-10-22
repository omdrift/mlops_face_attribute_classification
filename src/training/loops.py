"""Training and validation share the same fixed five-task loss."""
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm

from src.data.splits import ATTRIBUTES


def task_loss(outputs, targets):
    return sum(nn.functional.binary_cross_entropy_with_logits(outputs[k], targets[k].float())
               if classes == 2 else nn.functional.cross_entropy(outputs[k], targets[k].long())
               for k, (_, classes) in ATTRIBUTES.items())


def predictions(outputs):
    return {k: (torch.sigmoid(value) > 0.5).long() if ATTRIBUTES[k][1] == 2 else value.argmax(1)
            for k, value in outputs.items()}


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    loss_sum, count = 0.0, 0
    for images, targets in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        images = images.to(device)
        targets = {k: value.to(device) for k, value in targets.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(model(images), targets)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * len(images)
        count += len(images)
    if count == 0:
        raise ValueError("Empty training partition")
    return loss_sum / count


def validate(model, loader, device):
    model.eval()
    loss_sum, count = 0.0, 0
    trues, preds = {k: [] for k in ATTRIBUTES}, {k: [] for k in ATTRIBUTES}
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = {k: value.to(device) for k, value in targets.items()}
            outputs = model(images)
            loss_sum += task_loss(outputs, targets).item() * len(images)
            count += len(images)
            for k, value in predictions(outputs).items():
                preds[k].extend(value.cpu().tolist())
                trues[k].extend(targets[k].cpu().tolist())
    if count == 0:
        raise ValueError("Empty validation partition")
    return loss_sum / count, {k: float(accuracy_score(trues[k], preds[k])) for k in ATTRIBUTES}
