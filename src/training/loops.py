from  sklearn.metrics import accuracy_score


# Third-party imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import nn
import torch

bce = nn.BCEWithLogitsLoss()
ce = nn.CrossEntropyLoss()

# Suppress tqdm warnings if any
tqdm.pandas()


def train_one_epoch(model, loader, optimizer, device, epoch):
    """Version stabilisée de l'entraînement"""
    model.train()
    running_loss = 0
    running_losses = {"beard": 0, "mustache": 0, "glasses": 0, "hair_color": 0, "hair_length": 0}

    for batch_idx, (imgs, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        imgs = imgs.to(device)
        t_beard = targets["beard"].to(device)
        t_mustache = targets["mustache"].to(device)
        t_glasses = targets["glasses"].to(device)
        t_color = targets["hair_color"].to(device)
        t_length = targets["hair_length"].to(device)

        # Forward pass
        outs = model(imgs)

        loss_beard = bce(outs["beard"], t_beard)
        loss_mustache = bce(outs["mustache"], t_mustache)
        loss_glasses = bce(outs["glasses"], t_glasses)
        loss_color = ce(outs["hair_color"], t_color)
        loss_length = ce(outs["hair_length"], t_length)

        #PONDÉRATION
        total_loss = (
            loss_beard * 1.0 +
            loss_mustache * 1.0 +
            loss_glasses * 1.0 +
            loss_color * 1.0 +
            loss_length * 1.0
        )

        # Backward pass avec gradient clipping
        optimizer.zero_grad()
        total_loss.backward()

        # GRADIENT CLIPPING
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += total_loss.item()
        running_losses["beard"] += loss_beard.item()
        running_losses["mustache"] += loss_mustache.item()
        running_losses["glasses"] += loss_glasses.item()
        running_losses["hair_color"] += loss_color.item()
        running_losses["hair_length"] += loss_length.item()

        # LOGGING BATCH (seulement 1 batch / epoch)
        if batch_idx == 0:
            print(f" BATCH 0 - Losses:")
            print(f"   beard: {loss_beard.item():.4f}")
            print(f"   mustache: {loss_mustache.item():.4f}")
            print(f"   glasses: {loss_glasses.item():.4f}")
            print(f"   color: {loss_color.item():.4f}")
            print(f"   length: {loss_length.item():.4f}")
            print(f"   TOTAL: {total_loss.item():.4f}")

            # Prédictions du batch
            with torch.no_grad():
                beard_acc = ((torch.sigmoid(outs["beard"]) > 0.5).float() == t_beard).float().mean().item()
                mustache_acc = ((torch.sigmoid(outs["mustache"]) > 0.5).float() == t_mustache).float().mean().item()
                glasses_acc = ((torch.sigmoid(outs["glasses"]) > 0.5).float() == t_glasses).float().mean().item()

                print(f" BATCH 0 - Prédictions:")
                print(f"   beard: acc={beard_acc:.3f}")
                print(f"   mustache: acc={mustache_acc:.3f}")
                print(f"   glasses: acc={glasses_acc:.3f}")

    # Calcul des moyennes
    avg_total_loss = running_loss / len(loader)
    for k in running_losses:
        running_losses[k] /= len(loader)

    print(f" MOYENNES EPOCH - Total: {avg_total_loss:.4f}")
    for attr, loss_val in running_losses.items():
        print(f"   {attr:12}: {loss_val:.4f}")

    return avg_total_loss

def validate(model, loader, device):
    """Version stabilisée de la validation"""
    model.eval()
    running_loss = 0
    all_preds = {"beard":[], "mustache":[], "glasses":[], "hair_color":[], "hair_length":[]}
    all_trues = {"beard":[], "mustache":[], "glasses":[], "hair_color":[], "hair_length":[]}

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Validation"):
            imgs = imgs.to(device)
            t_beard = targets["beard"].to(device)
            t_mustache = targets["mustache"].to(device)
            t_glasses = targets["glasses"].to(device)
            t_color = targets["hair_color"].to(device)
            t_length = targets["hair_length"].to(device)

            outs = model(imgs)

            # Même calcul de loss qu'à l'entraînement
            loss_beard = bce(outs["beard"], t_beard)
            loss_mustache = bce(outs["mustache"], t_mustache)
            loss_glasses = bce(outs["glasses"], t_glasses)
            loss_color = ce(outs["hair_color"], t_color)
            loss_length = ce(outs["hair_length"], t_length)

            total_loss = (
                loss_beard * 1.0 +
                loss_mustache * 1.0 +
                loss_glasses * 1.0 +
                loss_color * 1.0 +
                loss_length * 1.0
            )

            running_loss += total_loss.item()

            # Collecte des prédictions
            pb = (torch.sigmoid(outs["beard"]) > 0.5).cpu().numpy().astype(int)
            pm = (torch.sigmoid(outs["mustache"]) > 0.5).cpu().numpy().astype(int)
            pg = (torch.sigmoid(outs["glasses"]) > 0.5).cpu().numpy().astype(int)
            pc = outs["hair_color"].argmax(dim=1).cpu().numpy()
            pl = outs["hair_length"].argmax(dim=1).cpu().numpy()

            all_preds["beard"].extend(pb.tolist())
            all_preds["mustache"].extend(pm.tolist())
            all_preds["glasses"].extend(pg.tolist())
            all_preds["hair_color"].extend(pc.tolist())
            all_preds["hair_length"].extend(pl.tolist())

            all_trues["beard"].extend(t_beard.cpu().numpy().astype(int).tolist())
            all_trues["mustache"].extend(t_mustache.cpu().numpy().astype(int).tolist())
            all_trues["glasses"].extend(t_glasses.cpu().numpy().astype(int).tolist())
            all_trues["hair_color"].extend(t_color.cpu().numpy().tolist())
            all_trues["hair_length"].extend(t_length.cpu().numpy().tolist())

    # Calcul des accuracies
    accs = {}
    for k in all_preds:
        accs[k] = accuracy_score(all_trues[k], all_preds[k])

    return running_loss / len(loader), accs