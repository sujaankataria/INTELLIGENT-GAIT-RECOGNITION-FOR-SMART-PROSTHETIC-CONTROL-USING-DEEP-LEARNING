
import os, time, random, numpy as np, torch
from torch.utils.data import DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_loaders(train_ds, val_ds, test_ds, batch_size=64):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    )

def _apply_augment(x, cfg_aug):
    import torch
    if not cfg_aug or not cfg_aug.get("enabled", False):
        return x
    noise_std = float(cfg_aug.get("gaussian_noise_std", 0.0))
    mask_pct = float(cfg_aug.get("time_mask_pct", 0.0))
    y = x.clone()
    if noise_std > 0:
        y = y + torch.randn_like(y) * noise_std
    if mask_pct > 0:
        T = y.size(1)
        m = max(1, int(T * mask_pct))
        import random
        t0 = random.randint(0, max(0, T - m))
        y[:, t0:t0+m, :] = 0.0
    return y

def train_model(model, train_loader, val_loader, epochs, lr, weight_decay, device, out_dir,
                early_stop_patience=10, dropout=None, cfg_aug=None, class_weights=None):
    os.makedirs(out_dir, exist_ok=True)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_f1, patience = -1, 0
    best_path = os.path.join(out_dir, "best_model.pt")

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = _apply_augment(xb, cfg_aug)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        va_loss = 0.0; preds=[]; trues=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += loss.item() * xb.size(0)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                trues.append(yb.cpu().numpy())
        va_loss /= len(val_loader.dataset)
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        acc = accuracy_score(trues, preds); f1m = f1_score(trues, preds, average="macro")

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(acc)
        history["val_macro_f1"].append(f1m)

        print(f"Epoch {ep:03d}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={acc:.4f} val_f1={f1m:.4f}")

        if f1m > best_f1 + 1e-5:
            best_f1 = f1m
            torch.save(model.state_dict(), best_path)
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print("Early stopping.")
                break

    try:
        import matplotlib.pyplot as plt
        plt.figure(); plt.plot(history["train_loss"], label="train_loss"); plt.plot(history["val_loss"], label="val_loss")
        plt.legend(); plt.title("Loss"); plt.xlabel("epoch"); plt.ylabel("loss")
        plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200); plt.close()

        plt.figure(); plt.plot(history["val_acc"], label="val_acc"); plt.plot(history["val_macro_f1"], label="val_macro_f1")
        plt.legend(); plt.title("Val Metrics"); plt.xlabel("epoch")
        plt.savefig(os.path.join(out_dir, "val_metrics.png"), dpi=200); plt.close()
    except Exception as e:
        print("Plotting failed:", e)

    return best_path, history

def evaluate_model(model, loader, device, out_dir, class_names=None):
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)

    # Confusion matrix on full label space (for plotting)
    cm = confusion_matrix(trues, preds)

    # Filter to labels that occur in y_true to avoid UndefinedMetricWarning
    present = np.unique(trues)
    if class_names is not None:
        target_names = [class_names[i] for i in present]
    else:
        target_names = None

    report = classification_report(
        trues, preds,
        labels=present,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm.astype(int), fmt="%d", delimiter=",")
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.2,5.8))
        im = plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar(im)
        if class_names is not None:
            plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=8)
            plt.yticks(range(len(class_names)), class_names, fontsize=8)
        else:
            plt.xticks(range(cm.shape[1]))
            plt.yticks(range(cm.shape[0]))
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200); plt.close()
    except Exception as e:
        print("CM plotting failed:", e)

    return cm, report
