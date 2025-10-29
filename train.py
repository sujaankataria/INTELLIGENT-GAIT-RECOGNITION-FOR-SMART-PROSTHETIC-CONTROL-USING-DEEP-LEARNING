import os, yaml, json, numpy as np, torch, sys
from utils.dataset_hugadb import collect_files, split_by_subject, WindowedIMUDataset
from utils.training import set_seed, make_loaders, train_model, evaluate_model
from utils.model_factory import build_model

def build_tensors(ds):
    import torch, numpy as np
    X = np.stack([x for x,_ in ds], axis=0)
    y = np.array([y for _,y in ds], dtype=np.int64)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def _safe_torch_load(path, map_location=None):
    import inspect, torch
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        return torch.load(path, map_location=map_location, weights_only=True)
    return torch.load(path, map_location=map_location)

def _compute_class_weights(y_tensor, num_classes):
    import numpy as np, torch
    y_np = y_tensor.cpu().numpy()
    counts = np.bincount(y_np, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

def main(cfg_path="configs/config_exp001.yaml"):
    with open(cfg_path, "r") as f:
        C = yaml.safe_load(f)

    out_dir = C["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    set_seed(C["training"]["seed"])

    files = collect_files(C["dataset"]["root"], C["dataset"]["pattern"])
    tr_ids, va_ids, te_ids = C["split"]["train_subjects"], C["split"]["val_subjects"], C["split"]["test_subjects"]
    tr_files, va_files, te_files = split_by_subject(files, C["dataset"]["subject_column"], tr_ids, va_ids, te_ids)

    T = int(C["preprocessing"]["window_seconds"] * C["preprocessing"]["sampling_rate_hz"])
    H = int(T * (1 - C["preprocessing"]["overlap"]))
    tr_ds = WindowedIMUDataset(tr_files, C["dataset"]["channels"], C["dataset"]["label_column"], T, H, C["preprocessing"]["normalization"])
    va_ds = WindowedIMUDataset(va_files, C["dataset"]["channels"], C["dataset"]["label_column"], T, H, C["preprocessing"]["normalization"])
    te_ds = WindowedIMUDataset(te_files, C["dataset"]["channels"], C["dataset"]["label_column"], T, H, C["preprocessing"]["normalization"])

    num_classes = len(set([y for _,y in tr_ds] + [y for _,y in va_ds] + [y for _,y in te_ds]))
    input_dim = len(C["dataset"]["channels"])

    Xtr, ytr = build_tensors(tr_ds)
    Xva, yva = build_tensors(va_ds)
    Xte, yte = build_tensors(te_ds)

    from torch.utils.data import TensorDataset
    train_loader, val_loader, test_loader = make_loaders(
        TensorDataset(Xtr, ytr), TensorDataset(Xva, yva), TensorDataset(Xte, yte), C["training"]["batch_size"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(C, input_dim=input_dim, num_classes=num_classes).to(device)

    # Optional class-weighted loss
    class_weights = None
    if C["training"].get("class_weighted", False):
        class_weights = _compute_class_weights(ytr, num_classes)
        print("Using class-weighted loss. Weights:", class_weights.tolist())

    best_path, hist = train_model(
        model, train_loader, val_loader, epochs=C["training"]["epochs"],
        lr=C["training"]["learning_rate"], weight_decay=C["training"]["weight_decay"],
        device=device, out_dir=out_dir, early_stop_patience=C["training"]["early_stop_patience"],
        cfg_aug=C["training"].get("augmentation", None),
        dropout=C["training"]["dropout"],
        class_weights=class_weights
    )

    state_dict = _safe_torch_load(best_path, map_location=device)
    model.load_state_dict(state_dict)

    class_names = [tr_ds.int_to_label[i] for i in range(len(tr_ds.int_to_label))]
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    cm, report = evaluate_model(model, test_loader, device, out_dir, class_names=class_names)

    print(report)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment_id": C["experiment_id"], "best_model_path": best_path, "history": hist}, f, indent=2)

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/config_exp001.yaml"
    main(cfg)
