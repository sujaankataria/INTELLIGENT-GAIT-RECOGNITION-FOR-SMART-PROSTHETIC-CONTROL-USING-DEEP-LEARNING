from models.lstm import LSTMClassifier
from models.cnn_lstm import CNNLSTMClassifier
from models.tcn import TCNClassifier
from models.transformer import TransformerClassifier
from models.gru import GRUClassifier
from models.resnet1d import ResNet1DClassifier

def build_model(cfg, input_dim, num_classes):
    mtype = cfg["model"]["type"].lower()
    if mtype == "lstm":
        return LSTMClassifier(
            input_dim=input_dim, num_classes=num_classes,
            hidden_size=cfg["model"].get("hidden_size",128),
            num_layers=cfg["model"].get("num_layers",2),
            dropout=cfg["training"].get("dropout",0.3),
            bidirectional=cfg["model"].get("bidirectional",False),
        )
    elif mtype == "cnn_lstm":
        return CNNLSTMClassifier(
            input_dim=input_dim, num_classes=num_classes,
            conv_channels=cfg["model"].get("conv_channels",64),
            kernel_size=cfg["model"].get("kernel_size",5),
            hidden_size=cfg["model"].get("hidden_size",128),
            lstm_layers=cfg["model"].get("lstm_layers",1),
            dropout=cfg["training"].get("dropout",0.3),
            bidirectional=cfg["model"].get("bidirectional",False),
        )
    elif mtype == "tcn":
        return TCNClassifier(
            input_dim=input_dim, num_classes=num_classes,
            channels=tuple(cfg["model"].get("channels",[64,64,64,64,64,64])),
            kernel_size=cfg["model"].get("kernel_size",3),
            dilations=tuple(cfg["model"].get("dilations",[1,2,4,8,16,32])),
            dropout=cfg["training"].get("dropout",0.2),
        )
    elif mtype == "transformer":
        return TransformerClassifier(
            input_dim=input_dim, num_classes=num_classes,
            d_model=cfg["model"].get("d_model",128),
            nhead=cfg["model"].get("nhead",4),
            num_layers=cfg["model"].get("num_layers",4),
            dim_feedforward=cfg["model"].get("ffn_dim",256),
            dropout=cfg["training"].get("dropout",0.1),
        )
    elif mtype == "gru":
        m = cfg["model"]
        return GRUClassifier(
            input_dim=input_dim,
            hidden_size=int(m.get("hidden_size", 128)),
            num_layers=int(m.get("num_layers", 2)),
            bidirectional=bool(m.get("bidirectional", True)),
            # prefer model.dropout if provided; else fall back to training.dropout; else 0.2
            dropout=float(m.get("dropout", cfg.get("training", {}).get("dropout", 0.2))),
            num_classes=num_classes,
        )

    elif mtype == "resnet1d":
        m = cfg["model"]
        return ResNet1DClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            channels=tuple(m.get("channels", [64, 128, 128, 256])),
            kernel_size=int(m.get("kernel_size", 5)),
            dropout=float(m.get("dropout", cfg.get("training", {}).get("dropout", 0.2))),
        )
    else:
        raise ValueError(f"Unknown model.type: {mtype}")