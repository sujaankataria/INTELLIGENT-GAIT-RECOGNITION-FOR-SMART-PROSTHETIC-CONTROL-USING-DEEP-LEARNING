# Gait Project — All Experiments in One Folder

Run any experiment by pointing to a config in `configs/`. Models live in `models/`. Results go to `artifacts/EXP-XXX/`.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
# EXP-001 (LSTM)
python train.py configs/config_exp001.yaml

# EXP-002 (BiLSTM)
python train.py configs/config_exp002.yaml

# EXP-003 (CNN-LSTM)
python train.py configs/config_exp003.yaml

# EXP-004 (TCN)
python train.py configs/config_exp004.yaml

# EXP-005 (Transformer)
python train.py configs/config_exp005.yaml
```

## Add your data path
Edit each config's `dataset.root` to your HuGaDB folder.


## More Experiments

```bash
# EXP-006 (LSTM, placeholder for multi-task; classification-only config)
python train.py configs/config_exp006.yaml

# EXP-007 (TCN with augmentation: gaussian noise + time-mask)
python train.py configs/config_exp007.yaml

# EXP-008 (Transformer, LOSO-style split for subject 18)
python train.py configs/config_exp008.yaml
```

### Augmentation
You can toggle/train simple robustness augments per-config under:
```yaml
training:
  augmentation:
    enabled: true
    gaussian_noise_std: 0.03
    time_mask_pct: 0.08
```
