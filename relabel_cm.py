import os
import numpy as np
import matplotlib.pyplot as plt

labels = [
    "down_by_elevator","going_down","going_up","running","sitting",
    "sitting_down","standing","standing_up","up_by_elevator","walking"
]

art = r".\artifacts\EXP-001"
cm_path = os.path.join(art, "confusion_matrix.csv")
cm = np.loadtxt(cm_path, delimiter=",")

plt.figure(figsize=(7.2, 6.4))
im = plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (labeled)")
plt.colorbar(im)
plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(labels)), labels, fontsize=8)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
out = os.path.join(art, "confusion_matrix_labeled.png")
plt.savefig(out, dpi=200)
print("Saved:", out)
