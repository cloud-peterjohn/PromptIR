import os
import torch
import numpy as np
from model import PromptIR
import matplotlib.pyplot as plt
import random


def visualize(
    model_path,
    test_dir="hw4_realse_dataset/test/degraded",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptIR(decoder=True).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])
    selected_files = random.sample(files, 30)

    fig, axes = plt.subplots(10, 6, figsize=(5.5, 12))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    with torch.no_grad():
        for idx, fname in enumerate(selected_files):
            x = load_img(os.path.join(test_dir, fname)).to(device)
            out = model(x)
            out_img = torch.clamp(out, 0, 1).cpu().squeeze(0).numpy()
            out_img = np.transpose(out_img, (1, 2, 0))

            degraded_img = x.cpu().squeeze(0).numpy()
            degraded_img = np.transpose(degraded_img, (1, 2, 0))

            row = (idx // 6) * 2
            col = idx % 6

            axes[row, col].imshow(degraded_img, aspect="auto")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel("Degraded", fontsize=14)

            axes[row + 1, col].imshow(out_img, aspect="auto")
            axes[row + 1, col].axis("off")
            if col == 0:
                axes[row + 1, col].set_ylabel("Restored", fontsize=14)

    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "visualization.svg")
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"Visualization saved to {save_path}")
    plt.show()
