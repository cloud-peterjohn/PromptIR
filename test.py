import os
import torch
from PIL import Image
import numpy as np
from model import PromptIR
import tqdm
from visualize import visualize


def load_img(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    t = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
    return t.unsqueeze(0)


def save_npz(images_dict, out_path):
    np.savez(out_path, **images_dict)
    print(f"Saved {len(images_dict)} images to {out_path}")


def test(model_path, test_dir="hw4_realse_dataset/test/degraded", output_dir="results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    model = PromptIR(decoder=True).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_npz = os.path.join(output_dir, "pred.npz")
    files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])
    images_dict = {}

    with torch.no_grad():
        for fname in tqdm.tqdm(files, desc="Testing", ncols=120):
            x = load_img(os.path.join(test_dir, fname)).to(device)
            out = model(x)
            out_img = torch.clamp(out, 0, 1).cpu().squeeze(0).numpy()
            out_img = (out_img * 255).round().astype(np.uint8)
            images_dict[fname] = out_img
            # print(f"Processed {fname}, shape: {out_img.shape}")
    save_npz(images_dict, out_npz)


if __name__ == "__main__":
    model_path = "checkpoints/your_model.pth"  # Replace with your model path
    test(model_path=model_path)
    visualize(model_path=model_path)
