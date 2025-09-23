import matplotlib.pyplot as plt
import random

# ------------------------------
# Visualization utility
# ------------------------------
def visualize_predictions(model, dataset, device, num_samples=3):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)

    plt.figure(figsize=(12, num_samples * 4))
    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)  # add batch dim

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu()

        # convert tensor back to numpy for plotting
        img_np = img[0].cpu().numpy()  # grayscale since you repeat channels
        mask_np = mask.squeeze().cpu().numpy()
        pred_np = pred.numpy()

        # Plot: input, target, prediction
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(img_np, cmap="gray")
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred_np, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()