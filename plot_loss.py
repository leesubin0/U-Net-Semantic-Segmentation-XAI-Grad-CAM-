import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]

train_loss = [2.0059, 1.3876, 1.1290, 0.9744, 0.8883]
val_loss = [1.7390, 1.3112, 0.9962, 0.8683, 0.8317]

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")

plt.title("UNet Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.show()

print("loss_curve.png 저장 완료")