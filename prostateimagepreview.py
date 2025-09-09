import numpy as np
import matplotlib.pyplot as plt

image = np.load("C:\\Users\\DELL\\Downloads\\MSD_PROSTATE_PREPROCESSED\\image_014.npy")
label = np.load("C:\\Users\\DELL\\Downloads\\MSD_PROSTATE_PREPROCESSED\\label_014.npy")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image[:, :, image.shape[2]//2], cmap="gray")
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(label[:, :, label.shape[2]//2], cmap="gray")
plt.title("Label")

plt.show()
