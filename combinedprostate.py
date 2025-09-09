import os 
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Set dataset paths
data_path = "C:\\Users\\DELL\\Downloads\\Task05_Prostate\\Task05_Prostate"
image_folder = os.path.join(data_path, "imagesTr")
label_folder = os.path.join(data_path, "labelsTr")

# Target shape (adjust if needed)
TARGET_SHAPE = (128, 128, 64)  

# Function to normalize images
def normalize_image(image):
    min_val, max_val = image.min(), image.max()
    return (image - min_val) / (max_val - min_val) if max_val > min_val else image

# Function to resize images
def resize_image(image, new_shape=TARGET_SHAPE):
    if len(image.shape) == 4:  # If shape is (1, H, W, D), remove extra dimension
        image = image[0]
    
    zoom_factors = [new_dim / old_dim for new_dim, old_dim in zip(new_shape, image.shape)]
    return zoom(image, zoom_factors, order=1)  # Linear interpolation

# Get all valid image and label files (ignore hidden files)
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".nii.gz") and not f.startswith("._")])
label_files = sorted([f for f in os.listdir(label_folder) if f.endswith(".nii.gz") and not f.startswith("._")])

print(f"Total Images: {len(image_files)}, Total Labels: {len(label_files)}")

# Create folders for preprocessed data
preprocessed_path = "C:\\Users\\DELL\\Downloads\\MSD_PROSTATE_PREPROCESSED"
os.makedirs(preprocessed_path, exist_ok=True)

# Loop through all images and process them
for i, (image_name, label_name) in enumerate(zip(image_files, label_files)):
    print(f"Processing {i+1}/{len(image_files)}: {image_name}")

    # Load NIfTI images
    image_nifti = nib.load(os.path.join(image_folder, image_name))
    label_nifti = nib.load(os.path.join(label_folder, label_name))

    # Convert to NumPy arrays
    image_data = image_nifti.get_fdata()
    label_data = label_nifti.get_fdata()

    # Debugging print: Check original shapes
    print(f"Original Image Shape: {image_data.shape}, Label Shape: {label_data.shape}")

    # Handle 4D images (remove extra dimension)
    if len(image_data.shape) == 4:
        image_data = image_data[0]
    if len(label_data.shape) == 4:
        label_data = label_data[0]

    # Normalize image
    image_data = normalize_image(image_data)

    # Resize image and label
    image_resized = resize_image(image_data)
    label_resized = resize_image(label_data)

    # Convert labels to binary (0 or 1)
    label_binary = (label_resized > 0).astype(np.uint8)

    # Scale up mask for visibility
    label_binary = label_binary * 255  # Converts from {0,1} to {0,255}

    # Save preprocessed files
    image_path = os.path.join(preprocessed_path, f"image_{i:03d}.npy")
    label_path = os.path.join(preprocessed_path, f"label_{i:03d}.npy")
    np.save(image_path, image_resized)
    np.save(label_path, label_binary)

    # Visualization for each processed image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_resized[:, :, image_resized.shape[2] // 2], cmap="gray")
    plt.title(f"Image {i+1}")

    plt.subplot(1, 2, 2)
    plt.imshow(label_binary[:, :, label_binary.shape[2] // 2], cmap="gray")
    plt.title(f"Label {i+1}")

    plt.show()

print("All images preprocessed and saved successfully!")
