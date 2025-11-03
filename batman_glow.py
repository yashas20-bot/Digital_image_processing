import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the image (use full path)
img = cv2.imread(r"C:\Users\Admin\Desktop\DIP\Digital_image_processing-main\batman_with_holes.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found! Double-check your path and filename.")

# Step 1: Convert to binary (tune threshold for best result)
_, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

# Step 2: Define a larger structuring element to fill more holes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# Step 3: Perform flood-fill to fill background
im_floodfill = binary.copy()
h, w = binary.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Fill from top-left corner
cv2.floodFill(im_floodfill, mask, (0, 0), 255)

# Step 4: Invert the floodfilled image and OR it with the original to fill holes
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
filled = binary | im_floodfill_inv

# Step 5: Apply morphological closing to fill remaining gaps
filled_cleaned = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=2)

# Step 6: Visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 6))

# Original (binary)
axes[0].imshow(binary, cmap='gray')
axes[0].set_title("Original (With Holes)")
axes[0].axis('off')

# Filled image
axes[1].imshow(filled_cleaned, cmap='gray')
axes[1].set_title("Filled Image (More Regions Filled)")
axes[1].axis('off')

# Red overlay (shows filled regions)
overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
filled_overlay = cv2.cvtColor(filled_cleaned, cv2.COLOR_GRAY2BGR)
red_mask = cv2.subtract(filled_cleaned, binary)
red_overlay = filled_overlay.copy()
red_overlay[red_mask > 0] = [255, 0, 0]  # red = newly filled parts

axes[2].imshow(red_overlay)
axes[2].set_title("Red Overlay = Filled Areas")
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(" Hole filling completed successfully with enhanced coverage!")
