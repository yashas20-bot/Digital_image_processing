import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

# Step 1: Load the image
img = cv2.imread(r"C:\Users\MITE\Desktop\DIP\batman_with_holes.png", 0)

# Step 2: Convert to binary
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Invert so Batman shape = white (foreground)
batman = cv2.bitwise_not(binary)

# Step 3: Create a glowing reveal animation
for alpha in np.linspace(0, 1, 40):  # 40 frames (smooth)
    glow = (batman * alpha).astype(np.uint8)
    
    # Optional: add a blur to create a “soft glow” effect
    blurred = cv2.GaussianBlur(glow, (9, 9), 0)
    
    # Combine glow + original for more cinematic look
    combined = cv2.addWeighted(glow, 1.0, blurred, 0.8, 0)
    
    # Show animation frame by frame
    clear_output(wait=True)
    plt.imshow(combined, cmap='gray')
    plt.title("🦇 Batman Logo Glow Reveal")
    plt.axis('off')
    plt.show(block=False)
    time.sleep(0.05)

# Step 4: Final logo shown bright and clear
plt.imshow(batman, cmap='gray')
plt.title("✅ Final Revealed Batman Logo")
plt.axis('off')
plt.show()
