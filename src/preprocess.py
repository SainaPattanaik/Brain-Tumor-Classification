import os
from pathlib import Path
from PIL import Image, ImageChops
import numpy as np
import random
# NOTE: We can use PIL and NumPy to simulate basic morphological operations
from scipy.ndimage import binary_fill_holes # Adding this common scientific library for robust mask cleaning

DATA_DIR = Path("./data")  # run from src/
TARGET_SIZE = (224, 224)

# --- Advanced Preprocessing Functions ---

def skull_strip_simple(img: Image.Image) -> Image.Image:
    """
    Conceptual skull stripping function using robust thresholding and mask cleaning.

    In a real-world scenario, this would involve complex algorithms 
    or dedicated libraries (e.g., SimpleITK, HD-BET) to accurately separate 
    the brain tissue from the skull and surrounding areas.
    
    This implementation uses median-based thresholding for robustness and fills 
    small internal holes in the mask (a common artifact when isolating brain tissue).
    """
    # Convert to grayscale NumPy array for processing
    img_array = np.array(img.convert("L"))
    
    # Calculate a more robust threshold
    img_array_flat = img_array.flatten()
    
    # Filter out near-black background pixels (intensity < 10)
    foreground_pixels = img_array_flat[img_array_flat > 10]
    
    if foreground_pixels.size == 0:
        # If no significant pixels are found, return the original image
        return img 
        
    # CRITICAL FIX: Use the median of the foreground pixels as a robust threshold base.
    # The median is less sensitive to high-intensity noise or bright skull/fat.
    # We use a factor (e.g., 0.6) to keep only the brightest core (the brain).
    threshold = np.median(foreground_pixels) * 0.6 
    
    # Set a minimum threshold to avoid aggressive stripping in very dark images
    threshold = max(30, threshold) 
    
    # 1. Create the binary mask
    mask_np = img_array > threshold
    
    # 2. Mask Cleaning (Fill Holes)
    # Filling holes helps to include internal areas of the brain that might 
    # have been falsely excluded by thresholding (like CSF spaces or low-intensity tumors).
    try:
        mask_np = binary_fill_holes(mask_np)
    except NameError:
        # Fallback if scipy.ndimage is not available (less robust)
        pass 

    # Find coordinates of the non-background area
    coords = np.argwhere(mask_np)
    if coords.size == 0:
        return img # Return original if the mask is empty

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Create a full-size mask image from the boolean array
    full_mask = Image.fromarray((mask_np * 255).astype(np.uint8))
    
    # Apply mask to the original RGB image (composite layers)
    # This sets everything outside the mask to black (0, 0, 0)
    stripped_img = Image.composite(img, Image.new('RGB', img.size, (0, 0, 0)), full_mask)
    
    # Crop to the bounding box of the content for normalization
    return stripped_img.crop((x_min, y_min, x_max, y_max))


def augment_image(img: Image.Image, base_path: str):
    """Applies basic geometric augmentations and saves new image files."""
    
    # 1. Horizontal Flip
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_img.save(f"{base_path}_aug_flip.jpg")

    # 2. Random Rotation (e.g., -10 to +10 degrees)
    angle = random.uniform(-10, 10)
    # Use Image.BILINEAR for resampling and filling empty areas (no expand)
    rotated_img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0,0,0)) 
    rotated_img.save(f"{base_path}_aug_rot.jpg")


def process_image(imgf: Path):
    """
    Pipeline for a single image:
    1. Load -> 2. Skull Strip (Simple) -> 3. Augment (Save new files) 
    -> 4. Resize (Overwrite original).
    """
    try:
        # 1. Load and Convert
        img = Image.open(imgf).convert("RGB")
        
        # 2. Skull Stripping
        img_stripped = skull_strip_simple(img)
        
        # 3. Augmentation 
        # We augment the stripped image to ensure consistency.
        base_path_without_ext = str(imgf.with_suffix(''))
        augment_image(img_stripped, base_path_without_ext)
        
        # 4. Final Resize and Overwrite (using the stripped version)
        final_img = img_stripped.resize(TARGET_SIZE, Image.BILINEAR)
        final_img.save(imgf) 
        
    except Exception as e:
        # Print error details but continue processing
        print(f"Error processing {imgf}: {e}")


def process_folder(folder: Path):
    """Processes all images within a split folder (train, val, test)."""
    for cls in folder.iterdir():
        if not cls.is_dir(): continue
        print(f"  Processing class: {cls.name}")
        # Convert glob to list to avoid issues when augmentation creates new files
        for imgf in list(cls.glob("*")): 
            if imgf.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                process_image(imgf)


if __name__ == "__main__":
    print(f"Starting Preprocessing Pipeline (Target Size: {TARGET_SIZE})")
    print("-" * 50)
    
    # NOTE ON REGISTRATION:
    print("Registration (alignment to a standard atlas) is omitted.")
    print("For full anatomical standardization, this step requires specialized tools.")
    print("It aligns different scans to minimize positional variation (e.g., T1 to MNI space).")
    print("-" * 50)
    
    for split in ["train", "val", "test"]:
        p = DATA_DIR / split
        if p.exists():
            print("Processing split:", p)
            process_folder(p)
        else:
            print("Skip split:", split, "- not found at", p)

    print("-" * 50)
    print("Preprocessing complete.")
    print("Original images have been skull-stripped and resized.")
    print("Augmented images (e.g., *_aug_flip.jpg, *_aug_rot.jpg) have been created.")