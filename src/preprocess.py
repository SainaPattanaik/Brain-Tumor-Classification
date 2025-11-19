# import os
# from pathlib import Path
# from PIL import Image, ImageChops
# import numpy as np
# import random
# # NOTE: We can use PIL and NumPy to simulate basic morphological operations
# from scipy.ndimage import binary_fill_holes # Adding this common scientific library for robust mask cleaning

# DATA_DIR = Path("./data")  # run from src/
# TARGET_SIZE = (224, 224)

# # --- Advanced Preprocessing Functions ---

# def skull_strip_simple(img: Image.Image) -> Image.Image:
#     """
#     Conceptual skull stripping function using robust thresholding and mask cleaning.

#     In a real-world scenario, this would involve complex algorithms 
#     or dedicated libraries (e.g., SimpleITK, HD-BET) to accurately separate 
#     the brain tissue from the skull and surrounding areas.
    
#     This implementation uses median-based thresholding for robustness and fills 
#     small internal holes in the mask (a common artifact when isolating brain tissue).
#     """
#     # Convert to grayscale NumPy array for processing
#     img_array = np.array(img.convert("L"))
    
#     # Calculate a more robust threshold
#     img_array_flat = img_array.flatten()
    
#     # Filter out near-black background pixels (intensity < 10)
#     foreground_pixels = img_array_flat[img_array_flat > 10]
    
#     if foreground_pixels.size == 0:
#         # If no significant pixels are found, return the original image
#         return img 
        
#     # CRITICAL FIX: Use the median of the foreground pixels as a robust threshold base.
#     # The median is less sensitive to high-intensity noise or bright skull/fat.
#     # We use a factor (e.g., 0.6) to keep only the brightest core (the brain).
#     threshold = np.median(foreground_pixels) * 0.6 
    
#     # Set a minimum threshold to avoid aggressive stripping in very dark images
#     threshold = max(30, threshold) 
    
#     # 1. Create the binary mask
#     mask_np = img_array > threshold
    
#     # 2. Mask Cleaning (Fill Holes)
#     # Filling holes helps to include internal areas of the brain that might 
#     # have been falsely excluded by thresholding (like CSF spaces or low-intensity tumors).
#     try:
#         mask_np = binary_fill_holes(mask_np)
#     except NameError:
#         # Fallback if scipy.ndimage is not available (less robust)
#         pass 

#     # Find coordinates of the non-background area
#     coords = np.argwhere(mask_np)
#     if coords.size == 0:
#         return img # Return original if the mask is empty

#     y_min, x_min = coords.min(axis=0)
#     y_max, x_max = coords.max(axis=0)
    
#     # Create a full-size mask image from the boolean array
#     full_mask = Image.fromarray((mask_np * 255).astype(np.uint8))
    
#     # Apply mask to the original RGB image (composite layers)
#     # This sets everything outside the mask to black (0, 0, 0)
#     stripped_img = Image.composite(img, Image.new('RGB', img.size, (0, 0, 0)), full_mask)
    
#     # Crop to the bounding box of the content for normalization
#     return stripped_img.crop((x_min, y_min, x_max, y_max))


# def augment_image(img: Image.Image, base_path: str):
#     """Applies basic geometric augmentations and saves new image files."""
    
#     # 1. Horizontal Flip
#     flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
#     flipped_img.save(f"{base_path}_aug_flip.jpg")

#     # 2. Random Rotation (e.g., -10 to +10 degrees)
#     angle = random.uniform(-10, 10)
#     # Use Image.BILINEAR for resampling and filling empty areas (no expand)
#     rotated_img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0,0,0)) 
#     rotated_img.save(f"{base_path}_aug_rot.jpg")


# def process_image(imgf: Path):
#     """
#     Pipeline for a single image:
#     1. Load -> 2. Skull Strip (Simple) -> 3. Augment (Save new files) 
#     -> 4. Resize (Overwrite original).
#     """
#     try:
#         # 1. Load and Convert
#         img = Image.open(imgf).convert("RGB")
        
#         # 2. Skull Stripping
#         img_stripped = skull_strip_simple(img)
        
#         # 3. Augmentation 
#         # We augment the stripped image to ensure consistency.
#         base_path_without_ext = str(imgf.with_suffix(''))
#         augment_image(img_stripped, base_path_without_ext)
        
#         # 4. Final Resize and Overwrite (using the stripped version)
#         final_img = img_stripped.resize(TARGET_SIZE, Image.BILINEAR)
#         final_img.save(imgf) 
        
#     except Exception as e:
#         # Print error details but continue processing
#         print(f"Error processing {imgf}: {e}")


# def process_folder(folder: Path):
#     """Processes all images within a split folder (train, val, test)."""
#     for cls in folder.iterdir():
#         if not cls.is_dir(): continue
#         print(f"  Processing class: {cls.name}")
#         # Convert glob to list to avoid issues when augmentation creates new files
#         for imgf in list(cls.glob("*")): 
#             if imgf.suffix.lower() in ['.jpg', '.jpeg', '.png']:
#                 process_image(imgf)


# if __name__ == "__main__":
#     print(f"Starting Preprocessing Pipeline (Target Size: {TARGET_SIZE})")
#     print("-" * 50)
    
#     # NOTE ON REGISTRATION:
#     print("Registration (alignment to a standard atlas) is omitted.")
#     print("For full anatomical standardization, this step requires specialized tools.")
#     print("It aligns different scans to minimize positional variation (e.g., T1 to MNI space).")
#     print("-" * 50)
    
#     for split in ["train", "val", "test"]:
#         p = DATA_DIR / split
#         if p.exists():
#             print("Processing split:", p)
#             process_folder(p)
#         else:
#             print("Skip split:", split, "- not found at", p)

#     print("-" * 50)
#     print("Preprocessing complete.")
#     print("Original images have been skull-stripped and resized.")
#     print("Augmented images (e.g., *_aug_flip.jpg, *_aug_rot.jpg) have been created.")

#!/usr/bin/env python3
"""
preprocess.py

Preprocessing pipeline for 2D brain MRI slices with an optional Nilearn-based skull-stripper.

Usage examples:
    # use simple skull-stripper (default)
    python preprocess.py --data_dir ./data --target_size 224 --skull_strip --augment --norm_method zscore

    # use Nilearn skull-stripper (preferred if nilearn+nibabel installed)
    python preprocess.py --data_dir ./data --target_size 224 --nilearn_strip --augment --norm_method zscore
"""

import os
import argparse
import random
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageEnhance
import numpy as np

# optional nilearn / nibabel imports will be attempted lazily
try:
    import nibabel as nib
    from nilearn.masking import compute_brain_mask
    _HAS_NILEARN = True
except Exception:
    _HAS_NILEARN = False

from scipy.ndimage import binary_fill_holes

# reproducibility
RND = random.Random(42)
np.random.seed(42)


# ---------- Utility & Preprocessing Functions ----------

def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")  # work in RGB space


def array_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """Convert single-channel array [H,W] (0-255 uint8) to 3-channel PIL Image."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def duplicate_channels(img: Image.Image) -> Image.Image:
    """If image is grayscale, duplicate channel to make 3-channel RGB"""
    if img.mode == "L":
        return Image.merge("RGB", (img, img, img))
    return img


# --- Simple median-threshold skull-stripping (fallback) ---
def skull_strip_simple(img: Image.Image) -> Image.Image:
    """
    Robust median-based threshold skull-stripper (conceptual fallback).
    Returns an RGB PIL image with background forced to black and cropped to content bbox.
    """
    img_gray = img.convert("L")
    img_array = np.array(img_gray)
    foreground_pixels = img_array[img_array > 10]
    if foreground_pixels.size == 0:
        return img

    threshold = np.median(foreground_pixels) * 0.6
    threshold = max(30, threshold)
    mask_np = img_array > threshold

    # fill holes for cleaner brain mask
    try:
        mask_np = binary_fill_holes(mask_np)
    except Exception:
        pass

    coords = np.argwhere(mask_np)
    if coords.size == 0:
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    full_mask = Image.fromarray((mask_np * 255).astype(np.uint8))
    stripped_img = Image.composite(img, Image.new('RGB', img.size, (0, 0, 0)), full_mask)
    return stripped_img.crop((x_min, y_min, x_max + 1, y_max + 1))


# --- Nilearn-based skull-stripper for 2D slices ---
def skull_strip_nilearn(img: Image.Image) -> Image.Image:
    """
    Use Nilearn (compute_brain_mask) to create a brain mask for a 2D slice by:
      1) converting 2D slice to a single-slice NIfTI (3D array with shape (H, W, 1)),
      2) running nilearn.masking.compute_brain_mask,
      3) applying the returned 3D mask to the 2D slice and cropping to bbox.

    Falls back to skull_strip_simple on failure.
    """
    if not _HAS_NILEARN:
        # Nilearn not available -> fallback
        return skull_strip_simple(img)

    try:
        # Convert to grayscale np array
        gray = np.array(img.convert("L"), dtype=np.float32)

        # Create a single-slice 3D volume (H, W, 1) -> transpose to (X,Y,Z) nibabel expects (Z last is fine)
        vol3 = gray[..., np.newaxis]  # shape (H, W, 1)

        # Create a simple identity affine (voxel size = 1)
        affine = np.eye(4)

        # Build nibabel NIfTI image
        nifti = nib.Nifti1Image(vol3, affine)

        # compute brain mask using nilearn (this expects a 3D image)
        mask_3d = compute_brain_mask(nifti)  # returns a Nifti1Image or boolean ndarray

        # ensure mask array
        if hasattr(mask_3d, "get_fdata"):
            mask_arr = np.asarray(mask_3d.get_fdata()).astype(bool)
        else:
            mask_arr = np.asarray(mask_3d).astype(bool)

        # mask_arr shape should be (H, W, 1) -> squeeze last dim
        if mask_arr.ndim == 3 and mask_arr.shape[2] == 1:
            mask2d = mask_arr[:, :, 0]
        else:
            # in unexpected shape, fallback
            mask2d = mask_arr.reshape(gray.shape[:2])

        # hole filling for safety
        try:
            mask2d = binary_fill_holes(mask2d)
        except Exception:
            pass

        # If mask is empty, fallback
        if mask2d.sum() == 0:
            return skull_strip_simple(img)

        # Apply mask to original RGB image: set outside to black
        mask_pil = Image.fromarray((mask2d * 255).astype(np.uint8))
        stripped_img = Image.composite(img, Image.new('RGB', img.size, (0, 0, 0)), mask_pil)

        # Crop bounding box to minimize background
        coords = np.argwhere(mask2d)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return stripped_img.crop((x_min, y_min, x_max + 1, y_max + 1))

    except Exception as e:
        # Any failure -> print debug and fallback to simple method
        print(f"[nilearn_strip] Warning: Nilearn skull-strip failed: {e}. Falling back to simple stripper.")
        return skull_strip_simple(img)


# ---------- Intensity Normalization ----------

def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-6:
        return np.clip(a, 0, 255).astype(np.uint8)
    norm = (a - lo) / (hi - lo)
    return (norm * 255.0).round().astype(np.uint8)

def normalize_zscore(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    mu, sigma = a.mean(), a.std()
    if sigma < 1e-6:
        return normalize_minmax(arr)
    z = (a - mu) / sigma
    z_clipped = np.clip(z, -3.0, 3.0)
    scaled = ( (z_clipped + 3.0) / 6.0 ) * 255.0
    return np.clip(scaled, 0, 255).round().astype(np.uint8)

def normalize_image(img: Image.Image, method: str) -> Image.Image:
    gray = np.array(img.convert("L"))
    if method == "minmax":
        out = normalize_minmax(gray)
    elif method == "zscore":
        out = normalize_zscore(gray)
    else:
        raise ValueError("Unknown normalization method: " + method)
    return array_to_pil_rgb(out)


# ---------- Augmentations ----------

def random_horizontal_flip(img: Image.Image, p=0.5) -> Image.Image:
    if RND.random() < p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def random_rotation(img: Image.Image, max_deg=15) -> Image.Image:
    angle = RND.uniform(-max_deg, max_deg)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0,0,0))

def random_brightness_contrast(img: Image.Image, b_range=(0.9, 1.1), c_range=(0.9, 1.1)) -> Image.Image:
    b = RND.uniform(*b_range)
    c = RND.uniform(*c_range)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    return img

def random_translation(img: Image.Image, max_shift_frac=0.05) -> Image.Image:
    w, h = img.size
    max_dx = int(w * max_shift_frac)
    max_dy = int(h * max_shift_frac)
    dx = RND.randint(-max_dx, max_dx)
    dy = RND.randint(-max_dy, max_dy)
    canvas = Image.new("RGB", (w, h), (0,0,0))
    left = max(0, dx)
    top = max(0, dy)
    src_left = max(0, -dx)
    src_top = max(0, -dy)
    region = img.crop((src_left, src_top, src_left + w - abs(dx), src_top + h - abs(dy)))
    canvas.paste(region, (left, top))
    return canvas

def apply_augmentations(img: Image.Image, aug_list=None) -> Image.Image:
    if aug_list is None:
        aug_list = ["flip", "rot", "bright", "trans"]
    out = img
    if "flip" in aug_list:
        out = random_horizontal_flip(out, p=0.5)
    if "rot" in aug_list:
        out = random_rotation(out, max_deg=15)
    if "bright" in aug_list:
        out = random_brightness_contrast(out, b_range=(0.9, 1.1), c_range=(0.9, 1.1))
    if "trans" in aug_list:
        out = random_translation(out, max_shift_frac=0.05)
    return out


# ---------- High-level Pipeline ----------

def process_image(
    img_path: Path,
    target_size: Tuple[int,int]=(224,224),
    skull_method: str="simple",  # 'simple' or 'nilearn' or 'none'
    do_augment: bool=True,
    norm_method: str="zscore",
    keep_three_channel: bool=True,
):
    """Process a single image path in-place (overwrites original) and optionally creates augmentations."""
    try:
        img = load_image(img_path)  # RGB

        # Skull strip based on requested method
        if skull_method == "nilearn":
            img = skull_strip_nilearn(img)
        elif skull_method == "simple":
            img = skull_strip_simple(img)
        elif skull_method == "none":
            pass
        else:
            raise ValueError("Unknown skull_method: " + skull_method)

        # Resize BEFORE normalization
        img = img.resize(target_size, Image.BILINEAR)

        # Normalize
        img = normalize_image(img, norm_method)

        # Ensure 3-channels if asked
        if keep_three_channel:
            img = duplicate_channels(img.convert("RGB"))

        # Save processed main image (overwrite original)
        img.save(img_path)

        # Augmentations: create a few augmented variants and save
        if do_augment:
            base = img_path.with_suffix("")  # path without extension
            aug_variants = [
                ("_aug_flip", ["flip"]),
                ("_aug_rot", ["rot"]),
                ("_aug_bright", ["bright"]),
                ("_aug_trans", ["trans"]),
                ("_aug_all", ["flip","rot","bright","trans"]),
            ]
            for suf, ops in aug_variants:
                aug_img = apply_augmentations(img, ops)
                out_path = Path(f"{str(base)}{suf}.jpg")
                aug_img.save(out_path)

    except Exception as e:
        print(f"[ERROR] processing {img_path}: {e}")


def process_folder(folder: Path, **kwargs):
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return
    for cls in folder.iterdir():
        if not cls.is_dir():
            continue
        print(f"Processing class: {cls.name} ({cls})")
        images = [p for p in cls.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        print(f"  Found {len(images)} images")
        for imgf in images:
            process_image(imgf, **kwargs)


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess MRI images (2D) — skull-strip (simple/nilearn), normalize, resize, augment")
    p.add_argument("--data_dir", type=str, default="./data", help="Root data directory (should contain train/val/test folders)")
    p.add_argument("--target_size", type=int, default=224, help="Resize to this square size (default 224).")
    p.add_argument("--skull_strip", action="store_true", help="Enable simple skull-stripping (legacy).")
    p.add_argument("--nilearn_strip", action="store_true", help="Enable Nilearn-based skull-stripping (preferred if available).")
    p.add_argument("--augment", action="store_true", help="Create augmented images alongside originals.")
    p.add_argument("--norm_method", type=str, choices=["minmax", "zscore"], default="zscore", help="Intensity normalization method per image.")
    p.add_argument("--no_dup_channels", action="store_true", help="Do NOT duplicate grayscale to 3 channels (keep RGB output from pipeline).")
    p.add_argument("--splits", type=str, nargs="+", default=["train","val","test"], help="Which splits to process (folder names inside data_dir).")
    return p.parse_args()

def main():
    args = parse_args()
    data_root = Path(args.data_dir)
    target = (args.target_size, args.target_size)

    # Decide skull method priority
    if args.nilearn_strip:
        skull_method = "nilearn"
        if not _HAS_NILEARN:
            print("[WARN] nilearn/nibabel not available in your environment. Falling back to simple skull-stripper.")
            skull_method = "simple"
    elif args.skull_strip:
        skull_method = "simple"
    else:
        skull_method = "none"

    opts = dict(
        target_size=target,
        skull_method=skull_method,
        do_augment=args.augment,
        norm_method=args.norm_method,
        keep_three_channel=not args.no_dup_channels
    )

    print("Preprocessing pipeline (2D) — settings:")
    print(f"  data_root: {data_root}")
    print(f"  target_size: {target}")
    print(f"  skull_method: {skull_method}")
    print(f"  augment: {opts['do_augment']}")
    print(f"  normalization: {opts['norm_method']}")
    print(f"  duplicate->3ch: {opts['keep_three_channel']}")
    print("-" * 60)

    for s in args.splits:
        split_dir = data_root / s
        if split_dir.exists():
            print(f"Processing split folder: {split_dir}")
            process_folder(split_dir, **opts)
        else:
            print(f"Skip (not found): {split_dir}")

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
