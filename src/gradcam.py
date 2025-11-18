# src/gradcam.py
"""
Robust Grad-CAM runner with multiple fallbacks and debug prints.
Run: python -m src.gradcam
"""

from pathlib import Path
from PIL import Image
import numpy as np
import inspect
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.train import build_model, DEVICE

MODEL_PATH = Path("models/best_model.pth")
DATA_DIR = Path("data/test")
OUT_DIR = Path("models/gradcam")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint():
    ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE)
    classes = ckpt.get("classes", None)
    if classes is None:
        raise RuntimeError("Checkpoint missing 'classes' key.")
    model = build_model(len(classes), feature_extract=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model, classes

def preprocess_pil(pil_img):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tf(pil_img).unsqueeze(0).to(DEVICE)

def pil_to_rgb_float(pil_img):
    arr = np.array(pil_img.resize((224,224))).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

def try_cam_variants(model, target_layer, input_tensor):
    """
    Try several Grad-CAM variants and calling conventions.
    Return grayscale_cam (H,W) numpy array or None.
    """
    candidates = []

    # try GradCAM, GradCAMPlusPlus, ScoreCAM
    try:
        candidates.append(GradCAM)
    except Exception:
        pass
    try:
        candidates.append(GradCAMPlusPlus)
    except Exception:
        pass
    try:
        candidates.append(ScoreCAM)
    except Exception:
        pass

    for CamClass in candidates:
        # instantiate in a few ways depending on signature
        sig = inspect.signature(CamClass)
        params = sig.parameters.keys()
        inst = None
        inst_info = None
        try:
            if "device" in params:
                inst = CamClass(model=model, target_layers=[target_layer], device=DEVICE)
                inst_info = "device"
            elif "use_cuda" in params:
                inst = CamClass(model=model, target_layers=[target_layer], use_cuda=(DEVICE.type=="cuda"))
                inst_info = "use_cuda"
            else:
                inst = CamClass(model=model, target_layers=[target_layer])
                inst_info = "no_flag"
        except Exception as e:
            # try simple constructor
            try:
                inst = CamClass(model=model, target_layers=[target_layer])
                inst_info = "constructor_fallback"
            except Exception as e2:
                # cannot construct this CamClass
                # print minimal debug
                print(f"[DEBUG] Could not construct {CamClass._name_}: {e2}")
                continue

        # try several calling forms
        call_variants = [
            {"kw": {"input_tensor": input_tensor}, "tag": "input_tensor_kw"},
            {"kw": {"input_tensor": input_tensor, "eigen_smooth": True}, "tag": "eigen_smooth"},
            {"kw": {"input_tensor": input_tensor, "aug_smooth": True}, "tag": "aug_smooth"},
            {"kw": {"input_tensor": input_tensor, "targets": None}, "tag": "targets_none"},
            {"kw": {"input_tensor": input_tensor, "targets": None, "eigen_smooth": True}, "tag": "targets_eigen"},
        ]
        for variant in call_variants:
            try:
                out = inst(**variant["kw"])
                # Some versions return numpy array, some list/tuple; normalize
                if out is None:
                    # debug print but continue trying
                    print(f"[DEBUG] {CamClass._name_} returned None for call {variant['tag']} (inst_mode={inst_info})")
                    continue
                # out could be numpy of shape (N,H,W) or list
                if isinstance(out, (list, tuple)):
                    out_arr = np.array(out)
                else:
                    out_arr = np.array(out)
                # First dimension often batch dim
                if out_arr.ndim == 3:
                    grayscale = out_arr[0]
                elif out_arr.ndim == 2:
                    grayscale = out_arr
                else:
                    # unexpected shape
                    print(f"[DEBUG] {CamClass._name_} gave unusual shape {out_arr.shape} with tag {variant['tag']}")
                    continue

                # sanity check values
                if np.isnan(grayscale).any() or np.max(grayscale) == 0:
                    print(f"[DEBUG] {CamClass._name_} produced empty/nan cam for {variant['tag']}")
                    continue

                return grayscale  # success
            except Exception as e:
                # catch and continue to next variant
                # keep debug short
                # print minimal debug only occasionally
                # suppressed verbose prints to avoid clutter
                # but show message for first few failures
                #print(f"[DEBUG] {CamClass._name_} call {variant['tag']} failed: {e}")
                continue

    # if we reach here, nothing worked
    return None

def main():
    if not MODEL_PATH.exists():
        print("Model not found:", MODEL_PATH)
        return
    if not DATA_DIR.exists():
        print("Test data folder not found:", DATA_DIR)
        return

    model, classes = load_checkpoint()
    target_layer = model.layer4[-1]

    for cls_dir in sorted(DATA_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = list(cls_dir.glob("*"))
        if not imgs:
            continue

        for imgp in imgs[:6]:
            try:
                pil = Image.open(imgp).convert("RGB")
            except Exception as e:
                print("Could not open", imgp, e)
                continue

            input_tensor = preprocess_pil(pil)  # already on DEVICE
            rgb_img = pil_to_rgb_float(pil)

            grayscale_cam = try_cam_variants(model, target_layer, input_tensor)

            if grayscale_cam is None:
                print("Grad-CAM failed for", imgp, "— all variants returned none or invalid.")
                continue

            try:
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                outp = OUT_DIR / f"{cls_dir.name}_{imgp.name}"
                Image.fromarray(cam_image).save(outp)
                print("Saved Grad-CAM to", outp)
            except Exception as e:
                print("Error saving overlay for", imgp, e)

if __name__ == "__main__":
    main()