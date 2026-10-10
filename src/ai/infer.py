import torch
import numpy as np
from src.ai.model import create_model

def run_inference(pre_vv, post_vv, pre_vh, post_vh, model_path="models/unet_flood.pth"):
    """
    Runs the trained U-Net on a new S1 pair.
    Returns predicted classes and confidence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        import logging
        logging.warning(f"Model not found at {model_path}. Returning zeros.")
        return np.zeros_like(pre_vv, dtype=np.uint8), np.zeros_like(pre_vv, dtype=np.float32)
        
    model.eval()
    
    # Preprocess inputs (normalize, etc.)
    # Dummy normalization for prototype
    input_tensor = np.stack([pre_vv, post_vv, pre_vh, post_vh], axis=0)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, dim=1)
        
    return preds.squeeze().cpu().numpy().astype(np.uint8), confidence.squeeze().cpu().numpy()
