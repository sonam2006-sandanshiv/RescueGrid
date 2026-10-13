import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
from src.ai.model import create_model, get_loss_fn
from src.ai.dataset_kurosiwo import KuroSiwoDataset
from src.ai.dataset_sen1floods11 import Sen1Floods11Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(dataset_name="kurosiwo", fast_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device} using dataset: {dataset_name}")
    
    model = create_model().to(device)
    criterion = get_loss_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    if dataset_name == "sen1floods11":
        train_dataset = Sen1Floods11Dataset("data/sen1floods11", split="train")
    else:
        train_dataset = KuroSiwoDataset("data/kurosiwo", split="train")
        
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    epochs = 1 if fast_mode else 5
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        batches = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
            if fast_mode and batch_idx >= 1:
                break
                
        avg_loss = epoch_loss / max(1, batches)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    os.makedirs("models", exist_ok=True)
    out_model_path = f"models/unet_flood_{dataset_name}.pth"
    torch.save(model.state_dict(), out_model_path)
    # Also save as default unet_flood.pth
    torch.save(model.state_dict(), "models/unet_flood.pth")
    logger.info(f"Trained model saved to {out_model_path} and models/unet_flood.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flood Segmentation U-Net")
    parser.add_argument("--dataset", choices=["kurosiwo", "sen1floods11"], default="kurosiwo",
                        help="Choose training benchmark dataset (kurosiwo or sen1floods11)")
    parser.add_argument("--fast", action="store_true", help="Run a quick debug training loop")
    args = parser.parse_args()
    train(dataset_name=args.dataset, fast_mode=args.fast)
