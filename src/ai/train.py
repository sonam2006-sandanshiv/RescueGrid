import argparse
import logging
import torch
from torch.utils.data import DataLoader
from src.ai.model import create_model, get_loss_fn
from src.ai.dataset_kurosiwo import KuroSiwoDataset

logger = logging.basicConfig(level=logging.INFO)

def train(fast_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on device: {device}")
    
    model = create_model().to(device)
    criterion = get_loss_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    train_dataset = KuroSiwoDataset("data/kurosiwo", split="train")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    epochs = 1 if fast_mode else 10
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if fast_mode and batch_idx > 1:
                break
                
        logging.info(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss/len(train_loader):.4f}")
        
    # Save weights
    import os
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/unet_flood.pth")
    logging.info("Model saved to models/unet_flood.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run a quick debug training loop")
    args = parser.parse_args()
    train(args.fast)
