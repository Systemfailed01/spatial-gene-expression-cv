import torch
import torch.nn as nn
from torch.optim import Adam


def train_encoder(encoder, head, train_loader, val_loader,
                  encoder_name, save_path, device, epochs=30):
    """Train MLP head on frozen encoder features.
    
    Encoder weights are never updated. Only the MLP head learns.
    Best checkpoint by validation loss is saved to save_path.
    """
    from src.model import get_features

    optimizer = Adam(head.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.MSELoss()
    best_val  = float('inf')

    print(f'training {encoder_name}')
    print(f'{"epoch":>6}  {"train":>10}  {"val":>10}')
    print('-' * 30)

    for epoch in range(epochs):
        head.train()
        train_loss = 0.0
        for patches, expr in train_loader:
            patches, expr = patches.to(device), expr.to(device)
            with torch.no_grad():
                features = get_features(encoder, patches, encoder_name)
            features = features.detach()
            loss = criterion(head(features), expr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for patches, expr in val_loader:
                patches, expr = patches.to(device), expr.to(device)
                features = get_features(encoder, patches, encoder_name)
                val_loss += criterion(head(features), expr).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(head.state_dict(), save_path)

        if (epoch + 1) % 5 == 0:
            print(f'{epoch+1:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}')

    print(f'done  best_val={best_val:.4f}')
