"""
Training Loop
-------------
Handles training, logging and checkpointing.
"""

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler


class Trainer:
    def __init__(self, model, dataloader, lr=3e-4):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scaler = GradScaler()

    def train_epoch(self, device):
        self.model.train()
        total_loss = 0

        for batch in self.dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    targets.view(-1),
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)
