# please log in to your W&B account first
import wandb
import random

def run_training_run(epochs, lr):
    print(f"Training for {epochs} epochs with learning rate {lr}")

    wandb.init(
        # Set the project where this run will be logged
        project="example", 
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": epochs,
        }
    )
    
    offset = random.random() / 5
    print(f"lr: {lr}")
    for epoch in range(2, epochs):
        # simulating a training run
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, acc={acc}, loss={loss}")
        wandb.log({"acc": acc, "loss": loss})

    wandb.finish()

def run_multiple_training_runs(epochs, lrs):
    for epoch in epochs:
        for lr in lrs:
            run_training_run(epoch, lr)

# Try different values for the learning rate
epochs = [100, 120, 140]
lrs = [0.1, 0.01, 0.001, 0.0001]
run_multiple_training_runs(epochs, lrs)