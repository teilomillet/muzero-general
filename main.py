import wandb
import importlib
from muzero import MuZero

def main():
    # Initialize wandb with more detailed configuration
    wandb.init(
        project="muzero-boop",
        name="muzero-boop-optimized",
        config={
            "game": "boop",
            "num_simulations": 100,
            "num_workers": 4,
            "training_steps": 10000,
            "batch_size": 256,
            "optimizer": "SGD",
            "network": "resnet",
            "blocks": 6,
            "channels": 128
        }
    )
    
    # Create MuZero instance for Boop game
    muzero = MuZero("boop")
    
    # Start training
    muzero.train()
    
    # Close wandb run when done
    wandb.finish()


if __name__ == "__main__":
    main()
