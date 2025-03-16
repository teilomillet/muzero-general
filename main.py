import wandb
import importlib
from muzero import MuZero

def main():
    # Initialize wandb with more detailed configuration
    wandb.init(
        project="muzero-boop",
        name="muzero-boop-test-run",
        config={
            "game": "boop",
            "num_simulations": 50,
            "num_workers": 4,
            "training_steps": 10000,
            "batch_size": 128,
            "optimizer": "SGD",
            "network": "resnet",
            "blocks": 4,
            "channels": 64,
            "replay_buffer_size": 1000,
            "test_run": True
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
