import wandb
import importlib
from muzero import MuZero

def main():
    # Initialize wandb
    wandb.init(project="muzero-boop", name="muzero-boop-experiment")
    
    # Create MuZero instance for Boop game
    muzero = MuZero("boop")
    
    # Start training
    muzero.train()


if __name__ == "__main__":
    main()
