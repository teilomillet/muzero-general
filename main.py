import wandb
import importlib
from muzero import MuZero

def main(use_wandb=True):
    # Initialize wandb with more detailed configuration
    if use_wandb:
        try:
            wandb.init(
                project="muzero-boop",
                name="muzero-boop-simple-test",
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
                    "test_run": False,
                    "threefold_repetition_rule": True,
                    "max_moves": 50,
                    "rules_version": "simplified",
                    "rules_notes": "Using simplified version without Cats/Kittens distinction and line-of-two protection",
                    "muzero_player": None,
                    "opponent": "random",
                    "num_tests": 10,
                    "log_interval": 10,
                }
            )
            print("Wandb initialized successfully")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            use_wandb = False
    
    # Create MuZero instance for Boop game
    muzero = MuZero("boop")
    
    # Start training
    muzero.train()
    
    # Evaluate the final model with explicit test games
    print("\nEvaluating final model...")
    # Test as first player
    test_results_first = muzero.test(render=False, muzero_player=0, num_tests=25)
    print(f"\nFinal model evaluation as first player (average reward over 25 games): {test_results_first:.4f}")
    
    # Test as second player
    test_results_second = muzero.test(render=False, muzero_player=1, num_tests=25)
    print(f"\nFinal model evaluation as second player (average reward over 25 games): {test_results_second:.4f}")
    
    # Log final evaluation results to wandb
    if wandb.run is not None and use_wandb:
        wandb.log({
            "final_evaluation_reward_as_first": test_results_first,
            "final_evaluation_reward_as_second": test_results_second,
            "final_evaluation_reward_average": (test_results_first + test_results_second) / 2
        })
    
    # Close wandb run when done
    if wandb.run is not None and use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import sys
    # Allow disabling wandb from command line
    use_wandb = True
    if len(sys.argv) > 1 and sys.argv[1] == "--no-wandb":
        use_wandb = False
    main(use_wandb=use_wandb)
