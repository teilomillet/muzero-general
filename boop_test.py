import sys
import numpy as np
from games.boop import Game, Boop

def test_boop_game():
    """
    Simple test to ensure the Boop game implementation works correctly.
    """
    print("Testing Boop Game implementation...")
    
    # Create a new game
    game = Game(seed=42)
    
    # Reset the game and get initial observation
    obs = game.reset()
    print("Initial observation shape:", obs.shape)
    
    # Test legal actions
    legal_actions = game.legal_actions()
    print(f"Number of legal actions at start: {len(legal_actions)}")
    
    # Test making a few moves
    print("\nMaking some test moves:")
    actions = [13, 8, 20, 7]  # Some sample actions to test
    
    for i, action in enumerate(actions):
        if action in game.legal_actions():
            print(f"\nMove {i+1}: Player {game.to_play()+1} plays {game.action_to_string(action)}")
            obs, reward, done = game.step(action)
            print(f"Reward: {reward}, Game finished: {done}")
            game.env.render()
        else:
            print(f"Action {action} is not legal")
    
    # Test boop mechanics
    print("\nTesting boop mechanics with adjacent pieces:")
    # Place a piece that will boop another piece
    if 14 in game.legal_actions():
        print(f"\nPlayer {game.to_play()+1} plays {game.action_to_string(14)}")
        obs, reward, done = game.step(14)
        print(f"Reward: {reward}, Game finished: {done}")
        game.env.render()
    
    print("\nBoop Game test completed!\n")

if __name__ == "__main__":
    test_boop_game() 