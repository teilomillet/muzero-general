#!/usr/bin/env python3

import os
import sys
import numpy as np
import pathlib
import torch

# Add safe globals setup for numpy arrays
from torch.serialization import add_safe_globals
import numpy

# Make numpy arrays safe to load
add_safe_globals([numpy._core.multiarray.scalar, numpy.dtype, numpy.dtypes.Float64DType])

from muzero import MuZero
from games.boop import Game, Boop
from self_play import MCTS
# Import the Model class directly
from models import MuZeroNetwork

# Monkey patch Boop to track board history
original_step = Boop.step

def patched_step(self, action):
    """
    Track board observation history for repetition detection.
    This function replaces the original step method.
    """
    # Make sure the action is valid
    if action not in self.legal_actions():
        print(f"WARNING: Illegal action {action} attempted!")
        return self.get_observation(), -1, False
        
    # Initialize board_history if it doesn't exist
    if not hasattr(self, 'board_history'):
        self.board_history = []
    
    # Save the current observation state before making a move
    current_obs = self.get_observation().copy()
    self.board_history.append(current_obs)
    
    # Call the original step method
    return original_step(self, action)

# Apply the monkey patch 
Boop.step = patched_step

# Also patch the Game class to ensure it initializes properly
original_game_init = Game.__init__

def patched_game_init(self, seed=None):
    """Ensure the game initializes with a completely clean board."""
    # Call original init
    original_game_init(self, seed)
    
    # Force-reset the environment to ensure clean state
    self.env = Boop()  # Create a fresh environment
    self.reset()  # Reset it
    
Game.__init__ = patched_game_init

class BoopTextGame:
    """
    A text-based version of Boop to play against a trained MuZero model.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the game and load the MuZero model.
        
        Args:
            model_path: Path to the model checkpoint
        """
        # Initialize the game
        self.game = Game()
        self.observation = self.game.reset()
        self.done = False
        self.winner = None
        
        # Ensure the game board is properly reset
        self.game.env.board = np.zeros((2, self.game.env.board_size, self.game.env.board_size), dtype=np.int8)
        self.game.env.player = 0
        self.game.env.done = False
        self.game.env.move_count = 0
        self.game.env.board_history = []
        
        # Load MuZero model
        self.model_loaded = False
        if model_path:
            print(f"Loading model from {model_path}")
            self.muzero = MuZero("boop")
            
            # Custom loading to handle newer PyTorch versions
            try:
                # Try direct loading with trusted source
                print("Attempting to load model with trusted_data=True...")
                checkpoint_path = pathlib.Path(model_path)
                checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
                self.muzero.checkpoint = checkpoint
                
                # Initialize the model
                self.muzero.model = MuZeroNetwork(self.muzero.config)
                self.muzero.model.set_weights(checkpoint["weights"])
                self.muzero.model.to(torch.device("cpu"))
                self.muzero.model.eval()
                
                self.model_loaded = True
                print("Successfully loaded model")
                
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Model loading failed. Exiting program.")
                sys.exit(1)
        else:
            print("No model path provided, MuZero will play randomly")
            
        # State variables
        self.human_player = 0  # Human starts as Player 0 (X)
        self.last_move = None
        self.move_history = []
    
    def display_board(self):
        """Display the game board in the terminal."""
        print("\n   0 1 2 3 4 5")
        print("  ---------------------")
        for r in range(6):
            print(f"{r} |", end=" ")
            for c in range(6):
                if self.game.env.board[0, r, c] == 1:
                    print("X", end=" ")
                elif self.game.env.board[1, r, c] == 1:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print("|")
        print("  ---------------------")
        
        # Mark last move with *
        if self.last_move:
            row, col = self.last_move
            print(f"Last move: ({row},{col})")
    
    def make_human_move(self):
        """Get and process the human player's move."""
        while True:
            try:
                print("\nYour turn (Player X)")
                row = int(input("Enter row (0-5): "))
                col = int(input("Enter column (0-5): "))
                
                if not (0 <= row < 6 and 0 <= col < 6):
                    print("Invalid coordinates! Must be between 0-5.")
                    continue
                
                action = row * 6 + col
                
                # Debug info to help understand the issue
                legal_actions = self.game.legal_actions()
                if action not in legal_actions:
                    print(f"Illegal move! Position ({row},{col}) is already occupied or invalid.")
                    print(f"Board state at ({row},{col}): Player 1: {self.game.env.board[0, row, col]}, Player 2: {self.game.env.board[1, row, col]}")
                    print(f"Legal actions: {legal_actions}")
                    continue
                
                # Make the move
                self.observation, reward, self.done = self.game.step(action)
                
                # Record the move
                self.last_move = (row, col)
                self.move_history.append((row, col, self.human_player))
                
                # Check for win
                if self.done:
                    if reward > 0:
                        self.winner = self.human_player
                        print("\nYou win! Game over.")
                    else:
                        print("\nGame ended in a draw.")
                
                return
                
            except ValueError:
                print("Please enter numbers between 0-5.")
    
    def get_stacked_observations(self):
        """Get properly stacked observations for MCTS."""
        # Get current observation
        current_obs = self.game.env.get_observation()
        
        # The shape should be (channels, height, width)
        obs_shape = current_obs.shape
        stacked_shape = (obs_shape[0] * (self.muzero.config.stacked_observations + 1), 
                         obs_shape[1], obs_shape[2])
        
        # Create stacked observations filled with zeros
        stacked_obs = np.zeros(stacked_shape, dtype=np.float32)
        
        # Copy current observation to the first part
        stacked_obs[:obs_shape[0]] = current_obs
        
        # Add history if available
        if hasattr(self.game.env, 'board_history') and self.game.env.board_history:
            history = self.game.env.board_history
            for i in range(min(self.muzero.config.stacked_observations, len(history))):
                # Get observation from history, most recent first
                past_obs = history[-(i+1)]
                # Place it in the stacked observation
                start_ch = obs_shape[0] * (i+1)
                end_ch = start_ch + obs_shape[0]
                stacked_obs[start_ch:end_ch] = past_obs
                
        return stacked_obs
    
    def make_ai_move(self):
        """Make a move for the AI."""
        print("\nMuZero is thinking...")
        
        if self.model_loaded:
            # Get legal actions
            legal_actions = self.game.legal_actions()
            
            # If no legal actions, game is a draw
            if not legal_actions:
                self.done = True
                print("No legal moves. Game ended in a draw.")
                return
            
            # Get current player (1 - human_player for AI)
            to_play = 1 - self.human_player  # This is the AI's player number
            
            try:
                # Get stacked observations
                stacked_obs = self.get_stacked_observations()
                
                # Run MCTS on CPU
                self.muzero.model.to(torch.device("cpu"))  # Ensure model is on CPU
                mcts = MCTS(self.muzero.config)
                
                # Run MCTS
                root, mcts_info = mcts.run(
                    self.muzero.model,
                    stacked_obs,
                    legal_actions,
                    to_play,
                    True
                )
                
                # Select best action based on visit counts
                visit_counts = [(a, child.visit_count) for a, child in root.children.items()]
                action = max(visit_counts, key=lambda x: x[1])[0]
                print(f"MCTS search depth: {mcts_info['max_tree_depth']}")
                
            except Exception as e:
                print(f"Error in MCTS: {e}")
                # Fall back to random action
                print("Falling back to random action selection")
                action = np.random.choice(legal_actions)
        else:
            # Random play if no model is loaded
            legal_actions = self.game.legal_actions()
            if legal_actions:
                action = np.random.choice(legal_actions)
            else:
                self.done = True
                print("No legal moves. Game ended in a draw.")
                return
        
        # Convert action to coordinates
        row = action // 6
        col = action % 6
        print(f"MuZero selects position ({row},{col})")
        
        # Make the move
        self.observation, reward, self.done = self.game.step(action)
        
        # Record the move
        self.last_move = (row, col)
        self.move_history.append((row, col, 1 - self.human_player))
        
        # Check for win
        if self.done:
            if reward > 0:
                self.winner = 1 - self.human_player
                print("\nMuZero wins! Game over.")
            else:
                print("\nGame ended in a draw.")
    
    def play(self):
        """Main game loop."""
        print("\nWelcome to Boop vs MuZero!")
        print("You are Player X, MuZero is Player O")
        print("The goal is to get three of your pieces in a row")
        print("When you place a piece next to another piece, it 'boops' (pushes) it away")
        
        # Completely recreate and reset the game to ensure clean state
        self.game = Game()
        self.observation = self.game.reset()
        self.done = False
        self.winner = None
        self.last_move = None
        self.move_history = []
        
        # Verify board is empty
        board_is_clean = True
        for r in range(6):
            for c in range(6):
                if self.game.env.board[0, r, c] != 0 or self.game.env.board[1, r, c] != 0:
                    print(f"WARNING: Board position ({r},{c}) is not empty at game start!")
                    self.game.env.board[0, r, c] = 0
                    self.game.env.board[1, r, c] = 0
                    board_is_clean = False
        
        if not board_is_clean:
            print("Fixed board state before starting game.")
            
        # Verify all actions are legal at the start
        legal_actions = self.game.legal_actions()
        if len(legal_actions) != 36:
            print(f"WARNING: Only {len(legal_actions)} legal actions at start instead of 36!")
            print(f"Missing actions: {set(range(36)) - set(legal_actions)}")
            
            # Force all positions to be legal
            self.game.env.board = np.zeros((2, 6, 6), dtype=np.int8)
        
        while not self.done:
            self.display_board()
            self.make_human_move()
            
            if not self.done:
                self.display_board()
                self.make_ai_move()
        
        # Show final board
        self.display_board()
        print("\nGame over!")


def main():
    """Run the game."""
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Try to find most recent model by default
        boop_results = pathlib.Path("results/boop")
        if boop_results.exists():
            model_dirs = sorted(boop_results.glob("*/"), reverse=True)
            if model_dirs:
                model_path = model_dirs[0] / "model.checkpoint"
                if not model_path.exists():
                    model_path = None
            else:
                model_path = None
        else:
            model_path = None
    
    # If a specific model was mentioned in the query, use that one
    if not model_path or not pathlib.Path(model_path).exists():
        specific_model = pathlib.Path("/workspace/muzero-general/results/boop/2025-03-16--17-39-27/model.checkpoint")
        if specific_model.exists():
            model_path = specific_model
    
    if model_path and pathlib.Path(model_path).exists():
        print(f"Using model: {model_path}")
    else:
        print("No model found, AI will play randomly")
        model_path = None
    
    # Start a new game
    game = BoopTextGame(model_path)
    
    # Before playing, verify that the board is actually empty
    # Double-check all positions
    empty_board = True
    for r in range(6):
        for c in range(6):
            if game.game.env.board[0, r, c] != 0 or game.game.env.board[1, r, c] != 0:
                print(f"WARNING: Position ({r},{c}) is not empty at start!")
                print(f"Player 1: {game.game.env.board[0, r, c]}, Player 2: {game.game.env.board[1, r, c]}")
                # Force clear this position
                game.game.env.board[0, r, c] = 0
                game.game.env.board[1, r, c] = 0
                empty_board = False
    
    if not empty_board:
        print("Fixed non-empty positions on the board.")
    
    # Start the game
    game.play()


if __name__ == "__main__":
    main() 