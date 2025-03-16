#!/usr/bin/env python3

import os
import sys
import pygame
import numpy as np
import pathlib
import torch

# Add safe globals setup for numpy arrays
from torch.serialization import add_safe_globals
import numpy

# Make numpy arrays safe to load
add_safe_globals([numpy._core.multiarray.scalar])

from muzero import MuZero
from games.boop import Game, Boop

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 700
BOARD_SIZE = 6
SQUARE_SIZE = 80
PIECE_RADIUS = 30
MARGIN = 50

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_RED = (255, 150, 150)
LIGHT_BLUE = (150, 150, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

class Boop6x6GUI:
    """
    A GUI for playing Boop against a trained MuZero model.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the GUI and load the MuZero model.
        
        Args:
            model_path: Path to the model checkpoint
        """
        # Create the screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Boop vs MuZero")
        
        # Initialize the game
        self.game = Game()
        self.observation = self.game.reset()
        self.done = False
        self.winner = None
        
        # Font for text
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 18)
        
        # Load MuZero model
        if model_path:
            print(f"Loading model from {model_path}")
            self.muzero = MuZero("boop")
            
            # Custom loading to handle newer PyTorch versions
            try:
                # First try the modified loading approach
                checkpoint_path = pathlib.Path(model_path)
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                self.muzero.checkpoint = checkpoint
                print("Successfully loaded model checkpoint")
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Trying alternative loading method...")
                try:
                    # Fallback to the original method
                    self.muzero.load_model(checkpoint_path=model_path)
                    self.model_loaded = True
                except Exception as e2:
                    print(f"Failed to load model: {e2}")
                    self.model_loaded = False
        else:
            print("No model path provided, MuZero will play randomly")
            self.model_loaded = False
            
        # Player colors
        self.player_colors = [RED, BLUE]
        self.player_light_colors = [LIGHT_RED, LIGHT_BLUE]
        
        # State variables
        self.human_player = 0  # Human starts as Player 0 (RED)
        self.highlighted_square = None
        self.last_move = None
        self.waiting_for_ai = False
        self.animation_in_progress = False
        self.moved_pieces = []  # For animation
        self.removed_pieces = []  # For animation
        self.move_history = []
        
        # Info text
        self.info_text = "You are RED. Click on an empty square to place a piece."
    
    def draw_board(self):
        """Draw the 6x6 game board."""
        # Fill background
        self.screen.fill(WHITE)
        
        # Draw title
        title = self.font.render("Boop vs MuZero", True, BLACK)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 10))
        
        # Draw info text
        info = self.font.render(self.info_text, True, BLACK)
        self.screen.blit(info, (SCREEN_WIDTH // 2 - info.get_width() // 2, SCREEN_HEIGHT - 40))
        
        # Draw the grid
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x = col * SQUARE_SIZE + MARGIN
                y = row * SQUARE_SIZE + MARGIN + 40
                
                # Draw the square
                rect = pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)
                color = GRAY if (row + col) % 2 == 0 else WHITE
                
                # Highlight square under mouse
                if self.highlighted_square == (row, col):
                    color = YELLOW
                
                # Highlight last move
                if self.last_move == (row, col):
                    color = GREEN
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)  # Border
                
                # Draw coordinate text in corner of square
                coord_text = self.small_font.render(f"{row},{col}", True, BLACK)
                self.screen.blit(coord_text, (x + 5, y + 5))
                
                # Draw pieces
                if self.game.env.board[0, row, col] == 1:
                    pygame.draw.circle(self.screen, RED, 
                                      (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2), 
                                      PIECE_RADIUS)
                elif self.game.env.board[1, row, col] == 1:
                    pygame.draw.circle(self.screen, BLUE, 
                                      (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2), 
                                      PIECE_RADIUS)
    
    def get_square_at_pos(self, pos):
        """Convert screen position to board coordinates."""
        x, y = pos
        if x < MARGIN or x >= MARGIN + BOARD_SIZE * SQUARE_SIZE:
            return None
        if y < MARGIN + 40 or y >= MARGIN + 40 + BOARD_SIZE * SQUARE_SIZE:
            return None
        
        col = (x - MARGIN) // SQUARE_SIZE
        row = (y - MARGIN - 40) // SQUARE_SIZE
        return (row, col)
    
    def make_human_move(self, row, col):
        """Make a move for the human player."""
        action = row * BOARD_SIZE + col
        
        if action not in self.game.legal_actions():
            self.info_text = "Illegal move! Try again."
            return False
        
        # Save current board state for animation
        old_board = np.copy(self.game.env.board)
        
        # Make the move
        self.observation, reward, self.done = self.game.step(action)
        
        # Record the move
        self.last_move = (row, col)
        self.move_history.append((row, col, self.human_player))
        
        # Determine which pieces moved
        self.find_moved_pieces(old_board)
        
        # Check for win
        if self.done:
            if reward > 0:
                self.winner = self.human_player
                self.info_text = "You win! Game over."
            else:
                self.info_text = "Game ended in a draw."
            return True
        
        # Switch turn to AI
        self.info_text = "MuZero is thinking..."
        self.waiting_for_ai = True
        return True
    
    def find_moved_pieces(self, old_board):
        """Find which pieces moved for animation."""
        self.moved_pieces = []
        self.removed_pieces = []
        
        # Check for pieces that were on the old board but not on the new board
        for player in range(2):
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    # Piece was removed
                    if old_board[player, row, col] == 1 and self.game.env.board[player, row, col] == 0:
                        self.removed_pieces.append((row, col, player))
                    
                    # Piece is new or moved
                    if old_board[player, row, col] == 0 and self.game.env.board[player, row, col] == 1:
                        # Skip the just-placed piece
                        if (row, col) != self.last_move or player != self.game.env.to_play():
                            self.moved_pieces.append((row, col, player))
    
    def make_ai_move(self):
        """Make a move for the AI."""
        if self.model_loaded:
            # Use model to select the best action
            action = self.muzero.test(
                render=False, 
                opponent="self", 
                muzero_player=1 - self.human_player,
                num_tests=1
            )
            row = action // BOARD_SIZE
            col = action % BOARD_SIZE
            
            # In case there's an issue with the model's choice, fall back to random
            if action not in self.game.legal_actions():
                legal_actions = self.game.legal_actions()
                if legal_actions:
                    action = np.random.choice(legal_actions)
                    row = action // BOARD_SIZE
                    col = action % BOARD_SIZE
                else:
                    self.done = True
                    self.info_text = "No legal moves. Game ended in a draw."
                    return
        else:
            # Random play if no model is loaded
            legal_actions = self.game.legal_actions()
            if legal_actions:
                action = np.random.choice(legal_actions)
                row = action // BOARD_SIZE
                col = action % BOARD_SIZE
            else:
                self.done = True
                self.info_text = "No legal moves. Game ended in a draw."
                return
        
        # Save current board state for animation
        old_board = np.copy(self.game.env.board)
        
        # Make the move
        self.observation, reward, self.done = self.game.step(action)
        
        # Record the move
        self.last_move = (row, col)
        self.move_history.append((row, col, 1 - self.human_player))
        
        # Determine which pieces moved
        self.find_moved_pieces(old_board)
        
        # Check for win
        if self.done:
            if reward > 0:
                self.winner = 1 - self.human_player
                self.info_text = "MuZero wins! Game over."
            else:
                self.info_text = "Game ended in a draw."
        else:
            self.info_text = "Your turn. Click on an empty square to place a piece."
            
        self.waiting_for_ai = False
    
    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEMOTION:
                    self.highlighted_square = self.get_square_at_pos(event.pos)
                
                if event.type == pygame.MOUSEBUTTONDOWN and not self.done and not self.waiting_for_ai:
                    square = self.get_square_at_pos(event.pos)
                    if square:
                        row, col = square
                        if self.make_human_move(row, col):
                            # If move was successful, queue AI's move
                            pass
            
            # AI's turn
            if self.waiting_for_ai and not self.done:
                self.make_ai_move()
            
            # Draw the game
            self.draw_board()
            pygame.display.flip()
            
            # Cap at 60 FPS
            clock.tick(60)
        
        pygame.quit()


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
    
    game = Boop6x6GUI(model_path)
    game.run()


if __name__ == "__main__":
    main() 