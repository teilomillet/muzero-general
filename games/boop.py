import datetime
import pathlib
import numpy as np
import torch
import wandb

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 42  # Better seed for reproducibility
        self.max_num_gpus = 1  # Usually faster to use a single GPU

        ### Game
        self.observation_shape = (3, 6, 6)  # Dimensions of the game observation, must be 3D (channel, height, width)
        # Channel 0: Player 1 pieces
        # Channel 1: Player 2 pieces
        # Channel 2: Valid placement positions (1 if valid, 0 if not)
        self.action_space = list(range(36))  # 6x6 grid = 36 possible actions
        self.players = list(range(2))  # Two players
        self.stacked_observations = 2  # Increased to give more history context

        # Evaluate
        self.muzero_player = 0  # Turn MuZero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 4  # Increased for faster self-play
        self.selfplay_on_gpu = True  # Utilize GPU for self-play
        self.max_moves = 50  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Moderate number of simulations for faster iteration
        self.discount = 1.0  # Board game with final reward
        self.temperature_threshold = 10  # Apply temperature reduction after 10 moves

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3  # Slightly increased for more exploration
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # No downsampling needed for 6x6 board
        self.blocks = 4  # Moderate number of blocks for faster training
        self.channels = 64  # Moderate number of channels for faster training
        self.reduced_channels_reward = 64  # Match channels
        self.reduced_channels_value = 64  # Match channels
        self.reduced_channels_policy = 64  # Match channels
        self.resnet_fc_reward_layers = [64]  # Simplified for faster training
        self.resnet_fc_value_layers = [64]  # Simplified for faster training
        self.resnet_fc_policy_layers = [64]  # Simplified for faster training

        # Fully Connected Network
        self.encoding_size = 64  # Moderate size for faster training
        self.fc_representation_layers = [128]  # Simplified for faster training
        self.fc_dynamics_layers = [128]  # Simplified for faster training
        self.fc_reward_layers = [128]  # Simplified for faster training
        self.fc_value_layers = [128]  # Simplified for faster training
        self.fc_policy_layers = [128]  # Simplified for faster training

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000  # Reduced for initial testing
        self.batch_size = 128  # Moderate batch size for faster iteration
        self.checkpoint_interval = 100  # More frequent checkpoints relative to total steps
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "SGD"  # Changed to SGD as recommended in the paper
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.01  # Increased initial learning rate
        self.lr_decay_rate = 0.9  # Moderate decay
        self.lr_decay_steps = 1000  # Adjusted for shorter training

        ### Replay Buffer
        self.replay_buffer_size = 1000  # Scaled down for shorter training
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Match with max_moves for final reward
        self.PER = True  # Prioritized Replay
        self.PER_alpha = 1.0  # Full prioritization as suggested in paper

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value
        self.reanalyse_on_gpu = True  # Use GPU for reanalysis

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # No delay
        self.training_delay = 0  # No delay
        self.ratio = 1  # Balanced ratio of training steps to self-play steps
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:  # First 5000 steps
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:  # Next 2500 steps
            return 0.5
        else:  # Final 2500 steps
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper for Boop.
    """

    def __init__(self, seed=None):
        self.env = Boop()
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        
        # Log game state to wandb if it's initialized
        if wandb.run is not None and done:
            self.log_game_state_to_wandb()
            
        return observation, reward, done
        
    def log_game_state_to_wandb(self):
        """
        Log the current game board state to wandb for visualization.
        """
        if not hasattr(self, 'game_count'):
            self.game_count = 0
        else:
            self.game_count += 1
            
        # Create a visual representation of the board
        board_size = self.env.board_size
        board_image = np.zeros((board_size, board_size, 3), dtype=np.uint8)
        
        # Color coding: Player 1 (Red), Player 2 (Blue), Empty (White)
        for r in range(board_size):
            for c in range(board_size):
                if self.env.board[0, r, c] == 1:  # Player 1 pieces
                    board_image[r, c] = [255, 0, 0]  # Red
                elif self.env.board[1, r, c] == 1:  # Player 2 pieces
                    board_image[r, c] = [0, 0, 255]  # Blue
                else:
                    board_image[r, c] = [255, 255, 255]  # White for empty
        
        # Log the board state
        wandb.log({
            f"game_board_{self.game_count}": wandb.Image(board_image, 
                                                         caption=f"Game {self.game_count} - Final Board State")
        })

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                row = int(input(f"Enter the row (0-5) for player {self.to_play()}: "))
                col = int(input(f"Enter the column (0-5) for player {self.to_play()}: "))
                
                if 0 <= row <= 5 and 0 <= col <= 5:
                    action = row * 6 + col
                    if action in self.legal_actions():
                        return action
                
                print("Illegal action. Try again.")
            except ValueError:
                print("Please enter integers between 0 and 5.")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        row = action_number // 6
        col = action_number % 6
        return f"Place at position ({row}, {col})"


class Boop:
    """
    The Boop game.
    A 6x6 board where players place their pieces and try to create three in a row.
    When a piece is placed next to another piece, it "boops" (pushes) that piece in the opposite direction.
    """

    def __init__(self):
        self.board_size = 6
        self.board = np.zeros((2, self.board_size, self.board_size), dtype=np.int8)
        self.player = 0  # Player 0 starts
        self.done = False
        self.move_count = 0
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]  # All 8 directions

    def to_play(self):
        """
        Return the current player.
        """
        return self.player

    def reset(self):
        """
        Reset the game for a new game.
        """
        self.board = np.zeros((2, self.board_size, self.board_size), dtype=np.int8)
        self.player = 0
        self.done = False
        self.move_count = 0
        return self.get_observation()

    def step(self, action):
        """
        Apply action to the game.
        """
        if action not in self.legal_actions():
            # If the action is illegal, provide a negative reward
            return self.get_observation(), -1, False

        # Convert action to coordinates
        row = action // self.board_size
        col = action % self.board_size

        # Place the piece
        self.board[self.player, row, col] = 1

        # Apply "boop" (push) mechanics
        self.apply_boops(row, col)

        # Check for three in a row
        reward = 0
        if self.check_win():
            reward = 1
            self.done = True
        elif self.is_board_full():
            self.done = True  # Draw

        # Switch player if the game continues
        if not self.done:
            self.player = 1 - self.player
            
        self.move_count += 1
        
        # Check for max moves
        if self.move_count >= 50:
            self.done = True

        return self.get_observation(), reward, self.done

    def apply_boops(self, row, col):
        """
        Apply the "boop" mechanic - push adjacent pieces away from the placed piece.
        """
        for dr, dc in self.directions:
            adj_row, adj_col = row + dr, col + dc
            
            # Check if the adjacent position is valid and has a piece
            if (0 <= adj_row < self.board_size and 0 <= adj_col < self.board_size and 
                (self.board[0, adj_row, adj_col] == 1 or self.board[1, adj_row, adj_col] == 1)):
                
                # Calculate the new position after booping
                new_row, new_col = adj_row + dr, adj_col + dc
                
                # Check if the new position is valid (within the board)
                if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                    # Move the piece
                    for p in range(2):  # Check both players' pieces
                        if self.board[p, adj_row, adj_col] == 1:
                            # If the new position is already occupied, they both get removed
                            if (self.board[0, new_row, new_col] == 1 or self.board[1, new_row, new_col] == 1):
                                self.board[p, adj_row, adj_col] = 0  # Remove the booped piece
                                # Remove the piece at the new position
                                self.board[0, new_row, new_col] = 0
                                self.board[1, new_row, new_col] = 0
                            else:
                                # Move the piece to the new position
                                self.board[p, new_row, new_col] = 1
                                self.board[p, adj_row, adj_col] = 0
                else:
                    # The piece would go off the board, so it's removed
                    for p in range(2):
                        if self.board[p, adj_row, adj_col] == 1:
                            self.board[p, adj_row, adj_col] = 0

    def get_observation(self):
        """
        Get the current observation of the game.
        """
        # Create the observation with 3 channels:
        # - Channel 0: Player 1's pieces
        # - Channel 1: Player 2's pieces
        # - Channel 2: Valid placement positions (1 if valid, 0 if not)
        observation = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # Copy player pieces
        observation[0] = self.board[0]
        observation[1] = self.board[1]
        
        # Set valid placement positions
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[0, r, c] == 0 and self.board[1, r, c] == 0:
                    observation[2, r, c] = 1
        
        return observation

    def legal_actions(self):
        """
        Return the legal actions at the current state.
        """
        legal = []
        if self.done:
            return legal
        
        # Any empty cell is a legal move
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[0, r, c] == 0 and self.board[1, r, c] == 0:
                    legal.append(r * self.board_size + c)
        
        return legal

    def check_win(self):
        """
        Check if the current player has won by getting three in a row.
        """
        # Check rows
        for r in range(self.board_size):
            for c in range(self.board_size - 2):
                if (self.board[self.player, r, c] == 1 and
                    self.board[self.player, r, c+1] == 1 and
                    self.board[self.player, r, c+2] == 1):
                    return True
                    
        # Check columns
        for r in range(self.board_size - 2):
            for c in range(self.board_size):
                if (self.board[self.player, r, c] == 1 and
                    self.board[self.player, r+1, c] == 1 and
                    self.board[self.player, r+2, c] == 1):
                    return True
                    
        # Check diagonals (top-left to bottom-right)
        for r in range(self.board_size - 2):
            for c in range(self.board_size - 2):
                if (self.board[self.player, r, c] == 1 and
                    self.board[self.player, r+1, c+1] == 1 and
                    self.board[self.player, r+2, c+2] == 1):
                    return True
                    
        # Check diagonals (bottom-left to top-right)
        for r in range(2, self.board_size):
            for c in range(self.board_size - 2):
                if (self.board[self.player, r, c] == 1 and
                    self.board[self.player, r-1, c+1] == 1 and
                    self.board[self.player, r-2, c+2] == 1):
                    return True
        
        return False

    def is_board_full(self):
        """
        Check if the board is full.
        """
        return len(self.legal_actions()) == 0

    def render(self):
        """
        Display the board.
        """
        print("\n")
        print("   " + " ".join([str(i) for i in range(self.board_size)]))
        for r in range(self.board_size):
            print(f"{r}  ", end="")
            for c in range(self.board_size):
                if self.board[0, r, c] == 1:
                    print("X", end=" ")
                elif self.board[1, r, c] == 1:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print()
        print("\n")
        print(f"Player {self.player + 1}'s turn") 