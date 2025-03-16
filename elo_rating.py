import os
import json
import numpy as np
import pathlib
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import ray

from muzero import MuZero
import models  # Add this import for models module


class EloRating:
    """
    System to calculate Elo ratings for MuZero models.
    
    Elo rating is a method to calculate relative skill levels between players.
    This implementation allows MuZero models to compete and receive ratings.
    """
    
    def __init__(self, game_name, base_k=32, models_dir=None):
        """
        Initialize the Elo rating system.
        
        Args:
            game_name (str): Name of the game
            base_k (int): K-factor for Elo calculation (determines rating volatility)
            models_dir (str, optional): Directory containing model checkpoints,
                                       defaults to results/{game_name}
        """
        self.game_name = game_name
        self.base_k = base_k
        self.models_dir = models_dir or pathlib.Path(f"results/{game_name}")
        self.ratings = {}
        self.history = []
        self.results_file = pathlib.Path(f"results/{game_name}/elo_ratings.json")
        
        # Load existing ratings if available
        self._load_ratings()
    
    def _load_ratings(self):
        """Load existing ratings from the results file if it exists."""
        if self.results_file.exists():
            try:
                with open(self.results_file, "r") as f:
                    data = json.load(f)
                self.ratings = data.get("ratings", {})
                self.history = data.get("history", [])
                print(f"Loaded {len(self.ratings)} model ratings from {self.results_file}")
            except Exception as e:
                print(f"Error loading ratings: {e}")
                self.ratings = {}
                self.history = []
    
    def _save_ratings(self):
        """Save current ratings to the results file."""
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, "w") as f:
            json.dump({
                "ratings": self.ratings,
                "history": self.history,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Saved ratings to {self.results_file}")
    
    def _get_expected_score(self, rating_a, rating_b):
        """
        Calculate expected score based on Elo formula.
        
        Args:
            rating_a (float): Rating of player A
            rating_b (float): Rating of player B
            
        Returns:
            float: Expected score for player A (between 0 and 1)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def _update_rating(self, rating, expected, actual, k_factor=None):
        """
        Update rating based on game outcome.
        
        Args:
            rating (float): Current rating
            expected (float): Expected score
            actual (float): Actual score (1 for win, 0.5 for draw, 0 for loss)
            k_factor (float, optional): K-factor to use (defaults to self.base_k)
            
        Returns:
            float: Updated rating
        """
        k = k_factor or self.base_k
        return rating + k * (actual - expected)
    
    def _get_model_paths(self):
        """
        Get paths to all available model checkpoints.
        
        Returns:
            list: List of paths to model checkpoints
        """
        model_paths = []
        for model_dir in sorted(self.models_dir.glob("*/"), reverse=True):
            checkpoint_path = model_dir / "model.checkpoint"
            if checkpoint_path.exists():
                model_paths.append(model_dir)
        
        return model_paths
    
    def add_model(self, model_path, initial_rating=1200):
        """
        Add a new model to the rating system.
        
        Args:
            model_path (str or Path): Path to the model checkpoint directory
            initial_rating (float): Initial rating to assign
            
        Returns:
            str: Model ID
        """
        model_path = pathlib.Path(model_path)
        model_id = model_path.name
        
        if model_id not in self.ratings:
            self.ratings[model_id] = {
                "rating": initial_rating,
                "path": str(model_path),
                "games_played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0
            }
            print(f"Added model {model_id} with initial rating {initial_rating}")
            self._save_ratings()
        
        return model_id
    
    def play_match(self, model_a_path, model_b_path, num_games=10, muzero_instance=None):
        """
        Play matches between two models and update their ratings.
        
        Args:
            model_a_path (str or Path): Path to model A checkpoint directory
            model_b_path (str or Path): Path to model B checkpoint directory
            num_games (int): Number of games to play
            muzero_instance (MuZero, optional): Existing MuZero instance to use
            
        Returns:
            dict: Match results
        """
        model_a_path = pathlib.Path(model_a_path)
        model_b_path = pathlib.Path(model_b_path)
        
        model_a_id = model_a_path.name
        model_b_id = model_b_path.name
        
        # Add models if they don't exist in ratings
        if model_a_id not in self.ratings:
            self.add_model(model_a_path)
        if model_b_id not in self.ratings:
            self.add_model(model_b_path)
        
        # Initialize MuZero if not provided
        cleanup_muzero = False
        if muzero_instance is None:
            muzero_instance = MuZero(self.game_name)
            cleanup_muzero = True
        
        # Store original weights to restore after matches
        original_weights = muzero_instance.checkpoint["weights"]
        
        wins_a = 0
        wins_b = 0
        draws = 0
        
        print(f"Playing {num_games} games: {model_a_id} vs {model_b_id}")
        
        for i in tqdm(range(num_games)):
            # Alternate who plays first
            if i % 2 == 0:
                first_player = 0
                first_model_id = model_a_id
                second_model_id = model_b_id
            else:
                first_player = 1
                first_model_id = model_b_id
                second_model_id = model_a_id
            
            # First half of games: Model A as player 0, Model B as player 1
            # Second half: Model B as player 0, Model A as player 1
            
            # Load first player model
            first_model_path = model_a_path if first_model_id == model_a_id else model_b_path
            muzero_instance.load_model(checkpoint_path=first_model_path / "model.checkpoint")
            first_weights = muzero_instance.checkpoint["weights"]
            
            # Load second player model
            second_model_path = model_b_path if second_model_id == model_b_id else model_a_path
            muzero_instance.load_model(checkpoint_path=second_model_path / "model.checkpoint")
            second_weights = muzero_instance.checkpoint["weights"]
            
            # Create a new self play worker for the match
            config = muzero_instance.config
            config.opponent = "self"  # Make sure it's self-play mode
            
            # Play the game - we need to extend the Game class to support two different weights
            game_history = self._play_game_with_two_models(
                muzero_instance, 
                first_weights, 
                second_weights,
                first_player
            )
            
            # Determine the winner
            if len(config.players) == 1:
                # Single-player game (not typical for Elo comparison)
                print("Warning: Single-player games not ideal for Elo ratings")
                total_reward = sum(game_history.reward_history)
                winner = 0 if total_reward > 0 else 1 if total_reward < 0 else None
            else:
                # Two-player game
                rewards = [0, 0]
                for i, reward in enumerate(game_history.reward_history):
                    if i > 0:  # Skip initial reward
                        player = game_history.to_play_history[i-1]
                        rewards[player] += reward
                
                if rewards[0] > rewards[1]:
                    winner = 0
                elif rewards[1] > rewards[0]:
                    winner = 1
                else:
                    winner = None  # Draw
            
            # Record result
            if winner == 0:
                if first_player == 0:  # Model A won
                    wins_a += 1
                else:  # Model B won
                    wins_b += 1
            elif winner == 1:
                if first_player == 0:  # Model B won
                    wins_b += 1
                else:  # Model A won
                    wins_a += 1
            else:
                draws += 1
        
        # Restore original weights
        muzero_instance.checkpoint["weights"] = original_weights
        
        # Calculate results
        result = {
            "model_a": model_a_id,
            "model_b": model_b_id,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "games_played": num_games
        }
        
        # Update ratings
        self._update_ratings_from_match(result)
        
        # Cleanup if we created the muzero instance
        if cleanup_muzero:
            ray.shutdown()
        
        return result
    
    def _play_game_with_two_models(self, muzero_instance, first_model_weights, second_model_weights, first_player):
        """
        Play a game with two different model weights.
        
        Args:
            muzero_instance (MuZero): MuZero instance
            first_model_weights (dict): Weights for first player
            second_model_weights (dict): Weights for second player
            first_player (int): Which player goes first (0 or 1)
            
        Returns:
            GameHistory: History of the played game
        """
        config = muzero_instance.config
        game = muzero_instance.Game(config.seed)
        
        # Create a self-play worker with custom modifications
        self_play_worker = CustomSelfPlay.remote(
            muzero_instance.checkpoint,
            muzero_instance.Game,
            config,
            np.random.randint(10000),
            model2_weights=second_model_weights,
            first_model_player=first_player
        )
        
        # Play the game
        game_history = ray.get(
            self_play_worker.play_game.remote(
                0,  # temperature (0 = deterministic)
                config.temperature_threshold,
                False,  # render
                "self",  # opponent
                0,  # muzero_player
            )
        )
        
        # Clean up
        self_play_worker.close_game.remote()
        
        return game_history
    
    def _update_ratings_from_match(self, result):
        """
        Update ratings for both models based on match results.
        
        Args:
            result (dict): Match results
        """
        model_a = result["model_a"]
        model_b = result["model_b"]
        
        total_games = result["games_played"]
        score_a = result["wins_a"] + (result["draws"] / 2)
        score_b = result["wins_b"] + (result["draws"] / 2)
        
        # Get current ratings
        rating_a = self.ratings[model_a]["rating"]
        rating_b = self.ratings[model_b]["rating"]
        
        # Calculate expected scores
        expected_a = self._get_expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        # Update ratings
        new_rating_a = self._update_rating(rating_a, expected_a, score_a / total_games)
        new_rating_b = self._update_rating(rating_b, expected_b, score_b / total_games)
        
        # Update stats
        self.ratings[model_a]["rating"] = new_rating_a
        self.ratings[model_a]["games_played"] += total_games
        self.ratings[model_a]["wins"] += result["wins_a"]
        self.ratings[model_a]["draws"] += result["draws"]
        self.ratings[model_a]["losses"] += result["wins_b"]
        
        self.ratings[model_b]["rating"] = new_rating_b
        self.ratings[model_b]["games_played"] += total_games
        self.ratings[model_b]["wins"] += result["wins_b"]
        self.ratings[model_b]["draws"] += result["draws"]
        self.ratings[model_b]["losses"] += result["wins_a"]
        
        # Record match in history
        match_record = {
            "timestamp": datetime.now().isoformat(),
            "model_a": model_a,
            "model_b": model_b,
            "wins_a": result["wins_a"],
            "wins_b": result["wins_b"],
            "draws": result["draws"],
            "old_rating_a": rating_a,
            "new_rating_a": new_rating_a,
            "old_rating_b": rating_b,
            "new_rating_b": new_rating_b
        }
        self.history.append(match_record)
        
        # Print results
        print(f"Match results: {model_a} vs {model_b}")
        print(f"Games played: {total_games}, {model_a} wins: {result['wins_a']}, {model_b} wins: {result['wins_b']}, Draws: {result['draws']}")
        print(f"Rating changes: {model_a}: {rating_a:.1f} → {new_rating_a:.1f}, {model_b}: {rating_b:.1f} → {new_rating_b:.1f}")
        
        # Save updated ratings
        self._save_ratings()
    
    def run_tournament(self, model_dirs=None, num_games=10, muzero_instance=None):
        """
        Run a tournament between multiple models.
        
        Args:
            model_dirs (list, optional): List of model directories to include in tournament
            num_games (int): Number of games to play for each matchup
            muzero_instance (MuZero, optional): Existing MuZero instance to use
            
        Returns:
            dict: Tournament results
        """
        # Get model paths if not provided
        if model_dirs is None:
            model_dirs = self._get_model_paths()
        else:
            model_dirs = [pathlib.Path(d) for d in model_dirs]
        
        # Initialize MuZero if not provided
        cleanup_muzero = False
        if muzero_instance is None:
            muzero_instance = MuZero(self.game_name)
            cleanup_muzero = True
        
        # Play all pairs of models
        results = []
        for i, model_a in enumerate(model_dirs):
            for model_b in model_dirs[i+1:]:  # Only play each pair once
                result = self.play_match(model_a, model_b, num_games, muzero_instance)
                results.append(result)
        
        # Cleanup if we created the MuZero instance
        if cleanup_muzero:
            ray.shutdown()
        
        return results
    
    def get_ratings_table(self):
        """
        Generate a pandas DataFrame with ratings information.
        
        Returns:
            pd.DataFrame: DataFrame with model ratings
        """
        data = []
        for model_id, stats in self.ratings.items():
            data.append({
                "model_id": model_id,
                "rating": stats["rating"],
                "games_played": stats["games_played"],
                "wins": stats["wins"],
                "draws": stats["draws"],
                "losses": stats["losses"],
                "win_rate": stats["wins"] / max(1, stats["games_played"]),
                "path": stats["path"]
            })
        
        df = pd.DataFrame(data)
        return df.sort_values("rating", ascending=False).reset_index(drop=True)
    
    def plot_ratings(self, output_path=None):
        """
        Plot the evolution of ratings over time.
        
        Args:
            output_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if not self.history:
            print("No rating history available to plot")
            return None
        
        # Extract rating changes over time
        rating_history = {}
        
        # Initialize with starting ratings
        for match in self.history:
            if match["model_a"] not in rating_history:
                rating_history[match["model_a"]] = [(None, match["old_rating_a"])]
            if match["model_b"] not in rating_history:
                rating_history[match["model_b"]] = [(None, match["old_rating_b"])]
        
        # Add each rating change
        for i, match in enumerate(self.history):
            timestamp = datetime.fromisoformat(match["timestamp"])
            rating_history[match["model_a"]].append((timestamp, match["new_rating_a"]))
            rating_history[match["model_b"]].append((timestamp, match["new_rating_b"]))
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for model_id, ratings in rating_history.items():
            # Skip the first entry which has no timestamp
            timestamps = [r[0] for r in ratings[1:]]
            values = [r[1] for r in ratings[1:]]
            
            # Add the line
            ax.plot(timestamps, values, marker='o', label=model_id)
        
        ax.set_title("Model Rating Evolution")
        ax.set_xlabel("Time")
        ax.set_ylabel("Elo Rating")
        ax.grid(True)
        ax.legend()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Saved rating plot to {output_path}")
        
        return fig


@ray.remote
class CustomSelfPlay:
    """
    Modified version of SelfPlay class that supports two different model weights.
    """
    
    def __init__(self, initial_checkpoint, Game, config, seed, model2_weights=None, first_model_player=0):
        """
        Initialize with support for two models.
        
        Args:
            initial_checkpoint: Initial model checkpoint
            Game: Game class
            config: Configuration
            seed: Random seed
            model2_weights: Weights for the second model
            first_model_player: Which player uses the first model (0 or 1)
        """
        self.config = config
        self.game = Game(seed)
        
        # Fix random generator seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.selfplay_on_gpu else "cpu")
        
        # Initialize the primary model
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize the secondary model if provided
        self.model2 = None
        if model2_weights is not None:
            self.model2 = models.MuZeroNetwork(self.config)
            self.model2.set_weights(model2_weights)
            self.model2.to(self.device)
            self.model2.eval()
        
        # Which player uses which model
        self.first_model_player = first_model_player
    
    def play_game(self, temperature, temperature_threshold, render, opponent, muzero_player):
        """
        Play a game using the two different models for different players.
        """
        from self_play import MCTS, GameHistory
        
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())
        
        done = False
        
        if render:
            self.game.render()
        
        with torch.no_grad():
            while not done and len(game_history.action_history) <= self.config.max_moves:
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )
                
                # Choose the action
                if opponent == "self" or muzero_player == self.game.to_play():
                    # Select which model to use based on current player
                    current_player = self.game.to_play()
                    
                    # Determine which model to use
                    if (current_player == self.first_model_player) or self.model2 is None:
                        current_model = self.model
                    else:
                        current_model = self.model2
                    
                    # Run MCTS with the selected model
                    root, mcts_info = MCTS(self.config).run(
                        current_model,
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.to_play(),
                        True,
                    )
                    
                    # Select action based on visit counts
                    action = self.select_action(
                        root,
                        temperature if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )
                    
                    if render:
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
                        print(f"Model used: {'first' if current_model == self.model else 'second'}")
                else:
                    action, root = self.select_opponent_action(opponent, stacked_observations)
                
                observation, reward, done = self.game.step(action)
                
                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()
                
                game_history.store_search_statistics(root, self.config.action_space)
                
                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())
        
        return game_history
    
    def close_game(self):
        self.game.close()
    
    def select_opponent_action(self, opponent, stacked_observations):
        """Select opponent action - same as in original SelfPlay."""
        from self_play import MCTS
        
        if opponent == "human":
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert self.game.legal_actions(), f"Legal actions should not be empty. Got {self.game.legal_actions()}."
            return np.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )
    
    @staticmethod
    def select_action(node, temperature):
        """Select action according to the visit count distribution and temperature."""
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)
        
        return action


def main():
    """
    Run the ELO rating system as a standalone script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="MuZero ELO Rating System")
    parser.add_argument("--game", type=str, required=True, help="Name of the game")
    parser.add_argument("--models_dir", type=str, help="Directory containing model checkpoints")
    parser.add_argument("--tournament", action="store_true", help="Run a tournament between all models")
    parser.add_argument("--num_games", type=int, default=10, help="Number of games per matchup")
    parser.add_argument("--match", action="store_true", help="Run a match between two specific models")
    parser.add_argument("--model_a", type=str, help="Path to first model for match")
    parser.add_argument("--model_b", type=str, help="Path to second model for match")
    
    args = parser.parse_args()
    
    # Initialize the ELO rating system
    elo = EloRating(args.game, models_dir=args.models_dir)
    
    if args.tournament:
        print(f"Running tournament for {args.game} with {args.num_games} games per matchup")
        elo.run_tournament(num_games=args.num_games)
        
        # Print ratings table
        ratings_table = elo.get_ratings_table()
        print("\nFinal Ratings:")
        print(ratings_table[["model_id", "rating", "games_played", "wins", "draws", "losses", "win_rate"]])
        
        # Plot ratings
        fig = elo.plot_ratings(output_path=f"results/{args.game}/elo_ratings.png")
        plt.show()
    
    elif args.match:
        if not args.model_a or not args.model_b:
            print("Error: Both --model_a and --model_b must be specified for a match")
            return
        
        print(f"Running match between {args.model_a} and {args.model_b}")
        elo.play_match(args.model_a, args.model_b, num_games=args.num_games)
        
        # Print ratings
        ratings_table = elo.get_ratings_table()
        print("\nCurrent Ratings:")
        print(ratings_table[["model_id", "rating", "games_played", "wins", "draws", "losses", "win_rate"]])
    
    else:
        print("No action specified. Use --tournament or --match")
        
        # Print current ratings if they exist
        if elo.ratings:
            ratings_table = elo.get_ratings_table()
            print("\nCurrent Ratings:")
            print(ratings_table[["model_id", "rating", "games_played", "wins", "draws", "losses", "win_rate"]])


if __name__ == "__main__":
    main() 