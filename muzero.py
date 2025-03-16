import copy
import importlib
import json
import math
import pathlib
import pickle
import sys
import time

import nevergrad
import numpy
import ray
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

import diagnose_model
import models
import replay_buffer
import self_play
import shared_storage
import trainer

# Add ELO rating import
try:
    from elo_rating import EloRating
except ImportError:
    EloRating = None

class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        config (dict, MuZeroConfig, optional): Override the default config of the game.

        split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True)
    """

    def __init__(self, game_name, config=None, split_resources_in=1):
        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        # Overwrite the config
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    if hasattr(self.config, param):
                        setattr(self.config, param, value)
                    else:
                        raise AttributeError(
                            f"{game_name} config has no attribute '{param}'. Check the config file for the complete list of parameters."
                        )
            else:
                self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Manage GPUs
        if self.config.max_num_gpus == 0 and (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            raise ValueError(
                "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
            )
        if (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        ray.init(num_gpus=total_gpus, ignore_reinit_error=True)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }
        self.replay_buffer = {}

        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        if log_in_tensorboard or self.config.save_model:
            self.config.results_path.mkdir(parents=True, exist_ok=True)
            
        # Log config to wandb
        if wandb.run is not None:
            wandb.config.update({k: v for k, v in self.config.__dict__.items() 
                               if not k.startswith('__') and not callable(v)},
                              allow_val_change=True)

        # Manage GPUs
        if 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + log_in_tensorboard * self.config.selfplay_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0

        # Initialize workers
        self.training_worker = trainer.Trainer.options(
            num_cpus=0,
            num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        ).remote(self.checkpoint, self.config)

        self.shared_storage_worker = shared_storage.SharedStorage.remote(
            self.checkpoint,
            self.config,
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
            self.checkpoint, self.replay_buffer, self.config
        )

        if self.config.use_last_model_value:
            self.reanalyse_worker = replay_buffer.Reanalyse.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
            ).remote(self.checkpoint, self.config)

        self.self_play_workers = [
            self_play.SelfPlay.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            ).remote(
                self.checkpoint,
                self.Game,
                self.config,
                self.config.seed + seed,
            )
            for seed in range(self.config.num_workers)
        ]

        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_play_worker in self.self_play_workers
        ]
        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )
        if self.config.use_last_model_value:
            self.reanalyse_worker.reanalyse.remote(
                self.replay_buffer_worker, self.shared_storage_worker
            )

        if log_in_tensorboard:
            self.logging_loop(
                num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            )

    def logging_loop(self, num_gpus):
        """
        Keep track of the training performance.
        """
        # Launch the test worker to get performance metrics
        self.test_worker = self_play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(
            self.checkpoint,
            self.Game,
            self.config,
            self.config.seed + self.config.num_workers,
        )
        self.test_worker.continuous_self_play.remote(
            self.shared_storage_worker, None, True
        )

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path)

        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Save model representation
        writer.add_text(
            "Model summary",
            self.summary,
        )
        # Loop for updating the training performance
        counter = 0
        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                writer.add_scalar(
                    "1.Total_reward/1.Total_reward", info["total_reward"], counter
                )
                writer.add_scalar(
                    "1.Total_reward/2.Mean_value", info["mean_value"], counter
                )
                writer.add_scalar(
                    "1.Total_reward/3.Episode_length", info["episode_length"], counter
                )
                writer.add_scalar(
                    "1.Total_reward/4.MuZero_reward", info["muzero_reward"], counter
                )
                writer.add_scalar(
                    "1.Total_reward/5.Opponent_reward", info["opponent_reward"], counter
                )
                
                writer.add_scalar(
                    "2.Workers/1.Self_played_games", info["num_played_games"], counter
                )
                writer.add_scalar(
                    "2.Workers/2.Training_steps", info["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/3.Self_played_steps", info["num_played_steps"], counter
                )
                writer.add_scalar(
                    "2.Workers/4.Reanalysed_games",
                    info["num_reanalysed_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/5.Training_steps_per_self_played_step_ratio",
                    info["training_step"] / max(1, info["num_played_steps"]),
                    counter,
                )
                writer.add_scalar("3.Loss/1.Total_loss", info["total_loss"], counter)
                writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
                writer.add_scalar("3.Loss/3.Reward_loss", info["reward_loss"], counter)
                writer.add_scalar("3.Loss/4.Policy_loss", info["policy_loss"], counter)
                writer.add_scalar("4.Model/1.Learning_rate", info["lr"], counter)
                
                # Log metrics to wandb
                if wandb.run is not None:
                    wandb.log({
                        "total_reward": info["total_reward"],
                        "mean_value": info["mean_value"],
                        "episode_length": info["episode_length"],
                        "muzero_reward": info["muzero_reward"],
                        "opponent_reward": info["opponent_reward"],
                        "self_played_games": info["num_played_games"],
                        "training_step": info["training_step"],
                        "self_played_steps": info["num_played_steps"],
                        "reanalysed_games": info["num_reanalysed_games"],
                        "training_steps_per_self_played_step_ratio": info["training_step"] / max(1, info["num_played_steps"]),
                        "total_loss": info["total_loss"],
                        "value_loss": info["value_loss"],
                        "reward_loss": info["reward_loss"],
                        "policy_loss": info["policy_loss"],
                        "learning_rate": info["lr"],
                    })
                
                print(
                    f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
                    end="\r",
                )
                
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        self.terminate_workers()
        if wandb.run is not None:
            wandb.finish()

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def test(
        self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0
    ):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """
        opponent = opponent if opponent else self.config.opponent
        muzero_player = muzero_player if muzero_player else self.config.muzero_player
        self_play_worker = self_play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(self.checkpoint, self.Game, self.config, numpy.random.randint(10000))
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(
                ray.get(
                    self_play_worker.play_game.remote(
                        0,
                        0,
                        render,
                        opponent,
                        muzero_player,
                    )
                )
            )
        self_play_worker.close_game.remote()

        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [
                    sum(
                        reward
                        for i, reward in enumerate(history.reward_history)
                        if history.to_play_history[i - 1] == muzero_player
                    )
                    for history in results
                ]
            )
        return result

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            checkpoint_path = pathlib.Path(checkpoint_path)
            self.checkpoint = torch.load(checkpoint_path)
            print(f"\nUsing checkpoint from {checkpoint_path}")
            
            # Check for config file
            config_path = checkpoint_path.parent / "model.config.json"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    
                    # Update config with loaded values
                    for key, value in config_dict.items():
                        if hasattr(self.config, key):
                            if key == "results_path":
                                # Keep the current results_path
                                continue
                            setattr(self.config, key, value)
                    print(f"Loaded configuration from {config_path}")
                except Exception as e:
                    print(f"Error loading configuration: {e}")
            else:
                print("No configuration file found. Using default configuration.")

        # Load replay buffer
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_played_steps"] = replay_buffer_infos[
                "num_played_steps"
            ]
            self.checkpoint["num_played_games"] = replay_buffer_infos[
                "num_played_games"
            ]
            self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                "num_reanalysed_games"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")
        else:
            print(f"Using empty buffer.")
            self.replay_buffer = {}
            self.checkpoint["training_step"] = 0
            self.checkpoint["num_played_steps"] = 0
            self.checkpoint["num_played_games"] = 0
            self.checkpoint["num_reanalysed_games"] = 0

    def diagnose_model(self, horizon):
        """
        Play a game only with the learned model then play the same trajectory in the real
        environment and display information.

        Args:
            horizon (int): Number of timesteps for which we collect information.
        """
        game = self.Game(self.config.seed)
        obs = game.reset()
        dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
        dm.compare_virtual_with_real_trajectories(obs, game, horizon)
        input("Press enter to close all plots")
        dm.close_all()

    @staticmethod
    def get_config_from_file(config_path):
        """
        Load a configuration from a JSON file and return it as a dictionary.
        
        Args:
            config_path (str): Path to the config JSON file.
            
        Returns:
            dict: The loaded configuration, or None if loading failed.
        """
        try:
            config_path = pathlib.Path(config_path)
            if not config_path.exists():
                print(f"Configuration file {config_path} not found.")
                return None
            
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            
            return config_dict
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return None

    @staticmethod
    def get_game_parameters(config_dict):
        """
        Extract important game parameters from a configuration dictionary.
        
        Args:
            config_dict (dict): A configuration dictionary.
            
        Returns:
            dict: A dictionary of important game parameters.
        """
        if not config_dict:
            return {}
        
        important_keys = [
            "observation_shape", "action_space", "players", 
            "muzero_player", "opponent", "num_workers",
            "max_moves", "num_simulations", "discount",
            "network", "blocks", "channels",
            "training_steps", "train_on_gpu", 
            "selfplay_on_gpu", "seed"
        ]
        
        return {k: config_dict[k] for k in important_keys if k in config_dict}

    def evaluate_elo(self, other_model_paths=None, num_games=10, base_k=32):
        """
        Evaluate current model against other models using ELO rating system.
        
        Args:
            other_model_paths (list): List of paths to other model checkpoints to evaluate against
            num_games (int): Number of games to play per matchup
            base_k (int): K-factor for ELO calculation
            
        Returns:
            tuple: (EloRating object, ratings table DataFrame)
        """
        if EloRating is None:
            print("Error: Could not import EloRating. Make sure elo_rating.py is in your path.")
            return None, None
            
        # Extract game name from config
        game_name = None
        for module_name, module in sys.modules.items():
            if hasattr(module, 'Game') and hasattr(module, 'MuZeroConfig'):
                if isinstance(self.Game, module.Game.__class__):
                    game_name = module_name.split('.')[-1]
                    break
                    
        if game_name is None:
            game_name = input("Could not determine game name. Please enter it: ")
            
        # Initialize ELO rating system
        elo = EloRating(game_name, base_k=base_k)
        
        # Get current model path - use results_path if it exists
        current_model_path = self.config.results_path
        if not current_model_path.exists():
            # Create a temporary directory for the current model
            current_model_path = pathlib.Path(f"results/{game_name}/current_model")
            current_model_path.mkdir(parents=True, exist_ok=True)
            # Save the current model
            torch.save(
                self.checkpoint,
                current_model_path / "model.checkpoint",
            )
        
        # Add current model to ELO ratings
        elo.add_model(current_model_path)
        
        # If no other models specified, use all models in results directory
        if other_model_paths is None:
            other_model_paths = elo._get_model_paths()
            # Remove current model path if it's there
            other_model_paths = [p for p in other_model_paths if p != current_model_path]
        
        # Run tournament
        results = []
        for other_path in other_model_paths:
            result = elo.play_match(current_model_path, other_path, num_games=num_games, muzero_instance=self)
            results.append(result)
        
        # Get ratings table
        ratings_table = elo.get_ratings_table()
        
        # Print ratings
        print("\nELO Ratings:")
        print(ratings_table[["model_id", "rating", "games_played", "wins", "draws", "losses", "win_rate"]])
        
        return elo, ratings_table


@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = models.MuZeroNetwork(config)
        # Initialize weights with numerically stable values to prevent NaN issues
        model.initialize_weights()
        
        # Add additional stability check - run a test forward pass
        try:
            # Create a dummy input with the correct shape, accounting for stacked observations
            observation_shape = config.observation_shape
            stacked_observations = config.stacked_observations
            
            # The first conv layer expects:
            # observation_shape[0] * (stacked_observations + 1) + stacked_observations channels
            # This accounts for the current observation, stacked past observations, and stacked actions
            input_channels = observation_shape[0] * (stacked_observations + 1) + stacked_observations
            
            # Create properly shaped dummy input tensor
            dummy_input = torch.zeros((1, input_channels, observation_shape[1], observation_shape[2]))
            
            print(f"Testing model with dummy input shape: {dummy_input.shape}")
            
            # Run a test forward pass to verify numerical stability
            with torch.no_grad():
                # Initial inference
                try:
                    value, reward, policy, encoded_state = model.initial_inference(dummy_input)
                    
                    # Check for NaNs or infinite values
                    if (torch.isnan(value).any() or torch.isnan(reward).any() or torch.isnan(policy).any() or
                        torch.isinf(value).any() or torch.isinf(reward).any() or torch.isinf(policy).any()):
                        # Reinitialize if unstable values detected
                        print("Unstable values detected in initial model. Reinitializing...")
                        model = models.MuZeroNetwork(config)
                        model.initialize_weights()
                    
                    # Check the dynamics function too
                    dummy_action = torch.zeros((1, 1), dtype=torch.long)
                    value, reward, policy, _ = model.recurrent_inference(encoded_state, dummy_action)
                    
                    # Check for NaNs or infinite values again
                    if (torch.isnan(value).any() or torch.isnan(reward).any() or torch.isnan(policy).any() or
                        torch.isinf(value).any() or torch.isinf(reward).any() or torch.isinf(policy).any()):
                        # Apply stabilization to the problematic components
                        print("Stabilizing recurrent part of the model...")
                        model.initialize_weights()
                except Exception as e:
                    print(f"Forward pass failed: {e}")
                    # Try more aggressive initialization
                    print("Applying more conservative initialization strategy...")
                    model = models.MuZeroNetwork(config)
                    model.initialize_weights()
                
        except Exception as e:
            print(f"Error during model stability check: {e}")
            # Proceed with the current model regardless of the error
        
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary


def hyperparameter_search(
    game_name, parametrization, budget, parallel_experiments, num_tests
):
    """
    Search for hyperparameters by launching parallel experiments.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        parametrization : Nevergrad parametrization, please refer to nevergrad documentation.

        budget (int): Number of experiments to launch in total.

        parallel_experiments (int): Number of experiments to launch in parallel.

        num_tests (int): Number of games to average for evaluating an experiment.
    """
    optimizer = nevergrad.optimizers.OnePlusOne(
        parametrization=parametrization, budget=budget
    )

    running_experiments = []
    best_training = None
    try:
        # Launch initial experiments
        for i in range(parallel_experiments):
            if 0 < budget:
                param = optimizer.ask()
                print(f"Launching new experiment: {param.value}")
                muzero = MuZero(game_name, param.value, parallel_experiments)
                muzero.param = param
                muzero.train(False)
                running_experiments.append(muzero)
                budget -= 1

        while 0 < budget or any(running_experiments):
            for i, experiment in enumerate(running_experiments):
                if experiment and experiment.config.training_steps <= ray.get(
                    experiment.shared_storage_worker.get_info.remote("training_step")
                ):
                    experiment.terminate_workers()
                    result = experiment.test(False, num_tests=num_tests)
                    if not best_training or best_training["result"] < result:
                        best_training = {
                            "result": result,
                            "config": experiment.config,
                            "checkpoint": experiment.checkpoint,
                        }
                    print(f"Parameters: {experiment.param.value}")
                    print(f"Result: {result}")
                    optimizer.tell(experiment.param, -result)

                    if 0 < budget:
                        param = optimizer.ask()
                        print(f"Launching new experiment: {param.value}")
                        muzero = MuZero(game_name, param.value, parallel_experiments)
                        muzero.param = param
                        muzero.train(False)
                        running_experiments[i] = muzero
                        budget -= 1
                    else:
                        running_experiments[i] = None

    except KeyboardInterrupt:
        for experiment in running_experiments:
            if isinstance(experiment, MuZero):
                experiment.terminate_workers()

    recommendation = optimizer.provide_recommendation()
    print("Best hyperparameters:")
    print(recommendation.value)
    if best_training:
        # Save best training weights (but it's not the recommended weights)
        best_training["config"].results_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            best_training["checkpoint"],
            best_training["config"].results_path / "model.checkpoint",
        )
        # Save the recommended hyperparameters
        text_file = open(
            best_training["config"].results_path / "best_parameters.txt",
            "w",
        )
        text_file.write(str(recommendation.value))
        text_file.close()
    return recommendation.value


def load_model_menu(muzero, game_name):
    # Configure running options
    options = ["Specify paths manually"] + sorted(
        (pathlib.Path("results") / game_name).glob("*/")
    )
    options.reverse()
    print()
    for i in range(len(options)):
        if i > 0:  # Not for manual option
            # Try to get model info from config file
            config_path = options[i] / "model.config.json"
            if config_path.exists():
                config_dict = MuZero.get_config_from_file(config_path)
                if config_dict:
                    params = MuZero.get_game_parameters(config_dict)
                    info_str = f" - Training steps: {params.get('training_steps', '?')}"
                    info_str += f", Network: {params.get('network', '?')}"
                    info_str += f", Model date: {options[i].name[:10]}"
                    print(f"{i}. {options[i]} {info_str}")
                    continue
        print(f"{i}. {options[i]}")

    choice = input("Enter a number to choose a model to load: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)

    if choice == (len(options) - 1):
        # manual path option
        checkpoint_path = input(
            "Enter a path to the model.checkpoint, or ENTER if none: "
        )
        while checkpoint_path and not pathlib.Path(checkpoint_path).is_file():
            checkpoint_path = input("Invalid checkpoint path. Try again: ")
        replay_buffer_path = input(
            "Enter a path to the replay_buffer.pkl, or ENTER if none: "
        )
        while replay_buffer_path and not pathlib.Path(replay_buffer_path).is_file():
            replay_buffer_path = input("Invalid replay buffer path. Try again: ")
    else:
        checkpoint_path = options[choice] / "model.checkpoint"
        replay_buffer_path = options[choice] / "replay_buffer.pkl"
        
        # Display configuration summary
        config_path = options[choice] / "model.config.json"
        if config_path.exists():
            config_dict = MuZero.get_config_from_file(config_path)
            if config_dict:
                params = MuZero.get_game_parameters(config_dict)
                print("\nModel configuration summary:")
                for key, value in params.items():
                    if isinstance(value, list) and len(value) > 10:
                        value = f"[{len(value)} items]"
                    print(f"  {key}: {value}")
                print()

    muzero.load_model(
        checkpoint_path=checkpoint_path,
        replay_buffer_path=replay_buffer_path,
    )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Train directly with: python muzero.py cartpole
        muzero = MuZero(sys.argv[1])
        muzero.train()
    elif len(sys.argv) == 3:
        # Train directly with: python muzero.py cartpole '{"lr_init": 0.01}'
        config = json.loads(sys.argv[2])
        muzero = MuZero(sys.argv[1], config)
        muzero.train()
    else:
        print("\nWelcome to MuZero! Here's a list of games:")
        # Let user pick a game
        games = [
            filename.stem
            for filename in sorted(list((pathlib.Path.cwd() / "games").glob("*.py")))
            if filename.name != "abstract_game.py"
        ]
        for i in range(len(games)):
            print(f"{i}. {games[i]}")
        choice = input("Enter a number to choose the game: ")
        valid_inputs = [str(i) for i in range(len(games))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")

        # Initialize MuZero
        choice = int(choice)
        game_name = games[choice]
        muzero = MuZero(game_name)

        while True:
            # Configure running options
            options = [
                "Train",
                "Load pretrained model",
                "Diagnose model",
                "Render some self play games",
                "Play against MuZero",
                "Test the game manually",
                "Hyperparameter search",
                "Run ELO evaluation",
                "Exit",
            ]
            print()
            for i in range(len(options)):
                print(f"{i}. {options[i]}")

            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
            if choice == 0:
                muzero.train()
            elif choice == 1:
                load_model_menu(muzero, game_name)
            elif choice == 2:
                muzero.diagnose_model(30)
            elif choice == 3:
                muzero.test(render=True, opponent="self", muzero_player=None)
            elif choice == 4:
                muzero.test(render=True, opponent="human", muzero_player=0)
            elif choice == 5:
                env = muzero.Game()
                env.reset()
                env.render()

                done = False
                while not done:
                    action = env.human_to_action()
                    observation, reward, done = env.step(action)
                    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                    env.render()
            elif choice == 6:
                # Define here the parameters to tune
                # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
                muzero.terminate_workers()
                del muzero
                budget = 20
                parallel_experiments = 2
                lr_init = nevergrad.p.Log(lower=0.0001, upper=0.1)
                discount = nevergrad.p.Log(lower=0.95, upper=0.9999)
                parametrization = nevergrad.p.Dict(lr_init=lr_init, discount=discount)
                best_hyperparameters = hyperparameter_search(
                    game_name, parametrization, budget, parallel_experiments, 20
                )
                muzero = MuZero(game_name, best_hyperparameters)
            elif choice == 7:
                # Run ELO evaluation
                if EloRating is None:
                    print("Error: Could not import EloRating. Make sure elo_rating.py is in your path.")
                else:
                    num_games = int(input("Enter number of games to play per matchup: "))
                    elo, _ = muzero.evaluate_elo(num_games=num_games)
                    
                    # Plot ratings if we have a history
                    if elo and elo.history:
                        elo.plot_ratings(output_path=f"results/{game_name}/elo_ratings.png")
                        print(f"Rating plot saved to results/{game_name}/elo_ratings.png")
            else:
                break
            print("\nDone")

    ray.shutdown()
