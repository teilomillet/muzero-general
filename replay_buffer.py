import copy
import time

import numpy
import ray
import torch

import models


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed
        numpy.random.seed(self.config.seed)

    def save_game(self, game_history, shared_storage=None):
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    target_value = self.compute_target_value(game_history, i)
                    priority = (
                        numpy.abs(root_value - target_value) ** self.config.PER_alpha
                    )
                    # Check for NaN and replace with a small positive value
                    if numpy.isnan(priority):
                        priority = 1.0
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype="float32")
                # Ensure game_priority is not NaN
                game_history.game_priority = numpy.max(game_history.priorities)
                if numpy.isnan(game_history.game_priority):
                    game_history.game_priority = 1.0

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        for game_id, game_history, game_prob in self.sample_n_games(
            self.config.batch_size
        ):
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                )
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )
            if self.config.PER:
                # Add safe handling for None or NaN values
                if game_prob is None or pos_prob is None or numpy.isnan(game_prob) or numpy.isnan(pos_prob):
                    # Fall back to a default weight if probabilities are invalid
                    weight_batch.append(1.0)
                    if game_prob is None or numpy.isnan(game_prob):
                        print(f"Warning: game_prob is {'None' if game_prob is None else 'NaN'} for game {game_id}")
                    if pos_prob is None or numpy.isnan(pos_prob):
                        print(f"Warning: pos_prob is {'None' if pos_prob is None else 'NaN'} for position {game_pos} in game {game_id}")
                else:
                    weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            # Ensure weights are valid
            if len(weight_batch) == 0:
                # If we somehow have no valid weights, use uniform weights
                weight_batch = numpy.ones(len(index_batch), dtype="float32")
            else:
                # Normalize weights
                max_weight = max(weight_batch) if weight_batch else 1.0
                weight_batch = numpy.array(weight_batch, dtype="float32") / max_weight

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = numpy.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            
            # Check for NaN values in game_probs
            if numpy.isnan(game_probs).any():
                print("Warning: NaN detected in game priorities. Using uniform sampling instead.")
                force_uniform = True
            else:
                # Make sure the sum is not zero
                sum_probs = numpy.sum(game_probs)
                if sum_probs <= 0:
                    print("Warning: Sum of game priorities is zero or negative. Using uniform sampling instead.")
                    force_uniform = True
                else:
                    game_probs /= sum_probs
                    game_index = numpy.random.choice(len(self.buffer), p=game_probs)
                    game_prob = game_probs[game_index]
        
        if force_uniform or game_prob is None:
            game_index = numpy.random.choice(len(self.buffer))
            game_prob = 1.0 / len(self.buffer)  # Uniform probability
            
        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, force_uniform=False):
        if self.config.PER and not force_uniform:
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                # Ensure priorities are positive and not too close to zero
                priority = max(game_history.game_priority, 1e-8)
                # Check for NaN values
                if numpy.isnan(priority):
                    priority = 1e-8
                game_probs.append(priority)
            
            # Convert to numpy array for vectorized operations
            game_probs = numpy.array(game_probs, dtype="float32")
            
            # Check for invalid values in game_probs
            if numpy.isnan(game_probs).any() or numpy.sum(game_probs) <= 0:
                print("Warning: Invalid game priorities detected. Using uniform sampling instead.")
                force_uniform = True
            else:
                # Normalize to get probabilities
                game_probs = game_probs / numpy.sum(game_probs)
                
                # Sample indices according to priorities
                game_indices = numpy.random.choice(
                    len(game_id_list), n_games, p=game_probs, replace=True
                )
                
                return [
                    (game_id_list[index], self.buffer[game_id_list[index]], game_probs[index])
                    for index in game_indices
                ]
        
        # Uniform sampling (fallback or if PER is disabled)
        game_indices = numpy.random.choice(len(self.buffer), n_games, replace=True)
        game_id_list = list(self.buffer.keys())
        uniform_prob = 1.0 / len(self.buffer)
        
        return [
            (game_id_list[index], self.buffer[game_id_list[index]], uniform_prob)
            for index in game_indices
        ]

    def sample_position(self, game_history, force_uniform=False):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        if self.config.PER and not force_uniform and game_history.priorities is not None:
            # Add a small value to avoid zero priorities
            priorities = numpy.array(game_history.priorities, dtype="float32") + 1e-6
            
            # Check for NaN values in priorities
            if numpy.isnan(priorities).any():
                print("Warning: NaN detected in position priorities. Using uniform sampling instead.")
                force_uniform = True
            else:
                # Ensure sum is positive
                sum_priorities = numpy.sum(priorities)
                if sum_priorities <= 0:
                    print("Warning: Sum of position priorities is zero or negative. Using uniform sampling instead.")
                    force_uniform = True
                else:
                    priorities /= sum_priorities
                    position_index = numpy.random.choice(len(priorities), p=priorities)
                    position_prob = priorities[position_index]
        
        if force_uniform or position_prob is None:
            position_index = numpy.random.choice(len(game_history.root_values))
            position_prob = 1 / len(game_history.root_values)
            
        return position_index, position_prob

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Appendix Training.
        """
        # Clamp priorities to be positive and not NaN
        for i, priority in enumerate(priorities):
            # Check if any element is NaN or if any element is less than or equal to 0
            if (numpy.isnan(priority).any() if isinstance(priority, numpy.ndarray) else numpy.isnan(priority)) or \
               ((priority <= 0).any() if isinstance(priority, numpy.ndarray) else priority <= 0):
                print(f"Warning: Invalid priority detected. Using default priority.")
                priorities[i] = 1.0
            
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]
            
            # Ensure the game exists
            if game_id in self.buffer:
                # Update position priorities
                if self.buffer[game_id].priorities is None:
                    self.buffer[game_id].priorities = [1 for _ in range(len(self.buffer[game_id].root_values))]
                    
                # Ensure the position is valid
                if 0 <= game_pos < len(self.buffer[game_id].priorities):
                    # Extract scalar value from priority if it's an array
                    priority_value = priorities[i]
                    if isinstance(priority_value, numpy.ndarray):
                        if priority_value.size > 0:
                            # Take the first element if it's a non-empty array
                            priority_value = float(priority_value.flat[0])
                        else:
                            # Default value for empty arrays
                            priority_value = 1.0
                            print(f"Warning: Empty priority array for game {game_id}, position {game_pos}")
                    
                    # Ensure it's a scalar value
                    try:
                        priority_value = float(priority_value)
                        if numpy.isnan(priority_value):
                            priority_value = 1.0
                    except (TypeError, ValueError):
                        print(f"Warning: Invalid priority type for game {game_id}, position {game_pos}. Using default value.")
                        priority_value = 1.0
                        
                    self.buffer[game_id].priorities[game_pos] = priority_value
                    
                # Update game priorities
                if self.buffer[game_id].priorities is not None and len(self.buffer[game_id].priorities) > 0:
                    valid_priorities = [p for p in self.buffer[game_id].priorities if not numpy.isnan(p) and p > 0]
                    if valid_priorities:
                        self.buffer[game_id].game_priority = numpy.max(valid_priorities)
                    else:
                        self.buffer[game_id].game_priority = 1.0

    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
                if game_history.to_play_history[bootstrap_index]
                == game_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )

            value = last_step_value * self.config.discount**self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (
                reward
                if game_history.to_play_history[index]
                == game_history.to_play_history[index + i]
                else -reward
            ) * self.config.discount**i

        return value

    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            value = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(self.config.action_space))

        return target_values, target_rewards, target_policies, actions


@ray.remote
class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    See paper appendix Reanalyse.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()

        self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

    def reanalyse(self, replay_buffer, shared_storage):
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            game_id, game_history, _ = ray.get(
                replay_buffer.sample_game.remote(force_uniform=True)
            )

            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            if self.config.use_last_model_value:
                observations = numpy.array(
                    [
                        game_history.get_stacked_observations(
                            i,
                            self.config.stacked_observations,
                            len(self.config.action_space),
                        )
                        for i in range(len(game_history.root_values))
                    ]
                )

                observations = (
                    torch.tensor(observations)
                    .float()
                    .to(next(self.model.parameters()).device)
                )
                values = models.support_to_scalar(
                    self.model.initial_inference(observations)[0],
                    self.config.support_size,
                )
                game_history.reanalysed_predicted_root_values = (
                    torch.squeeze(values).detach().cpu().numpy()
                )

            replay_buffer.update_game_history.remote(game_id, game_history)
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )
