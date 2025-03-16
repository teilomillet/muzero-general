import copy
import time

import numpy
import ray
import torch
import wandb

import models


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
            ) = self.update_weights(batch)

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

    def update_weights(self, batch):
        """
        Perform one training step.
        """

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
        ) = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        observation_batch = (
            torch.tensor(numpy.array(observation_batch)).float().to(device)
        )
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        
        # Check for NaN values in inputs
        if (torch.isnan(observation_batch).any() or torch.isnan(target_value).any() or
            torch.isnan(target_reward).any() or torch.isnan(target_policy).any()):
            print("Warning: NaN detected in inputs to model")
        
        # Convert to support form
        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(
            target_reward, self.config.support_size
        )
        
        # Ensure no NaN in targets after conversion
        if torch.isnan(target_value).any() or torch.isnan(target_reward).any():
            print("Warning: NaN detected in converted targets")
            target_value = torch.nan_to_num(target_value, nan=0.0)
            target_reward = torch.nan_to_num(target_reward, nan=0.0)
        
        ## Generate predictions
        value, reward, policy_logits, hidden_state = self.model.initial_inference(
            observation_batch
        )
        predictions = [(value, reward, policy_logits)]
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                hidden_state, action_batch[:, i]
            )
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

        ## Compute losses
        value_loss, reward_loss, policy_loss = (0, 0, 0)
        value, reward, policy_logits = predictions[0]
        # Ignore reward loss for the first batch step
        current_value_loss, _, current_policy_loss = self.loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            target_value[:, 0],
            target_reward[:, 0],
            target_policy[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        
        # Set up array for all priorities
        batch_size = target_value_scalar.shape[0]
        all_priorities = numpy.ones((batch_size, len(predictions)))
        
        try:
            # Compute priorities for the prioritized replay
            pred_value_scalar = models.support_to_scalar(value, self.config.support_size).detach().cpu().numpy()
            
            # Ensure pred_value_scalar has correct shape: [batch_size, 1] 
            if pred_value_scalar.ndim != 2 or pred_value_scalar.shape[1] != 1:
                pred_value_scalar = pred_value_scalar.reshape(batch_size, 1)
            
            # Compute absolute difference as priority
            priorities = numpy.abs(pred_value_scalar[:, 0] - target_value_scalar[:, 0])
            
            # Handle any NaN values
            priorities = numpy.where(numpy.isnan(priorities), 1.0, priorities)
            
            # Ensure positive values and apply PER alpha
            priorities = numpy.maximum(priorities, 1e-6) ** self.config.PER_alpha
            
            # Store priorities for the first prediction
            all_priorities[:, 0] = priorities
            
        except Exception as e:
            print(f"Error calculating initial priorities: {e}")
            # Use default priorities of 1.0
            all_priorities[:, 0] = 1.0

        # Process other prediction steps
        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            current_value_loss, current_reward_loss, current_policy_loss = self.loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i],
            )
            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss
            
            try:
                # Compute priorities for the prioritized replay
                pred_value_scalar = models.support_to_scalar(value, self.config.support_size).detach().cpu().numpy()
                
                # Ensure pred_value_scalar has correct shape: [batch_size, 1]
                if pred_value_scalar.ndim != 2 or pred_value_scalar.shape[1] != 1:
                    pred_value_scalar = pred_value_scalar.reshape(batch_size, 1)
                
                # Compute absolute difference as priority
                step_priorities = numpy.abs(pred_value_scalar[:, 0] - target_value_scalar[:, i])
                
                # Handle any NaN values
                step_priorities = numpy.where(numpy.isnan(step_priorities), 1.0, step_priorities)
                
                # Ensure positive values and apply PER alpha
                step_priorities = numpy.maximum(step_priorities, 1e-6) ** self.config.PER_alpha
                
                # Store priorities for this prediction step
                all_priorities[:, i] = step_priorities
                
            except Exception as e:
                print(f"Error calculating priorities for step {i}: {e}")
                # Use default priorities of 1.0
                all_priorities[:, i] = 1.0

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = value_loss * self.config.value_loss_weight + reward_loss + policy_loss
        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        total_loss = (
            value_loss * self.config.value_loss_weight
            + reward_loss
            + policy_loss
        ).mean()
        
        # Check for NaN in total loss
        if torch.isnan(total_loss):
            print("Warning: NaN detected in total_loss, skipping backward pass")
            priorities = numpy.ones_like(target_value_scalar)
            return (
                priorities,
                float('nan'),
                float('nan'),
                float('nan'),
                float('nan'),
            )
            
        # Use gradient clipping to prevent exploding gradients
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # Return max priorities across all prediction steps for each batch element
        final_priorities = numpy.max(all_priorities, axis=1)

        # Get loss values as Python floats
        total_loss_val = float(total_loss.item()) if not torch.isnan(total_loss) else float('nan')
        value_loss_val = float(value_loss.mean().item()) if not torch.isnan(value_loss.mean()) else float('nan')
        reward_loss_val = float(reward_loss.mean().item()) if not torch.isnan(reward_loss.mean()) else float('nan')
        policy_loss_val = float(policy_loss.mean().item()) if not torch.isnan(policy_loss.mean()) else float('nan')

        self.training_step += 1

        # Log metrics to wandb
        if wandb.run is not None and self.training_step % 10 == 0:  # Log every 10 steps to avoid flooding
            wandb.log({
                "trainer/total_loss": total_loss_val,
                "trainer/value_loss": value_loss_val,
                "trainer/reward_loss": reward_loss_val,
                "trainer/policy_loss": policy_loss_val,
                "trainer/learning_rate": self.optimizer.param_groups[0]["lr"],
                "trainer/training_step": self.training_step,
            }, step=self.training_step)

        return (
            final_priorities,
            total_loss_val,
            value_loss_val,
            reward_loss_val,
            policy_loss_val,
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
    ):
        # Add numerical stability checks
        # Apply small epsilon to avoid log(0) which results in NaN
        epsilon = 1e-8
        
        # Ensure policy_logits doesn't have NaN or extreme values
        if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
            policy_logits = torch.nan_to_num(policy_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
        # Ensure values don't have NaN or extreme values
        if torch.isnan(value).any() or torch.isinf(value).any():
            value = torch.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0)
            
        # Ensure rewards don't have NaN or extreme values
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Clamp values to prevent extreme logits
        policy_logits = torch.clamp(policy_logits, -10.0, 10.0)
        value = torch.clamp(value, -10.0, 10.0)
        reward = torch.clamp(reward, -10.0, 10.0)
        
        # Custom implementation of numerically stable log softmax
        def stable_log_softmax(x):
            max_x = torch.max(x, dim=1, keepdim=True)[0]
            x_stable = x - max_x
            sum_exp = torch.sum(torch.exp(x_stable), dim=1, keepdim=True)
            # Ensure sum_exp is never too small
            sum_exp = torch.clamp(sum_exp, min=epsilon)
            log_sum_exp = torch.log(sum_exp) + max_x
            return x - log_sum_exp
            
        # Apply stable log softmax
        log_softmax_value = stable_log_softmax(value)
        log_softmax_reward = stable_log_softmax(reward)
        log_softmax_policy = stable_log_softmax(policy_logits)
        
        # Ensure no -inf in the log softmax results
        log_softmax_value = torch.clamp(log_softmax_value, min=-20.0)
        log_softmax_reward = torch.clamp(log_softmax_reward, min=-20.0)
        log_softmax_policy = torch.clamp(log_softmax_policy, min=-20.0)
        
        # Calculate losses with protection against NaNs
        value_loss = torch.sum(-target_value * log_softmax_value, dim=1)
        reward_loss = torch.sum(-target_reward * log_softmax_reward, dim=1)
        policy_loss = torch.sum(-target_policy * log_softmax_policy, dim=1)
        
        # Additional check for extreme loss values
        max_loss_value = 1000.0  # Cap individual losses to prevent explosion
        value_loss = torch.clamp(value_loss, 0.0, max_loss_value)
        reward_loss = torch.clamp(reward_loss, 0.0, max_loss_value)
        policy_loss = torch.clamp(policy_loss, 0.0, max_loss_value)
        
        # Final NaN check on losses
        if torch.isnan(value_loss).any() or torch.isnan(reward_loss).any() or torch.isnan(policy_loss).any() or \
           torch.isinf(value_loss).any() or torch.isinf(reward_loss).any() or torch.isinf(policy_loss).any():
            print("Warning: NaN detected in losses after calculation!")
            value_loss = torch.nan_to_num(value_loss, nan=1.0, posinf=max_loss_value)
            reward_loss = torch.nan_to_num(reward_loss, nan=1.0, posinf=max_loss_value)
            policy_loss = torch.nan_to_num(policy_loss, nan=1.0, posinf=max_loss_value)
            
        return value_loss, reward_loss, policy_loss
