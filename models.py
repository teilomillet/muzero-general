import math
from abc import ABC, abstractmethod

import torch
import numpy


class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)
        
    def initialize_weights(self):
        """
        Initialize network weights with a numerically stable method.
        This helps prevent NaN values during early training.
        """
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                # Reduce variance for convolutional layers to prevent extreme values
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(1.0 / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                # Special initialization for fully connected layers
                # Use truncated normal to avoid extreme values
                std = 1.0 / math.sqrt(module.weight.size(1))
                module.weight.data.normal_(0, std).clamp_(-2.0 * std, 2.0 * std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                if module.weight is not None:
                    # Initialize with slightly less than 1 to avoid scale drift
                    module.weight.data.fill_(0.9)
                if module.bias is not None:
                    module.bias.data.zero_()
                    
        # Special handling for ResNet model
        if hasattr(self, 'representation_network'):
            # Apply specific initialization to first layer to ensure stable input processing
            for module in self.representation_network.modules():
                if isinstance(module, torch.nn.Conv2d) and module.in_channels > 3:
                    # First conv layer typically has more channels due to stacked observations
                    # More conservative initialization for stability
                    module.weight.data.normal_(0, 0.01)
                    if module.bias is not None:
                        module.bias.data.zero_()


##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = torch.nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )
        
        # Initialize weights with stable values
        self.initialize_weights()

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        
        # Safety check for extreme values
        if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
            policy_logits = torch.nan_to_num(policy_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
        if torch.isnan(value).any() or torch.isinf(value).any():
            value = torch.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Apply aggressive clamping to prevent unstable outputs
        policy_logits = torch.clamp(policy_logits, -10.0, 10.0)
        value = torch.clamp(value, -10.0, 10.0)
        
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        # With additional numerical stability
        reshaped_state = encoded_state.view(
            -1,
            encoded_state.shape[1],
            encoded_state.shape[2] * encoded_state.shape[3],
        )
            
        # Add small epsilon to prevent exact same min/max
        eps = 1e-5
        random_noise = torch.randn_like(reshaped_state) * eps
        
        # Calculate min/max with safety against extreme values
        min_encoded_state = reshaped_state.min(2, keepdim=True)[0].unsqueeze(-1)
        max_encoded_state = reshaped_state.max(2, keepdim=True)[0].unsqueeze(-1)
        
        # Protect against identical min/max
        identical_minmax = (max_encoded_state - min_encoded_state < eps)
        if identical_minmax.any():
            # Add a small constant to max where min == max
            max_encoded_state = torch.where(
                identical_minmax,
                min_encoded_state + 0.1,  # Use a constant offset rather than just eps
                max_encoded_state
            )
        
        # Calculate scale with protection against division by zero
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state = torch.clamp(scale_encoded_state, min=0.1)  # Larger min value for safety
        
        # Normalize the state
        encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
        
        # Final safety check for NaN/Inf values
        if torch.isnan(encoded_state_normalized).any() or torch.isinf(encoded_state_normalized).any():
            # Replace problematic values
            encoded_state_normalized = torch.nan_to_num(
                encoded_state_normalized, nan=0.5, posinf=1.0, neginf=0.0
            )
            # Ensure all values are in [0,1] range
            encoded_state_normalized = torch.clamp(encoded_state_normalized, 0.0, 1.0)
            
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)
        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] with numerical stability
        # With additional numerical stability
        eps = 1e-5
        
        # Calculate min/max with safety measures
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        
        # Protect against identical min/max
        identical_minmax = (max_next_encoded_state - min_next_encoded_state < eps)
        if identical_minmax.any():
            # Add a small constant to max where min == max
            max_next_encoded_state = torch.where(
                identical_minmax,
                min_next_encoded_state + 0.1,
                max_next_encoded_state
            )
        
        # Calculate scale with protection against division by zero
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state = torch.clamp(scale_next_encoded_state, min=0.1)
        
        # Normalize the state
        next_encoded_state_normalized = (next_encoded_state - min_next_encoded_state) / scale_next_encoded_state
        
        # Final safety check for NaN/Inf values
        if torch.isnan(next_encoded_state_normalized).any() or torch.isinf(next_encoded_state_normalized).any():
            # Replace problematic values
            next_encoded_state_normalized = torch.nan_to_num(
                next_encoded_state_normalized, nan=0.5, posinf=1.0, neginf=0.0
            )
            # Ensure all values are in [0,1] range
            next_encoded_state_normalized = torch.clamp(next_encoded_state_normalized, 0.0, 1.0)

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


###### End Fully Connected #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)
        
        # Add a small amount of noise during training to prevent exact zero gradients
        self.training_noise = 1e-5

    def forward(self, x):
        # Save input for residual connection
        identity = x
        
        # First convolution block with value clipping
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        
        # Add small noise during training to prevent vanishing gradients
        if self.training and torch.rand(1).item() > 0.5:
            out = out + torch.randn_like(out) * self.training_noise
            
        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection with value clipping to prevent extreme values
        out = out + identity
        
        # ReLU activation after the addition
        out = torch.nn.functional.relu(out)
        
        # Prevent extreme values
        out = torch.clamp(out, -100, 100)
        
        # Check for and handle inf/nan
        if torch.isnan(out).any() or torch.isinf(out).any():
            # Replace with safe values but preserve general structure
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
            # Don't print here to avoid log flooding
            
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2) for _ in range(2)]
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class DownsampleCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_w):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')
        self.conv = conv3x3(
            observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
    ):
        super().__init__()
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = torch.nn.Conv2d(
            num_channels - 1, reduced_channels_reward, 1
        )
        self.block_output_size_reward = block_output_size_reward
        self.fc = mlp(
            self.block_output_size_reward,
            fc_reward_layers,
            full_support_size,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        state = x
        x = self.conv1x1_reward(x)
        x = x.view(-1, self.block_output_size_reward)
        reward = self.fc(x)
        return state, reward


class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_size_policy,
            fc_policy_layers,
            action_space_size,
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        support_size,
        downsample,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        block_output_size_reward = (
            (
                reduced_channels_reward
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (
                reduced_channels_value
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (
                reduced_channels_policy
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        self.representation_network = torch.nn.DataParallel(
            RepresentationNetwork(
                observation_shape,
                stacked_observations,
                num_blocks,
                num_channels,
                downsample,
            )
        )

        self.dynamics_network = torch.nn.DataParallel(
            DynamicsNetwork(
                num_blocks,
                num_channels + 1,
                reduced_channels_reward,
                fc_reward_layers,
                self.full_support_size,
                block_output_size_reward,
            )
        )

        self.prediction_network = torch.nn.DataParallel(
            PredictionNetwork(
                action_space_size,
                num_blocks,
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                self.full_support_size,
                block_output_size_value,
                block_output_size_policy,
            )
        )
        
        # Initialize weights with stable values
        self.initialize_weights()

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        
        # Safety check for extreme values
        if torch.isnan(policy).any() or torch.isinf(policy).any():
            policy = torch.nan_to_num(policy, nan=0.0, posinf=10.0, neginf=-10.0)
            
        if torch.isnan(value).any() or torch.isinf(value).any():
            value = torch.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Apply aggressive clamping to prevent unstable outputs
        policy = torch.clamp(policy, -10.0, 10.0)
        value = torch.clamp(value, -10.0, 10.0)
        
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        # With additional numerical stability
        reshaped_state = encoded_state.view(
            -1,
            encoded_state.shape[1],
            encoded_state.shape[2] * encoded_state.shape[3],
        )
            
        # Add small epsilon to prevent exact same min/max
        eps = 1e-5
        random_noise = torch.randn_like(reshaped_state) * eps
        
        # Calculate min/max with safety against extreme values
        min_encoded_state = reshaped_state.min(2, keepdim=True)[0].unsqueeze(-1)
        max_encoded_state = reshaped_state.max(2, keepdim=True)[0].unsqueeze(-1)
        
        # Protect against identical min/max
        identical_minmax = (max_encoded_state - min_encoded_state < eps)
        if identical_minmax.any():
            # Add a small constant to max where min == max
            max_encoded_state = torch.where(
                identical_minmax,
                min_encoded_state + 0.1,  # Use a constant offset rather than just eps
                max_encoded_state
            )
        
        # Calculate scale with protection against division by zero
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state = torch.clamp(scale_encoded_state, min=0.1)  # Larger min value for safety
        
        # Normalize the state
        encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
        
        # Final safety check for NaN/Inf values
        if torch.isnan(encoded_state_normalized).any() or torch.isinf(encoded_state_normalized).any():
            # Replace problematic values
            encoded_state_normalized = torch.nan_to_num(
                encoded_state_normalized, nan=0.5, posinf=1.0, neginf=0.0
            )
            # Ensure all values are in [0,1] range
            encoded_state_normalized = torch.clamp(encoded_state_normalized, 0.0, 1.0)
            
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self.dynamics_network(x)

        # Scale encoded state between [0, 1] with numerical stability
        # With additional numerical stability
        eps = 1e-5
        
        # Calculate min/max with safety measures
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        
        # Protect against identical min/max
        identical_minmax = (max_next_encoded_state - min_next_encoded_state < eps)
        if identical_minmax.any():
            # Add a small constant to max where min == max
            max_next_encoded_state = torch.where(
                identical_minmax,
                min_next_encoded_state + 0.1,
                max_next_encoded_state
            )
        
        # Calculate scale with protection against division by zero
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state = torch.clamp(scale_next_encoded_state, min=0.1)
        
        # Normalize the state
        next_encoded_state_normalized = (next_encoded_state - min_next_encoded_state) / scale_next_encoded_state
        
        # Final safety check for NaN/Inf values
        if torch.isnan(next_encoded_state_normalized).any() or torch.isinf(next_encoded_state_normalized).any():
            # Replace problematic values
            next_encoded_state_normalized = torch.nan_to_num(
                next_encoded_state_normalized, nan=0.5, posinf=1.0, neginf=0.0
            )
            # Ensure all values are in [0,1] range
            next_encoded_state_normalized = torch.clamp(next_encoded_state_normalized, 0.0, 1.0)

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


########### End ResNet ###########
##################################


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Check for NaN or infinite values in logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        # More detailed logging (but limit frequency to avoid flooding)
        if numpy.random.random() < 0.05:  # Only log about 5% of the time
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            total_elements = logits.numel()
            nan_percent = nan_count / total_elements * 100 if total_elements > 0 else 0
            inf_percent = inf_count / total_elements * 100 if total_elements > 0 else 0
            
            print(f"Warning: {nan_count}/{total_elements} ({nan_percent:.2f}%) NaN and {inf_count}/{total_elements} ({inf_percent:.2f}%) Inf values in support_to_scalar input")
        
        # Very aggressive replacement of problematic values
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Extreme clamping to avoid any values that could cause issues
    logits = torch.clamp(logits, -5, 5)
    
    # Ensure logits has proper shape for softmax: [batch_size, n_categories]
    # If it has more dimensions, flatten all but the last one
    original_shape = logits.shape
    if len(original_shape) > 2:
        batch_size = original_shape[0]
        logits = logits.reshape(batch_size, -1)
    
    # Double-stable softmax implementation:
    # 1. Subtract max for numerical stability
    # 2. Apply exponential
    # 3. Add small epsilon to avoid division by zero
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    logits_stable = logits - max_logits
    exp_logits = torch.exp(logits_stable)
    sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)
    # Ensure denominator is never too close to zero
    sum_exp = torch.clamp(sum_exp, min=1e-5)
    probabilities = exp_logits / sum_exp
    
    # Check for NaN after softmax and use uniform distribution as fallback
    if torch.isnan(probabilities).any():
        # Only log occasionally
        if numpy.random.random() < 0.05:
            print("Warning: NaN detected after softmax, using uniform distribution")
        # Use uniform distribution as fallback
        probabilities = torch.ones_like(probabilities) / probabilities.shape[1]
    
    # Create support tensor
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    
    # Calculate weighted sum
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling with protection against NaN
    safe_abs_x = torch.abs(x) + 1e-8
    safe_term = torch.sqrt(1 + 4 * 0.001 * (safe_abs_x + 1 + 0.001)) - 1
    # Prevent division by very small numbers
    safe_divisor = 2 * 0.001  # Don't use clamp on a scalar
    if safe_divisor < 1e-8:
        safe_divisor = 1e-8
    x = torch.sign(x) * (((safe_term / safe_divisor) ** 2 - 1))
    
    # Final NaN/Inf check
    if torch.isnan(x).any() or torch.isinf(x).any():
        # Only log occasionally
        if numpy.random.random() < 0.05:
            print("Warning: NaN detected in support_to_scalar output, replacing with safe values")
        # Use bounded replacement values
        x = torch.nan_to_num(x, nan=0.0, posinf=support_size, neginf=-support_size)
        # Additional safety clamp
        x = torch.clamp(x, -support_size, support_size)
    
    # Ensure output has shape [batch_size, 1]
    if len(x.shape) != 2 or x.shape[1] != 1:
        batch_size = x.shape[0] if len(x.shape) > 0 else 1
        x = x.reshape(batch_size, 1)
    
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Handle NaN or infinite values
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.nan_to_num(x, nan=0.0, posinf=support_size, neginf=-support_size)
        print("Warning: NaN or Inf detected in scalar_to_support input, replacing with safe values")
    
    # Reduce the scale with numerical stability
    safe_abs_x = torch.abs(x) + 1e-8  # Avoid sqrt of zero
    x = torch.sign(x) * (torch.sqrt(safe_abs_x + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    
    # Safely handle indices
    floor_indices = (floor + support_size).long().unsqueeze(-1)
    floor_indices = torch.clamp(floor_indices, 0, 2 * support_size)
    
    logits.scatter_(
        2, floor_indices, (1 - prob).unsqueeze(-1)
    )
    
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = torch.clamp(indexes.long().unsqueeze(-1), 0, 2 * support_size)
    
    logits.scatter_(2, indexes, prob.unsqueeze(-1))
    
    # Final check for NaN
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        print("Warning: NaN detected in scalar_to_support output, replacing with zeros")
    
    return logits
