"""PPO Training Script - Drone Racing

Based on CleanRL PPO implementation, using custom observation and reward wrappers.

Usage:
    # Training
    python ppo_racing.py --train True --wandb_enabled True

    # Evaluation (no rendering, using default best_model.ckpt)
    python ppo_racing.py --train False --eval 5

    # Evaluation (with rendering)
    python ppo_racing.py --train False --eval 5 -r True

    # Evaluation (specify checkpoint path)
    python ppo_racing.py --train False --eval 5 --ckpt_path ./checkpoints/ppo_racing.ckpt

    # Evaluation (specify checkpoint + rendering)
    python ppo_racing.py --train False --eval 5 -r True --ckpt_path ./checkpoints/ppo_racing.ckpt

    # WandB Sweep
    wandb sweep sweep.yaml
    wandb agent <sweep_id>

    # Load best parameters from WandB downloaded config.yaml
    python ppo_racing.py --load_config_from ./config.yaml --wandb_enabled True

    # Load config and override some parameters
    python ppo_racing.py --load_config_from ./config.yaml --total_timesteps 5000000
"""

from __future__ import annotations

import random
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch import Tensor
from torch.distributions.normal import Normal

# Environment related
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.utils import load_config
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch

# Custom Wrappers
# from lsy_drone_racing.rl_training.wrappers.observation import RacingObservationWrapper
from lsy_drone_racing.rl_training.wrappers.observation import RacingObservationWrapper

from lsy_drone_racing.rl_training.wrappers.reward import RacingRewardWrapper as BaseRewardWrapper
# from lsy_drone_racing.rl_training.wrappers.reward_racing_lv1 import RacingRewardWrapper as RacingRewardWrapperLv1
from lsy_drone_racing.rl_training.wrappers.reward_racing_progress_clip import RacingRewardWrapper as RacingRewardWrapperLv1


# ============================================================================
# Configuration Parameters
# ============================================================================

@dataclass
class Args:
    """Training configuration parameters.
    
    Can be overridden via command line or WandB Sweep.
    """
    
    # ---------- Basic Settings ----------
    seed: int = 42
    """Random seed"""
    torch_deterministic: bool = True
    """Whether to use deterministic CUDA operations"""
    cuda: bool = True
    """Whether to use CUDA"""
    jax_device: str = "gpu"
    """JAX environment device (cpu/gpu)"""
    
    # ---------- WandB Settings ----------
    wandb_project_name: str = "DroneRacing-PPO"
    """WandB project name"""
    wandb_entity: str = None
    """WandB team/username"""
    
# ---------- Environment Configuration ----------
    config_file: str = "level2.toml"

    num_envs: int = 1  
    
    # ---------- PPO Hyperparameters (Racing Tuned Version) ----------
    
    # Total training steps
    # Racing requires fine-tuned trajectories, usually needs more steps
    total_timesteps: int = 5_000_000  
    
    # Learning rate
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    
    # Rollout Length
    num_steps: int = 128  
    
    # Batch calculation
    # Batch Size = 64 * 128 = 8192
    # Minibatch Size = 8192 / 4 = 2048
    num_minibatches: int = 4
    update_epochs: int = 10
    
    # Entropy Coefficient (Entropy Coef)
    # Racing tasks (especially overfitting) require deterministic policies.
    # 0.01 is suitable for initial exploration. If convergence is too slow later, can reduce to 0.001 or 0.0
    ent_coef: float = 0.01  
    
    # Other standard parameters (keep defaults)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    norm_adv: bool = True
    
    # Network structure
    hidden_dim: int = 256
    """Hidden layer dimension"""
    
# ---------- Reward Coefficients (Adapted for RacingRewardWrapperLv0) ----------
    coef_progress: float = 20.0   
    coef_gate: float = 10.0       
    coef_finish: float = 50.0     
    coef_time: float = 0.05       
    coef_align: float = 0.5       
    coef_collision: float = 10.0  
    coef_smooth: float = 0.1      
    coef_spin: float = 0.1        
    coef_angle: float = 0.02
    stage: int = 1
    n_history: int = 2
    """State stacking count"""
    
    # ---------- Runtime Calculations ----------
    batch_size: int = 0
    """Batch size (computed at runtime)"""
    minibatch_size: int = 0
    """Minibatch size (computed at runtime)"""
    num_iterations: int = 0
    """Number of iterations (computed at runtime)"""
    
    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create and initialize Args instance."""
        #num_iterations = total_timesteps // batch_size = total_timesteps // (num_envs * num_steps)
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        return args


# ============================================================================
# Utility Functions
# ============================================================================

def set_seeds(seed: int):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal initialization for network layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ============================================================================
# Environment Creation
# ============================================================================

def make_env(
    args: Args,
    jax_device: str = "cpu",
    torch_device: torch.device = torch.device("cpu"),
) -> gym.vector.VectorEnv:
    """Create training environment.
    
    Wrapper chain:
        VecDroneRaceEnv
        → NormalizeActions (action normalization [-1,1] → actual range)
        → RacingRewardWrapper (compute dense reward)
        → RacingObservationWrapper (observation transformation: dict → 88D vector)
        → JaxToTorch (JAX Array → PyTorch Tensor)
    
    Args:
        args: Training configuration
        jax_device: JAX device
        torch_device: PyTorch device
        
    Returns:
        Wrapped vectorized environment
    """
    # Load configuration file
    config_path = Path(__file__).parents[2] / "config" / args.config_file
    config = load_config(config_path)
    
    # Automatically read gate and obstacle counts from config file
    n_gates = len(config.env.track.gates)
    n_obstacles = len(config.env.track.get("obstacles", []))
    print(f"[make_env] Config: {args.config_file}, Gates: {n_gates}, Obstacles: {n_obstacles}")
    
    # Create base environment
    env = VecDroneRaceEnv(
        num_envs=args.num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        disturbances=config.env.get("disturbances", None),
        randomizations=config.env.get("randomizations", None),
        seed=args.seed,
        max_episode_steps=1500,
        device=jax_device,
    )
    
    # 1. Action normalization: Map network output [-1, 1] to actual action range
    env = NormalizeActions(env)
    
    # 2. Reward wrapper (needs original obs dict)
    env = RacingRewardWrapperLv1(  # Ensure class name matches your import
        env,
        n_gates=n_gates,           # Explicitly pass n_gates        
        # Pass all coefficients
        coef_progress=args.coef_progress,
        coef_gate=args.coef_gate,
        coef_finish=args.coef_finish,     
        coef_time=args.coef_time,         
        coef_collision=args.coef_collision,
        coef_smooth=args.coef_smooth,
        coef_spin=args.coef_spin,
        coef_angle=args.coef_angle,
    )
    

    # 3. Observation wrapper (convert dict to vector)
    env = RacingObservationWrapper(
        env,
        n_gates=n_gates,
        n_obstacles=n_obstacles,
        stage=args.stage,  # Stage 1: Mask obstacles | Stage 2: Show obstacles
        n_history=args.n_history,  
    )
    
    # 4. Data type conversion: JAX Array → PyTorch Tensor
    env = JaxToTorch(env, torch_device)
    
    return env


# ============================================================================
# Neural Network
# ============================================================================

class Agent(nn.Module):
    """PPO Agent network.
    
    Actor-Critic structure:
    - Actor: Outputs action mean and standard deviation
    - Critic: Outputs state value
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize network.
        
        Args:
            obs_dim: Observation dimension (88)
            action_dim: Action dimension (4)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        
        # Actor network (outputs action mean)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
            nn.Tanh(),  # Output range [-1, 1]
        )
        
        # ============================================================
        # [Key Modification] Thrust Bias Initialization
        # ============================================================
        # We need to modify the bias of the last Linear layer in actor_mean
        # Structure: [Linear, Tanh, Linear, Tanh, Linear(index-2), Tanh(index-1)]
        with torch.no_grad():
            last_layer = self.actor_mean[-2] # Get the last Linear layer
            
            # 1. Ensure attitude channels (0,1,2) bias to 0 (horizontal)
            # Here 0 corresponds to the middle value after NormalizeActions mapping
            last_layer.bias[0] = 0.0  # Roll
            last_layer.bias[1] = 0.0  # Pitch
            last_layer.bias[2] = 0.0  # Yaw
            
            # 2. Add bias to thrust channel (3)
            last_layer.bias[3] = 1.0

        # Action standard deviation (learnable parameter)
        # Adaptively initialize based on action dimension
        if action_dim == 4:
            # attitude mode: [roll, pitch, yaw, thrust]
            init_logstd = torch.tensor([[-1.0, -1.0, -1.0, -0.5]])
        self.actor_logstd = nn.Parameter(init_logstd)
    
    def get_value(self, x: Tensor) -> Tensor:
        """Get state value."""
        return self.critic(x)
    
    def get_action_and_value(
        self, 
        x: Tensor, 
        action: Tensor | None = None, 
        deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get action and value.
        
        Args:
            x: Observation
            action: Existing action (for computing log_prob)
            deterministic: Whether to use deterministic action
            
        Returns:
            action: Action
            log_prob: Log probability of action
            entropy: Policy entropy
            value: State value
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = action_mean if deterministic else probs.sample()
        
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


# ============================================================================
# Training Function
# ============================================================================

def train_ppo(
    args: Args,
    model_path: Path,
    device: torch.device,
    jax_device: str,
    wandb_enabled: bool = False,
) -> list[float]:
    """PPO training main loop.
    
    Based on CleanRL implementation: https://docs.cleanrl.dev/
    
    Args:
        args: Training configuration
        model_path: Path to save model
        device: PyTorch device
        jax_device: JAX device
        wandb_enabled: Whether to enable WandB
        
    Returns:
        Episode reward history during training
    """
    # ========== Initialization ==========
    if wandb_enabled and wandb.run is None:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
    
    train_start_time = time.time()
    set_seeds(args.seed)
    print(f"Training on device: {device} | Environment device: {jax_device}")
    print(f"Config: {args.config_file}")

        # ========== Save Configuration ==========
    config_save_path = model_path.parent.parent / "train_args" / "train_args.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Config saved to {config_save_path}")

    # ========== Create Environment ==========
    envs = make_env(args, jax_device=jax_device, torch_device=device)
    
    obs_dim = envs.single_observation_space.shape[0]  # 88
    action_dim = envs.single_action_space.shape[0]    # 4
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # ========== Create Agent ==========
    agent = Agent(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # ========== Storage Buffers ==========
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # ================= Initialize save variables =================
    best_reward = -float('inf')
    best_model_path = model_path.parent / "best_model.ckpt"
    print(f"Training started. Press Ctrl+C to safely stop and save to: {model_path}")
    print(f"Best model will be saved under: {best_model_path}")
    # ===========================================================
    
    # ========== Start Training ==========
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = next_obs.to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # Statistics tracking
    sum_rewards = torch.zeros(args.num_envs).to(device)
    sum_rewards_hist = []
    len_hist = []
    episode_count = 0
    
    try:
        for iteration in range(1, args.num_iterations + 1):
            iter_start_time = time.time()
            
            # Learning rate annealing
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            
            # ========== Collect Data ==========
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                
                # Sample action
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                
                # Execute action (JaxToTorch wrapper handles tensor conversion)
                next_obs, reward, terminations, truncations, infos = envs.step(action)
                next_obs = next_obs.to(device)
                reward = reward.to(device)
                
                rewards[step] = reward
                sum_rewards += reward
                
                # Handle episode end
                next_done = (terminations | truncations).float().to(device)
                
                if next_done.any():
                    finished_rewards = sum_rewards[next_done.bool()]
                    for r in finished_rewards:
                        sum_rewards_hist.append(r.item())
                        episode_count += 1
                        if wandb_enabled:
                            wandb.log({"train/episode_reward": r.item()}, step=global_step)
                    sum_rewards[next_done.bool()] = 0
            
            # ========== Compute GAE ==========
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                
                returns = advantages + values
            
            # ========== Flatten Data ==========
            b_obs = obs.reshape((-1, obs_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, action_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            
            # ========== PPO Update ==========
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                # Early stopping
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            
            # ========== Logging ==========
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            iter_time = time.time() - iter_start_time
            
            if wandb_enabled:
                wandb.log({
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/SPS": int(args.num_envs * args.num_steps / iter_time),
                    "charts/episode_count": episode_count,
                    "losses/policy_loss": pg_loss.item(),
                    "losses/value_loss": v_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                }, step=global_step)
            

            # Print progress
            if iteration % 10 == 0 or iteration == 1:
                avg_reward = np.mean(sum_rewards_hist[-100:]) if sum_rewards_hist else 0
                # Calculate average length
            
                print(f"Iter {iteration}/{args.num_iterations} | Steps: {global_step/1e6:.2f}M/{args.total_timesteps/1e6:.2f}M")
                
                # Print more detailed debug information
                # 1. Task performance: Reward and survival time
                print(f"  [Perf] Avg Rew: {avg_reward:.2f}")
                
                # 2. Network health:
                #    Ent (Policy entropy): Represents exploration desire. If quickly drops to 0, means premature convergence (overfitted)
                #    KL  (Update magnitude): Should be around 0.01. Too large means learning rate too high, too small means not learning
                print(f"  [Loss] Val: {v_loss.item():.4f} | Pol: {pg_loss.item():.4f} | "
                    f"Ent: {entropy_loss.item():.4f} | KL: {approx_kl.item():.4f}")
                
                print(f"  [Time] {iter_time:.2f}s | SPS: {int(args.num_envs * args.num_steps / iter_time)}")
                print("-" * 50)
                # ================= Auto-save Logic =================
                # 1. Save latest model every time logging (overwrite)
                if model_path is not None:
                    torch.save(agent.state_dict(), model_path)
                
                # 2. If reward reaches new high, save an extra "best_model"
                if avg_reward > best_reward and iteration > 10:  # First 10 iterations unstable, don't save
                    best_reward = avg_reward
                    torch.save(agent.state_dict(), best_model_path)
                    print(f"  [★] New record! Best model saved (Rew: {best_reward:.2f})")
                # =========================================================
    # ========== Capture Interrupt ==========
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user (Ctrl+C)!")
        if model_path is not None:
            torch.save(agent.state_dict(), model_path)
            print(f"  [SAFETY] Model emergency saved to: {model_path}")
        print("Closing environment...")
        envs.close()
        return sum_rewards_hist
    # ===================================
    # ========== Save Model ==========
    train_time = time.time() - train_start_time
    # Convert to minutes and seconds
    m, s = divmod(int(train_time), 60)
    
    # Print format: "Training completed in 28m 52s ..."
    print(f"\nTraining completed in {m}m {s}s ({global_step:,} steps)")
    
    if model_path is not None:
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    envs.close()
    return sum_rewards_hist


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_ppo(
    args: Args,
    n_eval: int,
    model_path: Path,
    render: bool = False,
) -> tuple[list[float], list[int]]:
    """Evaluation function with success rate tracking based on track completion."""
    set_seeds(args.seed)
    # Force CPU for evaluation to avoid tensor device errors
    device = torch.device("cpu")

    print(f"\n[Eval] Loading model: {model_path}")
    print(f"[Eval] Render mode: {render}")
    print(f"[Eval] Success criterion: All gates passed (target_gate == -1)")

    # 1. Create single environment (num_envs=1)
    args_eval = Args.create(**{**vars(args), "num_envs": 1})
    eval_env = make_env(args_eval, jax_device="cpu", torch_device=device)
    
    obs_dim = eval_env.single_observation_space.shape[0]
    action_dim = eval_env.single_action_space.shape[0]
    
    # 2. Load model
    agent = Agent(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            agent.load_state_dict(checkpoint["model_state_dict"])
        else:
            agent.load_state_dict(checkpoint)
        print("[Eval] Model loaded successfully!")
    except Exception as e:
        print(f"[Eval] Model loading failed: {e}")
        return [], []

    agent.eval()
    
    episode_rewards = []
    episode_lengths = []
    successes = []  # Track success for each episode

    with torch.no_grad():
        for episode in range(n_eval):
            print(f"\n=== Episode {episode + 1} Start ===")
            obs, _ = eval_env.reset(seed=args.seed + episode)
            obs = obs.to(device)

            episode_reward = 0
            steps = 0
            done = False
            finished_track = False  # Track if all gates were passed
            last_target_gate = -1  # Track the last gate index to detect gate passing

            while not done:
                action, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
                action = torch.clamp(action, -1.0, 1.0)

                # Execute action
                obs, reward, terminated, truncated, info = eval_env.step(action)
                obs = obs.to(device)
                
                # Extract velocity from observation vector
                # Observation structure: [pos_z(1), vel_body(3), ang_vel(3), rot_mat(9), ...]
                # Linear velocity is at indices 1-3 (x, y, z in body frame)
                vel = obs[0, 1:4].cpu().numpy()  # Linear velocity vector [vx, vy, vz] in body frame
                vel_magnitude = np.linalg.norm(vel)  # Magnitude (speed)

                if render:
                    if steps % 10 == 0:
                        print(f"Step {steps}: magnitude={vel_magnitude:.3f} m/s")
                    eval_env.unwrapped.render()
                    time.sleep(0.02)

                episode_reward += reward[0].item()
                steps += 1

                # Check termination
                term = terminated[0].item()
                trunc = truncated[0].item()
                done = term or trunc

                if done:
                    print(f"\n{'='*40}")
                    print(f"Episode End at Step {steps}")
                    print(f"Reason: Terminated={term}, Truncated={trunc}")

                    # Check finish reward as another indicator
                    reward_wrapper = eval_env.env.env
                    finish_reward = 0.0
                    if len(reward_wrapper._finished_episodes.get("finish", [])) > 0:
                        finish_reward = reward_wrapper._finished_episodes["finish"][-1]
                        # If finish reward > 0, track was completed
                        if finish_reward > 0:
                            finished_track = True

                    print(f"Track Finished: {finished_track} (finish_reward={finish_reward:.2f})")

                    print(f"\n--- Reward Breakdown ---")
                    for key, values in reward_wrapper._finished_episodes.items():
                        if len(values) > 0:
                            print(f"  {key:12s}: {values[-1]:.4f}")  # Get the last completed episode

                    print(f"\n--- Summary ---")
                    print(f"  Total (accumulated): {episode_reward:.4f}")
                    print(f"  Steps: {steps}")
                    print(f"{'='*40}\n")

            # Success based on track completion (all gates passed)
            success = finished_track
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            successes.append(success)
            
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps} - {status}")
    
    # Calculate statistics
    num_successes = sum(successes)
    success_rate = (num_successes / n_eval) * 100

    print("\n" + "="*60)
    print(f"Success: {num_successes}/{n_eval}, Success Rate: {success_rate:.1f}%")
    print(f"Average Reward (All): {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length (All): {np.mean(episode_lengths):.1f}")

    if num_successes > 0:
        successful_rewards = [r for r, s in zip(episode_rewards, successes) if s]
        successful_lengths = [l for l, s in zip(episode_lengths, successes) if s]
        print(f"Average Reward (Successful): {np.mean(successful_rewards):.2f} ± {np.std(successful_rewards):.2f}")
        print(f"Average Episode Length (Successful): {np.mean(successful_lengths):.1f} ± {np.std(successful_lengths):.1f}")
    else:
        print("No successful episodes to calculate statistics.")
    print("="*60)
    
    eval_env.close()
    return episode_rewards, episode_lengths
# ============================================================================
# Main Function
# ============================================================================

def load_wandb_config(config_path: str | Path) -> dict:
    """Load parameters from WandB downloaded config.yaml.
    
    WandB config.yaml format may be:
        learning_rate:
          value: 0.001
    or:
        learning_rate: 0.001
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Flattened parameter dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten: if value is dict with 'value' key, extract it
    flat_config = {}
    
    # WandB internal fields to skip
    skip_keys = {'wandb_version', '_wandb', 'wandb_enabled', 'train', 'eval'}
    
    for key, val in config.items():
        if key in skip_keys:
            continue
        if isinstance(val, dict) and 'value' in val:
            flat_config[key] = val['value']
        else:
            flat_config[key] = val
    
    print(f"[load_wandb_config] Loading parameters from {config_path} :")
    for k, v in flat_config.items():
        print(f"  {k}: {v}")
    
    return flat_config


def main(
    wandb_enabled: bool = False,
    train: bool = True,
    eval: int = 0,
    r: bool = False,
    ckpt_path: str = None,
    config_file: str = None,
    load_config_from: str = None,
    **kwargs,
):
    """Main entry point.

    Args:
        wandb_enabled: Whether to enable WandB
        train: Whether to train
        eval: Number of evaluation episodes (0 means no evaluation)
        r: Whether to enable rendering (only valid during evaluation)
        ckpt_path: Model path to load during evaluation (default: best_model.ckpt)
        config_file: Environment configuration file
        load_config_from: Load parameters from WandB config.yaml (optional)
        **kwargs: Parameters to override default Args
    """
    # If wandb config file is specified, load it first
    if load_config_from is not None:
        wandb_config = load_wandb_config(load_config_from)
        # wandb config has lower priority than command line arguments
        for key, val in wandb_config.items():
            if key not in kwargs:
                kwargs[key] = val
    
    # Create configuration
    # kwargs["config_file"] = config_file
    args = Args.create(**kwargs)
    
    # Path settings
    model_save_path = Path(__file__).parent /"checkpoints" / "ppo_racing.ckpt"

    # If custom checkpoint path is specified, use it; otherwise use default best_model.ckpt
    if ckpt_path is not None:
        model_eval_path = Path(ckpt_path)
    else:
        model_eval_path = Path(__file__).parent /"checkpoints" / "best_model.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    jax_device = args.jax_device

    # Training
    if train:
        train_ppo(args, model_save_path, device, jax_device, wandb_enabled)

    # Evaluation
    if eval > 0:
        episode_rewards, episode_lengths = evaluate_ppo(args, eval, model_eval_path, render=r)

        if wandb_enabled and wandb.run is not None:
            wandb.log({
                "eval/mean_reward": np.mean(episode_rewards),
                "eval/std_reward": np.std(episode_rewards),
                "eval/mean_length": np.mean(episode_lengths),
            })
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)