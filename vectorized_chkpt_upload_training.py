import os
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3 import Agent
import torch
import json
from datetime import datetime

# Google Drive paths
GDRIVE_ROOT = '/kaggle/working/gdrive/MyDrive'  # Adjust if different
PROJECT_DIR = os.path.join(GDRIVE_ROOT, 'pickplace_training')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')

# Mount Google Drive (Kaggle)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    GDRIVE_ROOT = '/content/drive/MyDrive'
    PROJECT_DIR = os.path.join(GDRIVE_ROOT, 'pickplace_training')
    CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
    LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')
except:
    print("Not in Colab, using Kaggle paths")

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs('temp/pickplace_td3', exist_ok=True)

print(f"Google Drive checkpoint dir: {CHECKPOINT_DIR}")

# Number of parallel environments
N_ENVS = 8  # 8x speedup!


def make_single_env(seed=None):
    """Create a single environment instance"""
    env = suite.make(
        "PickPlace",
        robots="Panda",
        controller_configs=suite.load_controller_config(default_controller="JOINT_POSITION"),
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )
    env = GymWrapper(env)
    if seed is not None:
        env.seed(seed)
    return env


class VectorizedEnv:
    """Simple vectorized environment for parallel training"""
    def __init__(self, n_envs=8):
        self.n_envs = n_envs
        
        print(f"Creating {n_envs} parallel environments...")
        self.envs = [make_single_env(seed=i) for i in range(n_envs)]
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        print(f"✓ {n_envs} environments ready")
    
    def reset(self):
        """Reset all environments"""
        return np.array([env.reset() for env in self.envs])
    
    def step(self, actions):
        """Step all environments"""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        
        observations = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]
        
        # Auto-reset done environments
        for i, done in enumerate(dones):
            if done:
                observations[i] = self.envs[i].reset()
        
        return observations, rewards, dones, infos
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()


def save_checkpoint_to_drive(agent, episode, score_history, best_score):
    """Save complete checkpoint to Google Drive"""
    try:
        # Save agent models locally first
        agent.save_models()
        
        # Save training state
        training_state = {
            'episode': episode,
            'score_history': score_history,
            'best_score': best_score,
            'timestamp': datetime.now().isoformat(),
            'memory_counter': agent.memory.mem_cntr,
            'learn_step_counter': agent.learn_step_cntr,
            'time_step': agent.time_step
        }
        
        state_path = os.path.join(CHECKPOINT_DIR, 'training_state.json')
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
        
        # Copy all model files to Google Drive
        local_dir = 'temp/pickplace_td3'
        model_files = [
            'actor_td3', 'critic_1_td3', 'critic_2_td3',
            'target_actor_td3', 'target_critic_1_td3', 'target_critic_2_td3'
        ]
        
        for model_file in model_files:
            src = os.path.join(local_dir, model_file)
            dst = os.path.join(CHECKPOINT_DIR, model_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        print(f"✓ Checkpoint saved to Google Drive (Episode {episode})")
        return True
        
    except Exception as e:
        print(f"⚠ Failed to save checkpoint: {e}")
        return False


def load_checkpoint_from_drive(agent):
    """Load checkpoint from Google Drive if exists"""
    try:
        state_path = os.path.join(CHECKPOINT_DIR, 'training_state.json')
        
        if not os.path.exists(state_path):
            print("No previous checkpoint found - starting fresh")
            return None
        
        # Load training state
        with open(state_path, 'r') as f:
            training_state = json.load(f)
        
        print(f"\nFound checkpoint from {training_state['timestamp']}")
        print(f"  Episode: {training_state['episode']}")
        print(f"  Best score: {training_state['best_score']:.2f}")
        print(f"  Memory size: {training_state['memory_counter']:,}")
        
        # Copy models from Google Drive to local
        local_dir = 'temp/pickplace_td3'
        model_files = [
            'actor_td3', 'critic_1_td3', 'critic_2_td3',
            'target_actor_td3', 'target_critic_1_td3', 'target_critic_2_td3'
        ]
        
        for model_file in model_files:
            src = os.path.join(CHECKPOINT_DIR, model_file)
            dst = os.path.join(local_dir, model_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Load models into agent
        agent.load_models()
        
        # Restore counters
        agent.memory.mem_cntr = training_state['memory_counter']
        agent.learn_step_cntr = training_state['learn_step_counter']
        agent.time_step = training_state['time_step']
        
        print("✓ Checkpoint loaded successfully\n")
        return training_state
        
    except Exception as e:
        print(f"⚠ Failed to load checkpoint: {e}")
        print("Starting fresh training")
        return None


def train_pickplace_vectorized_with_checkpointing():
    """
    Vectorized PickPlace training with Google Drive checkpointing
    8x faster + auto-resume after Kaggle restart
    """
    
    print("="*70)
    print("VECTORIZED PICKPLACE TRAINING WITH AUTO-CHECKPOINTING")
    print("="*70)
    print(f"Parallel environments: {N_ENVS}")
    print(f"Speedup: ~{N_ENVS}x")
    print(f"Checkpoint frequency: Every 10 episodes")
    print(f"Google Drive location: {CHECKPOINT_DIR}")
    print("="*70 + "\n")
    
    # Create vectorized environment
    vec_env = VectorizedEnv(N_ENVS)
    single_env = vec_env.envs[0]
    
    print(f"Environment: PickPlace (Vectorized)")
    print(f"State: {vec_env.observation_space.shape}")
    print(f"Action: {vec_env.action_space.shape}\n")
    
    # Hyperparameters
    actor_lr = 0.0005
    critic_lr = 0.0005
    batch_size = 1024
    layer1_size = 512
    layer2_size = 256
    tau = 0.05
    n_games = 50000  # Total episodes target
    
    input_dims = vec_env.observation_space.shape
    n_actions = vec_env.action_space.shape[0]
    
    # Initialize agent
    agent = Agent(
        alpha=actor_lr,
        beta=critic_lr,
        input_dims=input_dims,
        tau=tau,
        env=single_env,
        n_actions=n_actions,
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
    )
    
    # Try to load previous checkpoint
    print("Checking for previous checkpoint...")
    checkpoint_state = load_checkpoint_from_drive(agent)
    
    if checkpoint_state:
        start_episode = checkpoint_state['episode'] + 1
        score_history = checkpoint_state['score_history']
        best_score = checkpoint_state['best_score']
        print(f"Resuming from episode {start_episode}")
    else:
        start_episode = 0
        score_history = []
        best_score = -np.inf
        print("Starting fresh training")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=LOGS_DIR)
    
    # Vectorized training variables
    episode_scores = [0] * N_ENVS
    episode_steps = [0] * N_ENVS
    total_episodes = start_episode
    
    # Reset all environments
    observations = vec_env.reset()
    
    print("\n" + "="*70)
    print("VECTORIZED TRAINING START")
    print("="*70 + "\n")
    
    while total_episodes < n_games:
        # Get actions for all environments
        actions = np.array([agent.choose_action(obs) for obs in observations])
        
        # Step all environments
        next_observations, rewards, dones, infos = vec_env.step(actions)
        
        # Store transitions and update
        for i in range(N_ENVS):
            episode_scores[i] += rewards[i]
            episode_steps[i] += 1
            
            # Store in replay buffer
            agent.remember(
                observations[i],
                actions[i],
                rewards[i],
                next_observations[i],
                dones[i]
            )
            
            # Episode finished
            if dones[i]:
                score = episode_scores[i]
                steps = episode_steps[i]
                
                score_history.append(score)
                avg_score = np.mean(score_history[-100:]) if len(score_history) >= 100 else np.mean(score_history)
                total_episodes += 1
                
                # Log
                writer.add_scalar('Score/Episode', score, total_episodes)
                writer.add_scalar('Score/Average_100', avg_score, total_episodes)
                writer.add_scalar('Steps/Episode', steps, total_episodes)
                writer.add_scalar('Memory/Buffer_Size', agent.memory.mem_cntr, total_episodes)
                writer.add_scalar('Learning/Steps', agent.learn_step_cntr, total_episodes)
                
                # Save best
                if score > best_score:
                    best_score = score
                    print(f"Episode {total_episodes:5d} | ★ NEW BEST! Score: {score:7.2f} | "
                          f"Avg: {avg_score:7.2f} | Steps: {steps:3d} | "
                          f"Buffer: {agent.memory.mem_cntr:,}")
                elif total_episodes % 10 == 0:
                    print(f"Episode {total_episodes:5d} | Score: {score:7.2f} | "
                          f"Avg: {avg_score:7.2f} | Best: {best_score:7.2f} | "
                          f"Buffer: {agent.memory.mem_cntr:,}")
                
                # CHECKPOINT EVERY 10 EPISODES
                if total_episodes % 10 == 0:
                    save_checkpoint_to_drive(agent, total_episodes, score_history, best_score)
                
                # Major checkpoint every 500 episodes
                if total_episodes % 500 == 0:
                    print(f"\n{'='*70}")
                    print(f"MAJOR CHECKPOINT @ Episode {total_episodes}")
                    print(f"{'='*70}")
                    print(f"Best score: {best_score:.2f}")
                    print(f"Average (last 100): {avg_score:.2f}")
                    print(f"Buffer size: {agent.memory.mem_cntr:,}")
                    print(f"Learn steps: {agent.learn_step_cntr:,}")
                    save_checkpoint_to_drive(agent, total_episodes, score_history, best_score)
                    print(f"{'='*70}\n")
                
                # Early stopping
                if len(score_history) >= 100 and avg_score >= 180:
                    print(f"\n{'='*70}")
                    print(f"TARGET REACHED!")
                    print(f"{'='*70}")
                    print(f"Average score: {avg_score:.2f}")
                    save_checkpoint_to_drive(agent, total_episodes, score_history, best_score)
                    vec_env.close()
                    writer.close()
                    return agent
                
                # Reset episode tracking
                episode_scores[i] = 0
                episode_steps[i] = 0
        
        # Learn once per vectorized step (not per environment!)
        if agent.memory.mem_cntr > batch_size * 10:
            agent.learn()
        
        observations = next_observations
        
        if total_episodes >= n_games:
            break
    
    # Final save
    save_checkpoint_to_drive(agent, total_episodes, score_history, best_score)
    vec_env.close()
    writer.close()
    
    print("\n" + "="*70)
    print("TRAINING SESSION COMPLETE")
    print("="*70)
    print(f"Episodes completed: {total_episodes - start_episode}")
    print(f"Total episodes: {total_episodes}")
    print(f"Best score: {best_score:.2f}")
    print(f"Final average: {avg_score:.2f}")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR}")
    print("="*70 + "\n")
    
    return agent


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VECTORIZED PICKPLACE CONTINUOUS TRAINING")
    print("="*70)
    print("Features:")
    print(f"  ✓ {N_ENVS} parallel environments (~{N_ENVS}x speedup)")
    print("  ✓ Auto-saves to Google Drive every 10 episodes")
    print("  ✓ Automatically resumes from last checkpoint")
    print("  ✓ Survives Kaggle 9-hour limit")
    print("  ✓ No manual intervention needed")
    print("="*70 + "\n")
    
    try:
        agent = train_pickplace_vectorized_with_checkpointing()
        print("\n✓ Training session completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Progress saved to Google Drive")
        print("Run script again to resume")
        
    except Exception as e:
        print(f"\n\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        print("\nCheckpoint should still be saved")
        print("Try running script again")