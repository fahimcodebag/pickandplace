#!/usr/bin/env python3
"""
Debug Vectorized Training Issues
Diagnose why learning is stuck
"""

import numpy as np
import torch
import os
import json

def check_replay_buffer(agent):
    """Check if replay buffer has enough data"""
    print("="*70)
    print("1. REPLAY BUFFER CHECK")
    print("="*70)
    
    mem_size = agent.memory.mem_cntr
    mem_capacity = agent.memory.mem_size
    batch_size = agent.batch_size
    min_needed = batch_size * 10
    
    print(f"Buffer size: {mem_size:,} / {mem_capacity:,}")
    print(f"Batch size: {batch_size}")
    print(f"Min needed for learning: {min_needed:,}")
    
    if mem_size < min_needed:
        print(f"\n❌ NOT ENOUGH DATA TO START LEARNING!")
        print(f"   Have: {mem_size:,}")
        print(f"   Need: {min_needed:,}")
        print(f"   Deficit: {min_needed - mem_size:,}")
        print(f"\n   Agent is still in warmup phase!")
        return False
    else:
        print(f"\n✓ Buffer has enough data")
        print(f"   Learning should be active")
        return True


def check_learning_activity(agent):
    """Check if learning is actually happening"""
    print("\n" + "="*70)
    print("2. LEARNING ACTIVITY CHECK")
    print("="*70)
    
    learn_steps = agent.learn_step_cntr
    time_steps = agent.time_step
    
    print(f"Total time steps: {time_steps:,}")
    print(f"Learning steps: {learn_steps:,}")
    
    if learn_steps == 0:
        print(f"\n❌ NO LEARNING HAS OCCURRED!")
        print(f"   Possible reasons:")
        print(f"   - Buffer not full enough")
        print(f"   - learn() not being called")
        print(f"   - Stuck in warmup phase")
        return False
    else:
        ratio = learn_steps / max(time_steps, 1)
        print(f"\n✓ Learning is active")
        print(f"   Learning ratio: {ratio:.4f}")
        print(f"   (Should be ~0.5 for update_actor_interval=2)")
        return True


def check_network_outputs(agent, test_states):
    """Check if networks are producing reasonable outputs"""
    print("\n" + "="*70)
    print("3. NETWORK OUTPUT CHECK")
    print("="*70)
    
    agent.actor.eval()
    agent.critic_1.eval()
    
    with torch.no_grad():
        # Test actor
        state_tensor = torch.FloatTensor(test_states).to(agent.actor.device)
        actions = agent.actor(state_tensor)
        
        print(f"Actor output:")
        print(f"  Shape: {actions.shape}")
        print(f"  Range: [{actions.min():.4f}, {actions.max():.4f}]")
        print(f"  Mean: {actions.mean():.4f}")
        print(f"  Std: {actions.std():.4f}")
        
        # Check if stuck at zero
        if actions.abs().max() < 0.01:
            print(f"\n❌ ACTIONS NEAR ZERO!")
            print(f"   Network might not be learning")
            print(f"   Or learning rate too low")
            return False
        
        # Test critic
        q1_values = agent.critic_1(state_tensor, actions)
        
        print(f"\nCritic output:")
        print(f"  Q-values range: [{q1_values.min():.4f}, {q1_values.max():.4f}]")
        print(f"  Q-values mean: {q1_values.mean():.4f}")
        
        if q1_values.abs().max() < 0.01:
            print(f"\n⚠ Q-VALUES NEAR ZERO")
            print(f"   Critic might not be trained yet")
        else:
            print(f"\n✓ Networks producing reasonable outputs")
            return True


def check_exploration(agent):
    """Check exploration parameters"""
    print("\n" + "="*70)
    print("4. EXPLORATION CHECK")
    print("="*70)
    
    warmup = agent.warmup
    time_step = agent.time_step
    noise = agent.noise
    
    print(f"Warmup period: {warmup:,} steps")
    print(f"Current step: {time_step:,}")
    print(f"Noise level: {noise}")
    
    if time_step < warmup:
        print(f"\n⚠ STILL IN WARMUP!")
        print(f"   Progress: {time_step/warmup*100:.1f}%")
        print(f"   Actions are random")
        print(f"   {warmup - time_step:,} steps remaining")
        return False
    else:
        print(f"\n✓ Past warmup phase")
        print(f"   Using trained policy + noise")
        return True


def analyze_training_log(log_dir='logs/PickPlace_vectorized'):
    """Analyze TensorBoard logs"""
    print("\n" + "="*70)
    print("5. TRAINING LOG ANALYSIS")
    print("="*70)
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # Get score history
        if 'Score/Episode' in ea.Tags()['scalars']:
            scores = ea.Scalars('Score/Episode')
            
            episodes = [s.step for s in scores]
            values = [s.value for s in scores]
            
            print(f"Episodes logged: {len(episodes)}")
            print(f"Score progression:")
            
            # Show chunks
            chunk_size = max(len(values) // 5, 1)
            for i in range(0, len(values), chunk_size):
                chunk = values[i:i+chunk_size]
                avg = np.mean(chunk)
                print(f"  Episodes {episodes[i]:4d}-{episodes[min(i+chunk_size, len(episodes))-1]:4d}: "
                      f"Avg score = {avg:.2f}")
            
            # Check if improving
            if len(values) > 100:
                early = np.mean(values[:50])
                late = np.mean(values[-50:])
                
                if late > early * 1.5:
                    print(f"\n✓ Learning is happening!")
                    print(f"   Early avg: {early:.2f}")
                    print(f"   Late avg: {late:.2f}")
                    print(f"   Improvement: {(late/early - 1)*100:.1f}%")
                    return True
                else:
                    print(f"\n❌ NOT IMPROVING!")
                    print(f"   Early avg: {early:.2f}")
                    print(f"   Late avg: {late:.2f}")
                    return False
        else:
            print("⚠ No score data in logs")
            return None
            
    except Exception as e:
        print(f"Could not read logs: {e}")
        return None


def check_hyperparameters(agent):
    """Check if hyperparameters are reasonable"""
    print("\n" + "="*70)
    print("6. HYPERPARAMETER CHECK")
    print("="*70)
    
    issues = []
    
    # Learning rates
    actor_lr = agent.actor.optimizer.param_groups[0]['lr']
    critic_lr = agent.critic_1.optimizer.param_groups[0]['lr']
    
    print(f"Actor LR: {actor_lr}")
    print(f"Critic LR: {critic_lr}")
    
    if actor_lr > 0.001:
        issues.append("Actor LR too high (> 0.001)")
    if critic_lr > 0.001:
        issues.append("Critic LR too high (> 0.001)")
    if actor_lr < 1e-5:
        issues.append("Actor LR too low (< 1e-5)")
    
    # Batch size
    batch_size = agent.batch_size
    print(f"Batch size: {batch_size}")
    
    if batch_size > 2048:
        issues.append("Batch size very large (> 2048)")
    
    # Tau
    tau = agent.tau
    print(f"Tau: {tau}")
    
    if tau > 0.1:
        issues.append("Tau too high (> 0.1)")
    if tau < 0.001:
        issues.append("Tau too low (< 0.001)")
    
    # Noise
    noise = agent.noise
    print(f"Noise: {noise}")
    
    if noise < 0.05:
        issues.append("Noise too low (< 0.05) - insufficient exploration")
    if noise > 0.3:
        issues.append("Noise too high (> 0.3) - too much randomness")
    
    if issues:
        print(f"\n⚠ POTENTIAL ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✓ Hyperparameters look reasonable")
        return True


def diagnose_vectorized_training(checkpoint_dir='temp/td3'):
    """Main diagnostic function"""
    print("\n" + "="*70)
    print("VECTORIZED TRAINING DIAGNOSTICS")
    print("="*70 + "\n")
    
    # Try to load agent
    try:
        from td3 import Agent
        import robosuite as suite
        from robosuite.wrappers import GymWrapper
        
        # Create dummy env for agent initialization
        env = suite.make("PickPlace", robots="Panda", 
                        controller_configs=suite.load_controller_config(default_controller="JOINT_POSITION"),
                        has_renderer=False, use_camera_obs=False, horizon=300,
                        reward_shaping=True, control_freq=20)
        env = GymWrapper(env)
        
        # Initialize agent
        agent = Agent(
            alpha=0.0005, beta=0.0005,
            input_dims=env.observation_space.shape,
            tau=0.05, env=env,
            n_actions=env.action_space.shape[0],
            layer1_size=512, layer2_size=256,
            batch_size=1024
        )
        
        # Load checkpoint if exists
        if os.path.exists(checkpoint_dir):
            agent.actor.checkpoint_dir = checkpoint_dir
            agent.critic_1.checkpoint_dir = checkpoint_dir
            agent.critic_2.checkpoint_dir = checkpoint_dir
            agent.actor.checkpoint_file = os.path.join(checkpoint_dir, 'actor_td3')
            agent.critic_1.checkpoint_file = os.path.join(checkpoint_dir, 'critic_1_td3')
            agent.critic_2.checkpoint_file = os.path.join(checkpoint_dir, 'critic_2_td3')
            
            try:
                agent.load_models()
                print("✓ Loaded checkpoint from", checkpoint_dir)
            except:
                print("⚠ Could not load checkpoint")
        
        # Run diagnostics
        print("\n")
        
        # 1. Buffer check
        has_data = check_replay_buffer(agent)
        
        # 2. Learning check
        is_learning = check_learning_activity(agent)
        
        # 3. Network check
        test_states = np.random.randn(10, env.observation_space.shape[0]).astype(np.float32)
        networks_ok = check_network_outputs(agent, test_states)
        
        # 4. Exploration check
        past_warmup = check_exploration(agent)
        
        # 5. Log analysis
        analyze_training_log()
        
        # 6. Hyperparameter check
        hyperparams_ok = check_hyperparameters(agent)
        
        # Summary
        print("\n" + "="*70)
        print("DIAGNOSIS SUMMARY")
        print("="*70)
        
        all_checks = [
            ("Replay buffer", has_data),
            ("Learning active", is_learning),
            ("Networks working", networks_ok),
            ("Past warmup", past_warmup),
            ("Hyperparameters", hyperparams_ok)
        ]
        
        for check_name, result in all_checks:
            status = "✓" if result else "❌"
            print(f"{status} {check_name}")
        
        # Recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        if not has_data:
            print("\n1. ❌ NOT ENOUGH DATA IN BUFFER")
            print("   - Agent hasn't started learning yet")
            print("   - Keep training! Need more episodes")
            print("   - Check that warmup period isn't too long")
        
        elif not is_learning:
            print("\n2. ❌ NO LEARNING HAPPENING")
            print("   - Check if learn() is being called")
            print("   - Verify training loop logic")
            print("   - Look for exceptions in training")
        
        elif not networks_ok:
            print("\n3. ❌ NETWORKS NOT PRODUCING GOOD OUTPUTS")
            print("   - Check for NaN/Inf in gradients")
            print("   - Lower learning rate")
            print("   - Check reward scale")
        
        elif not past_warmup:
            print("\n4. ⚠ STILL IN WARMUP PHASE")
            print("   - Agent is exploring randomly")
            print("   - Wait for warmup to finish")
            print("   - Or reduce warmup period")
        
        else:
            print("\n5. ⚠ SLOW LEARNING")
            print("   - Everything looks OK technically")
            print("   - Task might just be hard")
            print("   - Try:")
            print("     • Increase exploration noise")
            print("     • Train longer (20K+ episodes)")
            print("     • Check reward shaping is enabled")
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else 'temp/td3'
    
    diagnose_vectorized_training(checkpoint_dir)
