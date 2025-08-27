
import numpy as np
import torch
import time
import os
import glob
import re
import pandas as pd
from luxai_s3.wrappers import LuxAIS3GymEnv
from luxai_s3.params import EnvParams
from PPO_agent import Agent
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import traceback  # 상단에 추가

class SyncTrainer:
    def __init__(self, agents: List[Agent], target_steps: int = 2048, gae_lambda: float = 0.95):
        self.agents = agents
        self.target_steps = target_steps
        self.gae_lambda = gae_lambda
        self.total_steps = 0
        
    def should_update(self) -> bool:
        return all(agent.memory.steps_collected >= self.target_steps for agent in self.agents)
    
    def compute_gae_per_ship(self, rewards: List[float], values: List[float], 
                           next_value: float, dones: List[bool], gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            valid_data = [(r, v, d) for r, v, d in zip(rewards, values, dones) if r is not None]
            if not valid_data:
                return torch.tensor([]), torch.tensor([])
                
            rewards, values, dones = zip(*valid_data)
            
            gae = 0
            returns = []
            advantages = []
            
            for step in reversed(range(len(rewards))):
                if step == len(rewards) - 1:
                    next_non_terminal = 1.0 - float(dones[-1])
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - float(dones[step + 1])
                    next_val = values[step + 1]
                
                delta = rewards[step] + gamma * next_val * next_non_terminal - values[step]
                gae = delta + gamma * self.gae_lambda * next_non_terminal * gae
                
                returns.insert(0, gae + values[step])
                advantages.insert(0, gae)
                
            returns = torch.tensor(returns)
            advantages = torch.tensor(advantages)
            
            if len(advantages) > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return returns, advantages
            
        except Exception as e:
            print(f"Error in compute_gae_per_ship: {str(e)}")
            return torch.tensor([]), torch.tensor([])

    def sync_update(self) -> Dict[str, float]:
        if not self.should_update():
            return {}
            
        update_stats = {}
        
        print("\nDEBUG - Starting sync_update")
        print("Checking memory states before GAE computation...")
        
        # Check and update memories for all agents before computing GAE
        self.check_all_memories()
        
        for idx, agent in enumerate(self.agents):
            try:
                for ship_id, memory in agent.memory.ship_memories.items():
                    if memory.steps_collected == 0:
                        continue
                        
                    print(f"\nDEBUG - Processing Agent {idx} Ship {ship_id}")
                    print(f"Steps collected: {memory.steps_collected}")
                    print(f"Terminal states: {sum(memory.is_terminals)}/{len(memory.is_terminals)}")
                    
                    with torch.no_grad():
                        last_state = memory.states[-1]
                        if isinstance(last_state, torch.Tensor):
                            last_state = last_state.to(agent.ppo.device)
                        else:
                            last_state = torch.FloatTensor(last_state).to(agent.ppo.device)
                        
                        last_features = agent.ppo.policy_old(last_state)
                        next_value = agent.ppo.policy_old.value(last_features).item()
                        
                        print(f"Computing GAE for {len(memory.rewards)} steps")
                        values = [v.item() for v in memory.values]
                        returns, advantages = self.compute_gae_per_ship(
                            rewards=memory.rewards,
                            values=values,
                            next_value=next_value,
                            dones=memory.is_terminals,
                            gamma=agent.ppo.gamma
                        )
                        
                        if len(returns) > 0:
                            memory.returns = returns
                            memory.advantages = advantages
                            print(f"GAE computed successfully. Returns shape: {returns.shape}")
                
                # PPO update
                policy_loss, value_loss = agent.ppo.update(agent.memory)
                update_stats[f'agent{idx}_policy_loss'] = policy_loss
                update_stats[f'agent{idx}_value_loss'] = value_loss
                print(f"Agent {idx} updated - Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
                
                agent.memory.clear_all_memories()
                print(f"Agent {idx} memories cleared")
                
            except Exception as e:
                print(f"\nError updating agent {idx}:")
                print(f"Exception: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                continue
        
        return update_stats
        
    def check_all_memories(self):
        """Check and fix all memory states across all agents"""
        print("\nDEBUG - Checking all memories")
        
        for agent_idx, agent in enumerate(self.agents):
            print(f"\nAgent {agent_idx} Memory Check:")
            for ship_id, memory in agent.memory.ship_memories.items():
                if memory.steps_collected == 0:
                    continue
                    
                # 1. Check length consistency
                lengths = [
                    len(memory.states),
                    len(memory.action_types),
                    len(memory.detail_actions),
                    len(memory.type_logprobs),
                    len(memory.detail_logprobs),
                    len(memory.values),
                    len(memory.rewards),
                    len(memory.is_terminals)
                ]
                
                if len(set(lengths)) > 1:
                    print(f"Warning: Ship {ship_id} has inconsistent memory lengths:")
                    print(f"  States: {len(memory.states)}")
                    print(f"  Action types: {len(memory.action_types)}")
                    print(f"  Detail actions: {len(memory.detail_actions)}")
                    print(f"  Type logprobs: {len(memory.type_logprobs)}")
                    print(f"  Detail logprobs: {len(memory.detail_logprobs)}")
                    print(f"  Values: {len(memory.values)}")
                    print(f"  Rewards: {len(memory.rewards)}")
                    print(f"  Terminals: {len(memory.is_terminals)}")
                    
                    # Truncate to shortest length
                    min_length = min(lengths)
                    memory.states = memory.states[:min_length]
                    memory.action_types = memory.action_types[:min_length]
                    memory.detail_actions = memory.detail_actions[:min_length]
                    memory.type_logprobs = memory.type_logprobs[:min_length]
                    memory.detail_logprobs = memory.detail_logprobs[:min_length]
                    memory.values = memory.values[:min_length]
                    memory.rewards = memory.rewards[:min_length]
                    memory.is_terminals = memory.is_terminals[:min_length]
                    print(f"  Truncated all lists to length {min_length}")
                
                # 2. Check terminal states
                terminals_count = sum(memory.is_terminals)
                print(f"\nShip {ship_id} Terminal States:")
                print(f"  Total steps: {memory.steps_collected}")
                print(f"  Terminal states: {terminals_count}")
                print(f"  Is currently terminal: {memory.is_terminals[-1] if memory.is_terminals else False}")
                
                # 3. Update step counts
                memory.steps_collected = len(memory.states)
                
            # 4. Update total fleet memory steps
            agent.memory.steps_collected = sum(
                m.steps_collected for m in agent.memory.ship_memories.values()
            )
            print(f"Agent {agent_idx} total steps: {agent.memory.steps_collected}")
    
class CheckpointManager:
    def __init__(self, save_dir='checkpoints'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_checkpoint(self, agents, training_stats, episode):
        filename = f'checkpoint_episode_{episode:04d}.pth'
        path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'episode': episode,
            'agent0': {
                'model': agents[0].ppo.policy.state_dict(),
                'optimizer': agents[0].ppo.optimizer.state_dict()
            },
            'agent1': {
                'model': agents[1].ppo.policy.state_dict(),
                'optimizer': agents[1].ppo.optimizer.state_dict()
            },
            'training_stats': training_stats,
            'hyperparameters': {
                'learning_rate': agents[0].ppo.optimizer.param_groups[0]['lr'],
                'gamma': agents[0].ppo.gamma,
                'eps_clip': agents[0].ppo.eps_clip,
                'K_epochs': agents[0].ppo.K_epochs,
            }
        }
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # CSV도 함께 저장
        stats_df = pd.DataFrame(training_stats)
        stats_df.to_csv(os.path.join(self.save_dir, f'training_stats_{episode:04d}.csv'), index=False)
        
    def load_checkpoint(self, agents, filename=None):
        if filename is None:
            filename = self._get_latest_checkpoint()
            if filename is None:
                raise FileNotFoundError("No checkpoints found")
        
        path = os.path.join(self.save_dir, filename)
        print(f"Loading checkpoint: {path}")
        
        checkpoint = torch.load(path)
        
        # 모델과 옵티마이저 상태 복원
        agents[0].ppo.policy.load_state_dict(checkpoint['agent0']['model'])
        agents[0].ppo.optimizer.load_state_dict(checkpoint['agent0']['optimizer'])
        agents[1].ppo.policy.load_state_dict(checkpoint['agent1']['model'])
        agents[1].ppo.optimizer.load_state_dict(checkpoint['agent1']['optimizer'])
        
        return checkpoint['episode'], checkpoint['training_stats']
    
    def _get_latest_checkpoint(self):
        checkpoints = glob.glob(os.path.join(self.save_dir, 'checkpoint_episode_*.pth'))
        if not checkpoints:
            return None
        
        # 에피소드 번호로 정렬
        checkpoints.sort(key=lambda x: int(re.search(r'episode_(\d+)', x).group(1)))
        return os.path.basename(checkpoints[-1])

def process_reward(reward):
    """JAX → numpy float 변환 보조 함수"""
    if hasattr(reward, 'device_buffer'):
        return np.array(reward)
    return reward

def plot_training_stats(stats, save_dir='checkpoints'):
    # 보상 그래프
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(stats['episodes'], stats['rewards_0'], label='Agent 0')
    plt.plot(stats['episodes'], stats['rewards_1'], label='Agent 1')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()
    
    # Loss 그래프
    plt.subplot(2, 2, 2)
    plt.plot(stats['episodes'], stats['policy_losses_0'], label='Policy Loss')
    plt.plot(stats['episodes'], stats['value_losses_0'], label='Value Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    
    # 승리 수 그래프
    plt.subplot(2, 2, 3)
    plt.plot(stats['episodes'], stats['match_wins_0'], label='Agent 0 Wins')
    plt.plot(stats['episodes'], stats['match_wins_1'], label='Agent 1 Wins')
    plt.axhline(y=2.5, color='r', linestyle='--', alpha=0.3)  # 기대값 라인
    plt.xlabel('Episode')
    plt.ylabel('Wins per Episode')
    plt.title('Match Wins (out of 5)')
    plt.legend()
    
    # FPS 그래프
    plt.subplot(2, 2, 4)
    plt.plot(stats['episodes'], stats['fps'])
    plt.xlabel('Episode')
    plt.ylabel('Steps per Second')
    plt.title('Training Speed')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_stats.png'))
    plt.close()

def train(num_episodes=1001, log_interval=10, resume_training=False, target_steps=2048):
    device = "mps"
    checkpoint_manager = CheckpointManager()
    
    env = LuxAIS3GymEnv()
    env_params = EnvParams(
        map_type=1,
        max_steps_in_match=100,
        match_count_per_episode=5
    )
    
    agent0 = Agent("player_0", env_params, device=device, train_mode=True, target_steps=target_steps)
    agent1 = Agent("player_1", env_params, device=device, train_mode=True, target_steps=target_steps)
    agents = [agent0, agent1]
    
    sync_trainer = SyncTrainer(agents, target_steps=target_steps)
    
    start_episode = 0
    training_stats = {
        'episodes': [],
        'rewards_0': [],
        'rewards_1': [],
        'policy_losses_0': [],
        'value_losses_0': [],
        'steps_per_episode': [],
        'fps': [],
        'match_wins_0': [],
        'match_wins_1': []
    }
    
    if resume_training:
        try:
            start_episode, training_stats = checkpoint_manager.load_checkpoint(agents)
            print(f"Resumed training from episode {start_episode}")
        except FileNotFoundError as e:
            print(f"Warning: {e}. Starting fresh training.")
    
    total_steps = 0
    start_time = time.time()
    policy_loss0 = value_loss0 = 0.0
    policy_loss1 = value_loss1 = 0.0
    
    print("Starting training...")
    
    for i_episode in range(start_episode, num_episodes):
        episode_reward_0 = 0.0
        episode_reward_1 = 0.0
        steps_in_episode = 0
        match_wins_0 = 0
        match_wins_1 = 0
        
        obs, info = env.reset(seed=i_episode, options=dict(params=env_params))
        for agent in agents:
            agent.memory.clear_all_memories()
        
        for match_idx in range(5):
            match_steps = 0
            match_reward_0 = 0.0
            match_reward_1 = 0.0
            match_done = False
            
            while not match_done and match_steps < 100:
                match_steps += 1
                steps_in_episode += 1
                total_steps += 1
                
                # Check and update memories periodically during training
                if match_steps % 20 == 0:  # Every 20 steps
                    for agent in agents:
                        agent.check_and_update_memories()
                
                actions0 = agent0.act(steps_in_episode, obs["player_0"])
                actions1 = agent1.act(steps_in_episode, obs["player_1"])
                
                next_obs, reward, terminated, truncated, info = env.step({
                    "player_0": actions0,
                    "player_1": actions1
                })
                
                r0 = float(process_reward(reward["player_0"]))
                r1 = float(process_reward(reward["player_1"]))
                match_reward_0 += r0
                match_reward_1 += r1
                episode_reward_0 += r0
                episode_reward_1 += r1
                
                match_done = terminated["player_0"] or truncated["player_0"] or match_steps >= 100
                
                if sync_trainer.should_update():
                    # Check and update memories before PPO update
                    for agent in agents:
                        agent.check_and_update_memories()
                    update_stats = sync_trainer.sync_update()
                    if update_stats:
                        policy_loss0 = update_stats['agent0_policy_loss']
                        value_loss0 = update_stats['agent0_value_loss']
                        policy_loss1 = update_stats['agent1_policy_loss']
                        value_loss1 = update_stats['agent1_value_loss']
                
                obs = next_obs
            
            if match_done:
                # Check and update memories at match end
                for agent in agents:
                    agent.check_and_update_memories()
                
                if match_reward_0 > match_reward_1:
                    match_wins_0 += 1
                elif match_reward_1 > match_reward_0:
                    match_wins_1 += 1
            
            if match_idx < 4:
                obs = next_obs
        
        # 에피소드 종료 통계 저장 및 출력
        if i_episode % log_interval == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            fps = total_steps / (elapsed if elapsed > 0 else 1e-9)
            
            training_stats['episodes'].append(i_episode)
            training_stats['rewards_0'].append(episode_reward_0)
            training_stats['rewards_1'].append(episode_reward_1)
            training_stats['policy_losses_0'].append(policy_loss0)
            training_stats['value_losses_0'].append(value_loss0)
            training_stats['steps_per_episode'].append(steps_in_episode)
            training_stats['fps'].append(fps)
            training_stats['match_wins_0'].append(match_wins_0)
            training_stats['match_wins_1'].append(match_wins_1)
            
            print(f"Episode {i_episode:4d} | "
                  f"Reward0: {episode_reward_0:.2f} (Wins: {match_wins_0}) | "
                  f"Reward1: {episode_reward_1:.2f} (Wins: {match_wins_1}) | "
                  f"Steps: {steps_in_episode} | "
                  f"FPS: {fps:.2f} | "
                  f"(PolicyLoss0: {policy_loss0:.4f}, ValueLoss0: {value_loss0:.4f})")
        
        # 체크포인트 저장
        if i_episode % 10 == 0:
            checkpoint_manager.save_checkpoint(agents, training_stats, i_episode)
            plot_training_stats(training_stats)
    
    return training_stats

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--target-steps', type=int, default=2048, help='Steps to collect before PPO update')
    
    args = parser.parse_args()
    
    stats = train(
        num_episodes=args.episodes,
        log_interval=args.log_interval,
        resume_training=args.resume,
        target_steps=args.target_steps
    )
    
    # 최종 학습 결과 시각화
    plot_training_stats(stats)