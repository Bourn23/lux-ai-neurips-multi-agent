
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
from typing import List, Tuple, Dict
import traceback  # 상단에 추가
import os

class NodeType(Enum):
    unknown = -1
    empty = 0
    asteroid = 1
    nebula = 2

def to_numpy(x):
    """JAX 배열 또는 일반 배열을 numpy 배열로 변환"""
    if hasattr(x, 'device_buffer'):  # JAX 배열인 경우
        return np.array(x)
    return np.array(x)

def safe_squeeze(x):
    """안전하게 차원 축소"""
    x = to_numpy(x)
    if x.ndim > 1:
        return np.squeeze(x)
    return x

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown
        self.energy = None
        self.is_visible = False
        self._relic = False
        self._reward = False
        self._explored_for_relic = False
        self._explored_for_reward = False
        
    @property
    def coordinates(self):
        return (self.x, self.y)
        
    @property
    def is_walkable(self):
        return self.type != NodeType.asteroid

class Space:
    def __init__(self, size=24):
        self.size = size
        self._nodes = [[Node(x, y) for x in range(size)] for y in range(size)]
        self.relic_nodes = set()
        self.reward_nodes = set()
        
    def get_node(self, x: int, y: int) -> Node:
        if 0 <= x < self.size and 0 <= y < self.size:
            return self._nodes[y][x]
        return None
        
    def update(self, obs):
        if isinstance(obs, dict):
            sensor_mask = obs["sensor_mask"]
            map_features = obs["map_features"]
        else:
            sensor_mask = obs.sensor_mask
            map_features = obs.map_features
            
        # Update tiles and energy
        for x in range(self.size):
            for y in range(self.size):
                node = self.get_node(x, y)
                if sensor_mask[x, y]:
                    node.is_visible = True
                    if isinstance(map_features, dict):
                        tile_type = int(map_features["tile_type"][x, y])
                    else:
                        tile_type = int(map_features.tile_type[x, y])
                    node.type = NodeType(tile_type)
                else:
                    node.is_visible = False

class Ship:
    def __init__(self, unit_id: int, team_id: int, fleet=None):
        self.unit_id = unit_id
        self.team_id = team_id
        self.fleet = fleet
        self.energy = 0
        self.position = None
        self.task = None
        self.sub_task = None
        self.target = None
        self.is_active = False
        
    def update(self, position, energy, is_active):
        self.position = position
        self.energy = energy
        self.is_active = is_active
        
    def clean(self):
        self.energy = 0
        self.position = None
        self.task = None
        self.sub_task = None
        self.target = None
        self.is_active = False

class ShipMemory:
    def __init__(self, ship_id, device=torch.device("mps")):
        self.ship_id = ship_id
        self.device = device
        self.clear_memory()
        
    def clear_memory(self):
        self.states = []
        self.action_types = []
        self.detail_actions = []
        self.type_logprobs = []
        self.detail_logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        self.returns = None
        self.advantages = None
        self.steps_collected = 0

class FleetMemory:
    def __init__(self, max_ships=16, device=torch.device("mps")):
        self.device = device
        self.ship_memories = {i: ShipMemory(i, device) for i in range(max_ships)}
        self.steps_collected = 0
    
    def push_ship_transition(self, ship_id, state=None, action_type=None, detail_action=None, 
                           type_logprob=None, detail_logprob=None, value=None, reward=None, 
                           is_terminal=None):
        memory = self.ship_memories[ship_id]
        
        # Skip if ship was already terminated
        if memory.steps_collected > 0 and memory.is_terminals and memory.is_terminals[-1]:
            return
            
        # Case 1: Pushing full transition (including state)
        if state is not None:
            memory.states.append(state)
            memory.steps_collected += 1
            self.steps_collected += 1
            
            if action_type is not None:
                memory.action_types.append(action_type)
            if detail_action is not None:
                memory.detail_actions.append(detail_action)
            if type_logprob is not None:
                memory.type_logprobs.append(type_logprob)
            if detail_logprob is not None:
                memory.detail_logprobs.append(detail_logprob)
            if value is not None:
                memory.values.append(value)
                
        # Case 2: Just updating last transition's reward/terminal
        elif memory.steps_collected > 0:
            if reward is not None:
                # Update or append reward based on current length
                if len(memory.rewards) < memory.steps_collected:
                    memory.rewards.append(reward)
                else:
                    memory.rewards[-1] = reward
                    
            if is_terminal is not None:
                # Update or append terminal flag based on current length
                if len(memory.is_terminals) < memory.steps_collected:
                    memory.is_terminals.append(is_terminal)
                else:
                    memory.is_terminals[-1] = is_terminal
            
    def update_terminal_flag(self, ship_id):
        memory = self.ship_memories[ship_id]
        if memory.steps_collected > 0 and memory.is_terminals and not memory.is_terminals[-1]:
            memory.is_terminals[-1] = True
            # print(f"\nDEBUG - Updated terminal flag for Ship {ship_id}")
            # print(f"Steps: {memory.steps_collected}")
            # print(f"Terminals: {len(memory.is_terminals)}")
            
    def handle_ship_death(self, ship_id):
        memory = self.ship_memories[ship_id]
        if memory.steps_collected > 0:
            if not memory.is_terminals or not memory.is_terminals[-1]:
                self.update_terminal_flag(ship_id)
            # print(f"\nDEBUG - Ship {ship_id} death handled")
            # print(f"Final steps: {memory.steps_collected}")
            # print(f"Final terminals: {len(memory.is_terminals)}")

    def clean_invalid_data(self):
        for ship_id, memory in self.ship_memories.items():
            if memory.steps_collected == 0:
                continue
                
            # Get minimum valid length
            lengths = [
                len(memory.states),
                len(memory.action_types) if memory.action_types else float('inf'),
                len(memory.detail_actions) if memory.detail_actions else float('inf'),
                len(memory.type_logprobs) if memory.type_logprobs else float('inf'),
                len(memory.detail_logprobs) if memory.detail_logprobs else float('inf'),
                len(memory.values) if memory.values else float('inf'),
                len(memory.rewards) if memory.rewards else float('inf'),
                len(memory.is_terminals) if memory.is_terminals else float('inf')
            ]
            
            min_length = min(length for length in lengths if length > 0)
            
            if min_length < memory.steps_collected:
                memory.states = memory.states[:min_length]
                if memory.action_types: memory.action_types = memory.action_types[:min_length]
                if memory.detail_actions: memory.detail_actions = memory.detail_actions[:min_length]
                if memory.type_logprobs: memory.type_logprobs = memory.type_logprobs[:min_length]
                if memory.detail_logprobs: memory.detail_logprobs = memory.detail_logprobs[:min_length]
                if memory.values: memory.values = memory.values[:min_length]
                if memory.rewards: memory.rewards = memory.rewards[:min_length]
                if memory.is_terminals: memory.is_terminals = memory.is_terminals[:min_length]
                memory.steps_collected = min_length
                print(f"\nDEBUG - Cleaned data for Ship {ship_id}")
                print(f"New steps: {memory.steps_collected}")
                
    def clear_all_memories(self):
        for memory in self.ship_memories.values():
            memory.clear_memory()
        self.steps_collected = 0
        
    def debug_ship_status(self, ship_id, is_active):
        memory = self.ship_memories[ship_id]
        if memory.steps_collected > 0:
            # print(f"\nDEBUG - Ship {ship_id} Status Update:")
            # print(f"Active: {is_active}")
            # print(f"Steps: {memory.steps_collected}")
            terminal_count = sum(1 for t in memory.is_terminals if t) if memory.is_terminals else 0
            # print(f"Terminal states: {terminal_count}/{len(memory.is_terminals) if memory.is_terminals else 0}")
            if not is_active:
                self.handle_ship_death(ship_id)
                
class Fleet:
    def __init__(self, team_id: int, max_units: int = 16):
        self.team_id = team_id
        self.ships = [Ship(i, team_id, self) for i in range(max_units)]
        self.points = 0
        self.memory = None
        
    def update(self, obs):
        if isinstance(obs, dict):
            self.points = float(obs["team_points"][self.team_id])
            units_mask = np.array(obs["units_mask"][self.team_id])
            positions = np.array(obs["units"]["position"][self.team_id])
            energies = np.array(obs["units"]["energy"][self.team_id])
        else:
            # JAX 배열 처리
            self.points = float(np.array(obs.team_points[self.team_id]))
            units_mask = np.array(obs.units_mask[self.team_id])
            positions = np.array(obs.units.position[self.team_id])
            energies = np.array(obs.units.energy[self.team_id])
        
        # 배열 차원 확인 및 필요시 squeeze
        if units_mask.ndim > 1:
            units_mask = np.squeeze(units_mask)
        if positions.ndim > 2:
            positions = np.squeeze(positions)
        if energies.ndim > 1:
            energies = np.squeeze(energies)
        
        # 각 유닛 업데이트
        for ship, active, pos, energy in zip(
            self.ships,
            units_mask,
            positions,
            energies
        ):
            was_active = ship.is_active
            if active:
                ship.update(pos, float(energy), True)
            else:
                if was_active:  # Ship이 방금 죽은 경우
                    self.memory.handle_ship_death(ship.unit_id)
                ship.clean()
            self.memory.debug_ship_status(ship.unit_id, active)
    
    @property
    def active_ships(self):
        return [ship for ship in self.ships if ship.is_active]

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def nearby_positions(x, y, distance):
    positions = []
    for dx in range(-distance, distance + 1):
        for dy in range(-distance, distance + 1):
            if abs(dx) + abs(dy) <= distance:
                pos = (x + dx, y + dy)
                positions.append(pos)
    return positions

class Memory:
    def __init__(self, device=torch.device("mps")):
        self.device = device
        self.states = []
        self.action_types = []
        self.detail_actions = []
        self.type_logprobs = []
        self.detail_logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        
        self.returns = None
        self.advantages = None
        self.steps_collected = 0
        
    def push(self, reward, is_terminal):
        try:
            self.rewards.append(float(reward))
            self.is_terminals.append(bool(is_terminal))
            self.steps_collected += 1
        except Exception as e:
            print(f"DEBUG - Error in Memory.push: {e}")
            
    def clear_memory(self):
        try:
            del self.states[:]
            del self.action_types[:]
            del self.detail_actions[:]
            del self.type_logprobs[:]
            del self.detail_logprobs[:]
            del self.rewards[:]
            del self.is_terminals[:]
            del self.values[:]
            self.returns = None
            self.advantages = None
            self.steps_collected = 0
        except Exception as e:
            print(f"DEBUG - Error in Memory.clear_memory: {e}")

    def safe_to_device(self, tensor_list):
        """텐서 리스트를 안전하게 device로 이동"""
        result = []
        for tensor in tensor_list:
            if tensor is None:
                result.append(torch.zeros(1, device=self.device))
            elif isinstance(tensor, torch.Tensor):
                # 이미 올바른 device에 있다면 그대로 사용
                if tensor.device == self.device:
                    result.append(tensor)
                else:
                    result.append(tensor.to(self.device))
            else:
                try:
                    # tensor가 아닌 경우 변환 시도
                    result.append(torch.tensor([tensor], device=self.device))
                except:
                    # 변환 실패 시 0 텐서 사용
                    print(f"DEBUG - Failed to convert to tensor: {tensor}")
                    result.append(torch.zeros(1, device=self.device))
        return result

    def get_batch(self):
        """안전하게 배치 반환"""
        try:
            # 모든 리스트에 대해 safe_to_device 적용
            states = self.safe_to_device(self.states)
            action_types = self.safe_to_device(self.action_types)
            detail_actions = self.safe_to_device(self.detail_actions)
            type_logprobs = self.safe_to_device(self.type_logprobs)
            detail_logprobs = self.safe_to_device(self.detail_logprobs)
            values = self.safe_to_device(self.values)
            
            # non-tensor 데이터는 그대로 사용
            rewards = self.rewards
            is_terminals = self.is_terminals
            
            return {
                'states': states,
                'action_types': action_types,
                'detail_actions': detail_actions,
                'type_logprobs': type_logprobs,
                'detail_logprobs': detail_logprobs,
                'rewards': rewards,
                'is_terminals': is_terminals,
                'values': values
            }
            
        except Exception as e:
            print(f"DEBUG - Error in Memory.get_batch: {e}")
            print(f"DEBUG - Memory state lengths: states={len(self.states)}, "
                  f"action_types={len(self.action_types)}, "
                  f"detail_actions={len(self.detail_actions)}")
            
            # 에러 발생 시 안전한 기본값 반환
            empty_tensor = torch.zeros(1, device=self.device)
            return {
                'states': [empty_tensor],
                'action_types': [empty_tensor],
                'detail_actions': [empty_tensor],
                'type_logprobs': [empty_tensor],
                'detail_logprobs': [empty_tensor],
                'rewards': [0.0],
                'is_terminals': [True],
                'values': [empty_tensor]
            }
        
    def print_debug_info(self):
        """현재 메모리 상태 출력"""
        print("\nDEBUG - Memory Status:")
        print(f"Steps collected: {self.steps_collected}")
        print(f"Device: {self.device}")
        print(f"List lengths:")
        print(f"  states: {len(self.states)}")
        print(f"  action_types: {len(self.action_types)}")
        print(f"  detail_actions: {len(self.detail_actions)}")
        print(f"  type_logprobs: {len(self.type_logprobs)}")
        print(f"  detail_logprobs: {len(self.detail_logprobs)}")
        print(f"  values: {len(self.values)}")
        print(f"  rewards: {len(self.rewards)}")
        print(f"  is_terminals: {len(self.is_terminals)}")
        
        # 샘플 데이터 출력
        if len(self.states) > 0:
            print("\nLatest data:")
            print(f"  state: {type(self.states[-1])}, device: {self.states[-1].device if isinstance(self.states[-1], torch.Tensor) else 'N/A'}")
            print(f"  action_type: {type(self.action_types[-1])}, device: {self.action_types[-1].device if isinstance(self.action_types[-1], torch.Tensor) else 'N/A'}")
            print(f"  detail_action: {type(self.detail_actions[-1])}, device: {self.detail_actions[-1].device if isinstance(self.detail_actions[-1], torch.Tensor) else 'N/A'}")
            print("="*50)

class PPO:
    def __init__(self, state_dim, 
                 lr=0.0001,
                 gamma=0.99,
                 eps_clip=0.15,
                 K_epochs=4,
                 target_steps=2048,
                 device=torch.device("mps"),
                 entropy_coef=0.01,
                 agent=None):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.target_steps = target_steps
        self.entropy_coef = entropy_coef
        self.agent = agent
        
        self.policy = SimplifiedActorCritic(state_dim).to(self.device)
        self.policy_old = SimplifiedActorCritic(state_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.cnn.parameters(), 'lr': lr * 0.1},
            {'params': self.policy.base_network.parameters(), 'lr': lr},
            {'params': self.policy.feature_combiner.parameters(), 'lr': lr},
            {'params': self.policy.action_type.parameters(), 'lr': lr * 2.0},
            {'params': self.policy.direction.parameters(), 'lr': lr * 2.0},
            {'params': self.policy.value.parameters(), 'lr': lr}
        ])

    def should_update(self, memory):
        """메모리에 충분한 데이터가 쌓였는지 확인"""
        return memory.steps_collected >= self.target_steps
        
    def collect_rollout(self, state, memory, ship_id, obs, ship):
        try:
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state)
                state = state.to(self.device)
                
                features = self.policy_old(state)
                
                # Pass agent's space instead of ship.fleet.space
                final_action, action_type, direction, type_logprob, direction_logprob = \
                    self.policy_old.get_masked_action(features, obs, ship, self.agent.space)
                
                memory.push_ship_transition(
                    ship_id=ship_id,
                    state=state.clone(),
                    action_type=action_type.clone(),
                    detail_action=direction.clone(),
                    type_logprob=type_logprob.clone(),
                    detail_logprob=direction_logprob.clone(),
                    value=self.policy_old.value(features).detach(),
                    reward=None,
                    is_terminal=False
                )
                
                return final_action
                
        except Exception as e:
            print(f"\nERROR in collect_rollout:")
            print(f"Exception: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            memory.print_debug_info(ship_id)
            return [0, 0, 0]
            
    def act(self, state, memory=None):
        """act 메서드"""
        return self.collect_rollout(state, memory) if memory is not None else self.policy.act(state)
        
    def update(self, memory):
        """PPO 업데이트"""
        if not hasattr(memory, 'returns') or not hasattr(memory, 'advantages'):
            return 0.0, 0.0
        
        batch = memory.get_batch()
        states = torch.stack([torch.FloatTensor(s) for s in batch['states']]).to(self.device)
        action_types = torch.stack(batch['action_types']).to(self.device)
        directions = torch.stack(batch['detail_actions']).to(self.device)
        old_type_logprobs = torch.stack(batch['type_logprobs']).detach()
        old_direction_logprobs = torch.stack(batch['detail_logprobs']).detach()
        
        returns = memory.returns.to(self.device)
        advantages = memory.advantages.to(self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.K_epochs):
            features = self.policy(states)
            
            # 행동 타입 선택
            action_probs = self.policy.action_type(features)
            action_dist = Categorical(action_probs)
            curr_type_logprobs = action_dist.log_prob(action_types)
            
            # 방향 선택
            direction_probs = self.policy.direction(features)
            direction_dist = Categorical(direction_probs)
            curr_direction_logprobs = direction_dist.log_prob(directions)
            
            state_values = self.policy.value(features).squeeze()
            
            # 엔트로피 보너스 계산
            entropy = (action_dist.entropy().mean() + direction_dist.entropy().mean()) / 2
            
            # Ratios
            type_ratios = torch.exp(curr_type_logprobs - old_type_logprobs)
            direction_ratios = torch.exp(curr_direction_logprobs - old_direction_logprobs)
            
            # Surrogate losses
            surr1_type = type_ratios * advantages
            surr2_type = torch.clamp(type_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            surr1_direction = direction_ratios * advantages
            surr2_direction = torch.clamp(direction_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # 최종 손실 계산 (엔트로피 보너스 포함)
            action_loss = -torch.min(surr1_type, surr2_type).mean() - torch.min(surr1_direction, surr2_direction).mean()
            value_loss = 0.5 * self.MseLoss(state_values, returns)
            
            loss = action_loss + value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # 0
            self.optimizer.step()
            
            total_policy_loss += float(action_loss.detach())
            total_value_loss += float(value_loss.detach())
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return total_policy_loss / self.K_epochs, total_value_loss / self.K_epochs

class SimplifiedActorCritic(nn.Module):
    def __init__(self, state_dim):
        super(SimplifiedActorCritic, self).__init__()
        
        # CNN for map features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU()
        )
        
        # Base network for state features
        self.base_network = nn.Sequential(
            nn.Linear(26, 64),
            nn.ReLU()
        )
        
        # Feature combiner
        self.feature_combiner = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU()
        )
        
        # Action networks
        self.action_type = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Raw logits for Wait/Move/Combat
        )
        
        self.direction = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Raw logits for directions
        )
        
        # Value network
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Split state into map and other features
        if state.dim() == 1:
            map_state = state[:576].view(1, 1, 24, 24)
            other_state = state[576:].unsqueeze(0)
        else:
            map_state = state[:, :576].view(-1, 1, 24, 24)
            other_state = state[:, 576:]
        
        # Extract features
        map_features = self.cnn(map_state)
        base_features = self.base_network(other_state)
        combined = self.feature_combiner(torch.cat([map_features, base_features], dim=-1))
        
        return combined
        
    def get_masked_action(self, features, obs, ship, space, return_logprobs=True):
        """Get masked actions and their log probabilities"""
        action_type_mask, direction_mask = self.get_action_mask(obs, ship, space)
        action_type_mask = action_type_mask.to(features.device)
        direction_mask = direction_mask.to(features.device)
        
        # Action type selection with masking
        action_logits = self.action_type(features)
        masked_logits = action_logits * action_type_mask - 1e8 * (1 - action_type_mask)
        action_probs = F.softmax(masked_logits, dim=-1)
        action_dist = Categorical(action_probs)
        action_type = action_dist.sample()
        
        # Direction selection with masking
        direction = torch.zeros(1, device=features.device)
        direction_logprob = torch.zeros(1, device=features.device)
        
        if action_type.item() in [1, 2]:  # Move or Combat
            direction_logits = self.direction(features)
            masked_dir_logits = direction_logits * direction_mask - 1e8 * (1 - direction_mask)
            direction_probs = F.softmax(masked_dir_logits, dim=-1)
            direction_dist = Categorical(direction_probs)
            direction = direction_dist.sample()
            if return_logprobs:
                direction_logprob = direction_dist.log_prob(direction)
        
        # Convert to final action format
        if action_type.item() == 0:  # Wait
            final_action = [0, 0, 0]
        elif action_type.item() == 1:  # Move
            final_action = [direction.item(), 0, 0]
        else:  # Combat
            final_action = [0, direction.item(), 0]
            
        if return_logprobs:
            return (
                final_action,
                action_type,
                direction,
                action_dist.log_prob(action_type),
                direction_logprob
            )
        return final_action

    def get_action_mask(self, obs, ship, space):
        """Generate action masks based on game state"""
        action_type_mask = torch.ones(3)  # [Wait, Move, Combat]
        direction_mask = torch.ones(5)    # [Stay, Up, Right, Down, Left]
        
        # Energy constraints
        if ship.energy < 10:
            action_type_mask[1:] = 0  # Disable Move and Combat
            direction_mask[1:] = 0    # Disable all movement
        
        # Combat availability 
        has_enemy_nearby = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = ship.position[0] + dx, ship.position[1] + dy
                if 0 <= x < 24 and 0 <= y < 24:
                    if isinstance(obs, dict):
                        enemy_units = obs["units"]["position"][1 - ship.team_id]
                    else:
                        enemy_units = obs.units.position[1 - ship.team_id]
                    for enemy_pos in enemy_units:
                        if enemy_pos[0] == x and enemy_pos[1] == y:
                            has_enemy_nearby = True
                            break
        if not has_enemy_nearby:
            action_type_mask[2] = 0  # Disable Combat
        
        # Movement constraints
        directions = [(0,0), (0,1), (1,0), (0,-1), (-1,0)]
        for i, (dx, dy) in enumerate(directions):
            x, y = ship.position[0] + dx, ship.position[1] + dy
            if not (0 <= x < 24 and 0 <= y < 24):
                direction_mask[i] = 0
                continue
            
            node = space.get_node(x, y)
            if node and node.type == NodeType.asteroid:
                direction_mask[i] = 0
        
        return action_type_mask, direction_mask
    
class Agent:
    def __init__(self, player: str, env_cfg=None, device=torch.device("mps"), train_mode=False, target_steps=2048) -> None:
        # Basic parameters
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.env_cfg = env_cfg
        self.device = device
        self.train_mode = train_mode
        self.prev_team_points = None

        # State dimensions
        map_dim = 576
        base_dim = 8
        task_dim = 12
        target_dim = 3
        sub_task_dim = 3
        self.state_dim = map_dim + base_dim + task_dim + sub_task_dim + target_dim

        # Initialize components
        self.memory = FleetMemory(max_ships=16, device=device)
        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.fleet.memory = self.memory

        # Initialize PPO
        self.ppo = PPO(
            state_dim=self.state_dim,
            lr=0.0002,
            gamma=0.99,
            eps_clip=0.15,
            K_epochs=4,
            device=device,
            target_steps=2048,
            entropy_coef=0.01,
            agent=self  # Add reference to agent
        )

        # Load pre-trained model if in inference mode
        if not self.train_mode:
            try:
                checkpoint = torch.load('../checkpoints/checkpoint_episode_0050.pth', map_location=self.device)
                self.ppo.policy.load_state_dict(checkpoint['agent0']['model'])
                self.ppo.policy.eval()
                print("Loaded pre-trained model for inference.", file=sys.stderr)
            except Exception as e:
                print("No pre-trained model found or error:", e)
    
    def _process_state(self, obs, ship):
        # 맵 상태를 하나의 24x24 배열로 표현
        map_state = np.zeros((24, 24), dtype=np.float32)
        for x in range(24):
            for y in range(24):
                node = self.space.get_node(x, y)
                if node:
                    value = 0.0
                    if node.is_visible:
                        value += 1.0
                    if node._explored_for_relic:
                        value += 2.0
                    if node._explored_for_reward:
                        value += 4.0
                    if node.type == NodeType.asteroid:
                        value += 8.0
                    elif node.type == NodeType.nebula:
                        value += 16.0
                    map_state[x][y] = value/31.0  # 정규화
        
        # 1차원으로 변환
        map_state = map_state.flatten()  # 576 크기
        
        # 기본 상태 정보
        base_state = [
            ship.position[0] / 24.0,
            ship.position[1] / 24.0,
            ship.energy / 100.0,
            float(ship.is_active),
            float(obs["team_points"][self.team_id] if isinstance(obs, dict) else obs.team_points[self.team_id]) / 1000.0,
            float(obs["team_points"][1 - self.team_id] if isinstance(obs, dict) else obs.team_points[1 - self.team_id]) / 1000.0,
            float(ship.energy < 50),
            float(obs.get("match_steps", 0) if isinstance(obs, dict) else getattr(obs, "match_steps", 0)) / 100.0
        ]
        
        # 태스크 상태
        task_state = [
            float(ship.task == 'explore'),
            float(ship.task == 'harvest'),
            float(ship.task == 'combat'),
            float(ship.task is None)
        ] * 3
        
        # (4) sub_task_state 추가
        sub_task_state = [
            float(ship.sub_task == 'deep_scout'),
            float(ship.sub_task == 'guard_relic'),
            float(ship.sub_task is None)
        ]

        # 목표 상태
        target_state = []
        if ship.target:
            dist = manhattan_distance(ship.position, ship.target.coordinates)
            target_state.extend([
                ship.target.x / 24.0,
                ship.target.y / 24.0,
                dist / 24.0
            ])
        else:
            target_state.extend([0.0, 0.0, 1.0])
        
        return np.concatenate([
            map_state,      # 576 (24*24)
            base_state,     # 8
            task_state,     # 12
            sub_task_state,  # 3 (예시)
            target_state    # 3,
        ]).astype(np.float32)
        
    def calculate_reward(self, obs, ship):
        reward = 0.0
        
        # Survival & Energy management (increased base rewards)
        if ship.energy > 50:
            reward += 0.5  # Increased from 0.1
        else:
            reward -= 1.0  # Increased from 0.2
        
        # Team score (main objective)
        if self.prev_team_points is not None:
            if isinstance(obs, dict):
                current_points = float(obs["team_points"][self.team_id])
            else:
                current_points = float(obs.team_points[self.team_id])
            
            point_gained = current_points - self.prev_team_points
            if point_gained > 0:
                reward += point_gained * 2  # Doubled the reward for points
        
        # Match victory/defeat (increased rewards)
        if isinstance(obs, dict):
            match_done = obs.get("terminated", False) or obs.get("truncated", False)
            match_steps = obs.get("match_steps", 0)
        else:
            match_done = getattr(obs, "terminated", False) or getattr(obs, "truncated", False)
            match_steps = getattr(obs, "match_steps", 0)
            
        if match_done or match_steps >= 100:
            if isinstance(obs, dict):
                current_points = float(obs["team_points"][self.team_id])
                opponent_points = float(obs["team_points"][1 - self.team_id])
            else:
                current_points = float(obs.team_points[self.team_id])
                opponent_points = float(obs.team_points[1 - self.team_id])
                
            if current_points > opponent_points:
                reward += 10.0  # Increased from 5.0
            elif current_points < opponent_points:
                reward -= 5.0   # Increased from 2.0
        
        # Inaction penalty (increased)
        if ship.action is None or (isinstance(ship.action, list) and all(a == 0 for a in ship.action)):
            reward -= 0.5  # Increased from 0.1
        
        return reward
    
    def check_and_update_memories(self):
        """Properly update all ship memories, ensuring terminal states are marked"""
        for ship in self.fleet.ships:
            if not ship.is_active:
                continue
                
            ship_memory = self.memory.ship_memories[ship.unit_id]
            
            # 1. Check for length consistency
            lengths = [
                len(ship_memory.states),
                len(ship_memory.action_types),
                len(ship_memory.detail_actions),
                len(ship_memory.type_logprobs),
                len(ship_memory.detail_logprobs),
                len(ship_memory.values),
                len(ship_memory.rewards),
                len(ship_memory.is_terminals)
            ]
            
            # 2. If lengths are inconsistent
            if len(set(lengths)) > 1:
                print(f"Warning: Inconsistent memory lengths for ship {ship.unit_id}")
                min_length = min(lengths)
                ship_memory.states = ship_memory.states[:min_length]
                ship_memory.action_types = ship_memory.action_types[:min_length]
                ship_memory.detail_actions = ship_memory.detail_actions[:min_length]
                ship_memory.type_logprobs = ship_memory.type_logprobs[:min_length]
                ship_memory.detail_logprobs = ship_memory.detail_logprobs[:min_length]
                ship_memory.values = ship_memory.values[:min_length]
                ship_memory.rewards = ship_memory.rewards[:min_length]
                ship_memory.is_terminals = ship_memory.is_terminals[:min_length]
            
            # 3. Mark terminal states for destroyed ships
            if not ship.is_active and lengths[0] > 0 and not ship_memory.is_terminals[-1]:
                ship_memory.is_terminals[-1] = True
            
            # 4. Update step counts
            ship_memory.steps_collected = len(ship_memory.states)
        
        # 5. Update total fleet memory steps
        self.memory.steps_collected = sum(
            memory.steps_collected 
            for memory in self.memory.ship_memories.values()
        )

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # print("Got into act method")
        try:
            # Space와 Fleet 업데이트 전 이전 상태 저장
            prev_active_ships = {ship.unit_id: ship.is_active for ship in self.fleet.ships}
            
            self.space.update(obs)
            self.fleet.update(obs)
            
            actions = np.zeros((len(self.fleet.ships), 3), dtype=int)
            
            if not self.train_mode:
                with torch.no_grad():
                    for i, ship in enumerate(self.fleet.ships):
                        # 이전에 활성화되어 있었고 현재는 비활성화된 경우 스킵
                        if prev_active_ships.get(ship.unit_id, False) and not ship.is_active:
                            continue
                        if not ship.is_active:
                            continue
                            
                        try:
                            state = self._process_state(obs, ship)
                            final_action = self.ppo.policy.act(state)
                            print("FINAL Actions: ", actions)
                            actions[i] = final_action
                            ship.action = final_action
                        except Exception as e:
                            print(f"Error processing ship {i}: {e}")
                            continue
            else:
                # 학습 모드
                for i, ship in enumerate(self.fleet.ships):
                    # 이전에 활성화되어 있었고 현재는 비활성화된 경우 스킵
                    if prev_active_ships.get(ship.unit_id, False) and not ship.is_active:
                        continue
                    if not ship.is_active:
                        continue
                        
                    try:
                        # 1. 상태 처리 및 행동 선택
                        state = self._process_state(obs, ship)
                        final_action = self.ppo.collect_rollout(
                            state, 
                            self.memory,
                            ship.unit_id,
                            obs,
                            ship
                        )
                        actions[i] = final_action
                        ship.action = final_action
                        
                        # 2. 보상 계산
                        reward = self.calculate_reward(obs, ship)
                        if isinstance(obs, dict):
                            terminated = obs.get("terminated", False)
                        else:
                            terminated = getattr(obs, "terminated", False)
                        
                        # 3. 보상과 종료 상태만 별도로 저장
                        self.memory.push_ship_transition(
                            ship_id=ship.unit_id,
                            state=None,
                            action_type=None,
                            detail_action=None,
                            type_logprob=None,
                            detail_logprob=None,
                            value=None,
                            reward=reward,
                            is_terminal=terminated
                        )
                        
                    except Exception as e:
                        print(f"\nError processing ship {i}:")
                        print(f"Exception: {str(e)}")
                        continue
            
            # 다음 스텝을 위해 점수 기록
            current_points = float(obs["team_points"][self.team_id]) if isinstance(obs, dict) else float(obs.team_points[self.team_id])
            self.prev_team_points = current_points
            
            return actions
            
        except Exception as e:
            print(f"\nUnexpected error in act method:")
            print(f"Exception: {str(e)}")
            return np.zeros((len(self.fleet.ships), 3), dtype=int)