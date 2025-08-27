import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

import py_trees
from py_trees import common, composites, behaviours, blackboard

# =============================================================================
# DQN Network and Replay Buffer (from the DQN agent)
# =============================================================================

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# =============================================================================
# Helper Function: direction_to (for generating move directions)
# =============================================================================

def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0  # stay
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 4
    else:
        return 3 if dy > 0 else 1

# =============================================================================
# Behavior Tree Nodes
# =============================================================================

class LowEnergyCondition(py_trees.behaviour.Behaviour):
    """
    Succeeds if the unit's energy is below a threshold.
    """
    def __init__(self, threshold=50, name="LowEnergyCondition"):
        super(LowEnergyCondition, self).__init__(name)
        self.threshold = threshold

    def update(self):
        bb = blackboard.Blackboard()
        unit_energy = bb.get("unit_energy")
        if unit_energy is None:
            return common.Status.FAILURE
        return common.Status.SUCCESS if unit_energy < self.threshold else common.Status.FAILURE

class WaitAction(py_trees.behaviour.Behaviour):
    """
    When low on energy, the unit takes a wait (no–move, no–sap) action.
    """
    def __init__(self, name="WaitAction"):
        super(WaitAction, self).__init__(name)
        self.action = ((0, 0), (0, 0))  # (move, sap)

    def update(self):
        bb = blackboard.Blackboard()
        bb.set("action", self.action)
        return common.Status.SUCCESS

class RelicVisibleCondition(py_trees.behaviour.Behaviour):
    """
    Succeeds if any relic node is visible.
    Expects on the blackboard:
      - relic_nodes: np.array of relic positions, shape (N,2)
      - relic_nodes_mask: boolean np.array of length N.
    """
    def __init__(self, name="RelicVisibleCondition"):
        super(RelicVisibleCondition, self).__init__(name)

    def update(self):
        bb = blackboard.Blackboard()
        relics = bb.get("relic_nodes")
        relic_mask = bb.get("relic_nodes_mask")
        if relics is None or relic_mask is None:
            return common.Status.FAILURE
        return common.Status.SUCCESS if np.any(relic_mask) else common.Status.FAILURE

class MoveToRelicAction(py_trees.behaviour.Behaviour):
    """
    Chooses a move action toward the closest visible relic.
    Sets the action on the blackboard as ((dx, dy), (0,0)).
    """
    def __init__(self, name="MoveToRelicAction"):
        super(MoveToRelicAction, self).__init__(name)
        self.action = ((0, 0), (0, 0))

    def update(self):
        bb = blackboard.Blackboard()
        relics = bb.get("relic_nodes")
        relic_mask = bb.get("relic_nodes_mask")
        unit_pos = bb.get("unit_position")
        if relics is None or relic_mask is None or unit_pos is None:
            return common.Status.FAILURE
        # Filter visible relics.
        visible_relics = relics[relic_mask]
        if len(visible_relics) == 0:
            return common.Status.FAILURE
        # Use Manhattan distance.
        def manhattan(p1, p2):
            return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
        nearest = min(visible_relics, key=lambda r: manhattan(np.array(unit_pos), np.array(r)))
        dx = nearest[0] - unit_pos[0]
        dy = nearest[1] - unit_pos[1]
        move_dx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        move_dy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        self.action = ((move_dx, move_dy), (0, 0))
        bb.set("action", self.action)
        return common.Status.SUCCESS

class ExploreAction(py_trees.behaviour.Behaviour):
    """
    Default action: choose a random move (exploration).
    """
    def __init__(self, name="ExploreAction"):
        super(ExploreAction, self).__init__(name)
        self.possible_moves = [(0,0), (0,-1), (1,0), (0,1), (-1,0)]

    def update(self):
        move = random.choice(self.possible_moves)
        action = (move, (0, 0))
        bb = blackboard.Blackboard()
        bb.set("action", action)
        return common.Status.SUCCESS

class EnemyVisibleCondition(py_trees.behaviour.Behaviour):
    """
    Succeeds if at least one enemy unit is visible.
    Expects on the blackboard:
      - enemy_positions: np.array of enemy unit positions, shape (M,2)
      - enemy_mask: boolean np.array of length M.
    """
    def __init__(self, name="EnemyVisibleCondition"):
        super(EnemyVisibleCondition, self).__init__(name)

    def update(self):
        bb = blackboard.Blackboard()
        enemy_positions = bb.get("enemy_positions")
        enemy_mask = bb.get("enemy_mask")
        if enemy_positions is None or enemy_mask is None:
            return common.Status.FAILURE
        return common.Status.SUCCESS if np.any(enemy_mask) else common.Status.FAILURE

class SapEnemyAction(py_trees.behaviour.Behaviour):
    """
    Chooses a sap action toward the first visible enemy.
    The action is encoded as:
        move: (0,0)   and  sap: (dx, dy) = difference between enemy position and unit position.
    """
    def __init__(self, name="SapEnemyAction"):
        super(SapEnemyAction, self).__init__(name)

    def update(self):
        bb = blackboard.Blackboard()
        enemy_positions = bb.get("enemy_positions")
        enemy_mask = bb.get("enemy_mask")
        unit_pos = bb.get("unit_position")
        if enemy_positions is None or enemy_mask is None or unit_pos is None:
            return common.Status.FAILURE
        # Choose the first visible enemy.
        visible_indices = np.where(enemy_mask)[0]
        if len(visible_indices) == 0:
            return common.Status.FAILURE
        target_pos = enemy_positions[visible_indices[0]]
        sap_dx = target_pos[0] - unit_pos[0]
        sap_dy = target_pos[1] - unit_pos[1]
        action = ((0, 0), (sap_dx, sap_dy))
        bb.set("action", action)
        return common.Status.SUCCESS

class DQNMoveAction(py_trees.behaviour.Behaviour):
    """
    Uses DQN to choose movement actions when relics are visible.
    """
    def __init__(self, agent, name="DQNMoveAction"):
        super(DQNMoveAction, self).__init__(name)
        self.agent = agent

    def initialise(self):
        """
        Set default action when the behavior is first initialized.
        """
        bb = blackboard.Blackboard()
        bb.set("action", ((0, 0), (0, 0)))

    def update(self):
        bb = blackboard.Blackboard()
        unit_pos = bb.get("unit_position")
        unit_energy = bb.get("unit_energy")
        relic_nodes = bb.get("relic_nodes")
        relic_mask = bb.get("relic_nodes_mask")
        enemy_positions = bb.get("enemy_positions")
        enemy_mask = bb.get("enemy_mask")
        step = bb.get("step")
        
        if any(x is None for x in [unit_pos, unit_energy, relic_nodes, relic_mask, enemy_positions, enemy_mask, step]):
            bb.set("action", ((0, 0), (0, 0)))  # Set default action on failure
            return common.Status.FAILURE
        
        state = self.agent._state_representation(
            unit_pos, unit_energy, relic_nodes, enemy_positions, step, relic_mask, enemy_mask
        )
        
        with torch.no_grad():
            q_values = self.agent.policy_net(state.unsqueeze(0))
            if random.random() < self.agent.epsilon and self.agent.training:
                action = random.randint(0, 4)
            else:
                action = q_values.max(1)[1].item()
        
        # Convert action to movement direction
        moves = [(0,0), (0,-1), (1,0), (0,1), (-1,0)]
        move = moves[action]
        
        bb.set("action", (move, (0, 0)))
        return common.Status.SUCCESS
# =============================================================================
# PyTreeAgent Implementation (Integrating Behavior Tree and DQN mechanisms)
# =============================================================================

class PyTreeAgent:
    def __init__(self, player: str, env_cfg, training=False):
        self.player = player
        self.env_cfg = env_cfg
        self.training = training
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id

        # DQN parameters
        self.state_size = 12  # unit_pos (2) + closest_relic (2) + closest_enemy (2) + 
                             # unit_energy (1) + step (1) + relic_visibility (1) +
                             # enemy_visibility (1) + distance_to_closest_relic (1) +
                             # distance_to_closest_enemy (1) + energy_ratio (1) +
                             # num_visible_relics (1) + num_visible_enemies (1)
        self.action_size = 5  # move actions: stay, up, right, down, left
        self.hidden_size = 512  # Increased network capacity
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        self.learning_rate = 0.0003

        class EnhancedDQN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(EnhancedDQN, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, output_size)
                )
            
            def forward(self, x):
                # print("Shape of input ", x.shape)
                return self.network(x)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.policy_net = EnhancedDQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net = EnhancedDQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(50000)

        if not self.training:
            self.load_model()
            self.epsilon = 0.0

        # Build behavior tree.
        # Tree structure (Selector):
        #   1. Sequence: EnemyVisibleCondition -> SapEnemyAction
        #   2. Sequence: LowEnergyCondition -> WaitAction
        #   3. Sequence: RelicVisibleCondition -> MoveToRelicAction
        #   4. ExploreAction (default)
        self.root = composites.Selector("RootSelector", memory=False)
        
        # Priority 1: Handle enemies if visible
        enemy_seq = composites.Sequence("EnemySeq", memory=False)
        enemy_seq.add_child(EnemyVisibleCondition())
        enemy_seq.add_child(SapEnemyAction())
        
        # Priority 2: Handle low level energy situation
        low_energy_seq = composites.Sequence("LowEnergySeq", memory=False)
        low_energy_seq.add_child(LowEnergyCondition(threshold=50))
        low_energy_seq.add_child(WaitAction())
        
        # Priority 3: Use DQN for movement decisions
        dqn_seq = composites.Sequence("DQNSeq", memory=False)
        dqn_seq.add_child(RelicVisibleCondition())
        dqn_seq.add_child(DQNMoveAction(self))
        
        self.root.add_child(enemy_seq)
        self.root.add_child(low_energy_seq)
        self.root.add_child(dqn_seq)

        
        self.tree = py_trees.trees.BehaviourTree(self.root)
        self.bb = blackboard.Blackboard()
        
        # For remembering discovered relic nodes (persist across matches)
        self.discovered_relic_nodes = []

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Main action selection function that coordinates between the behavior tree and DQN.
        """
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        team_units_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        relic_nodes = np.array(obs.get("relic_nodes", []))
        relic_nodes_mask = np.array(obs.get("relic_nodes_mask", []))
        enemy_positions = np.array(obs["units"]["position"][self.opp_team_id])
        enemy_mask = np.array(obs["units_mask"][self.opp_team_id])
        
        available_units = np.where(team_units_mask)[0]
        for unit_id in available_units:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            
            # Set default action before behavior tree execution
            self.bb.set("action", ((0, 0), (0, 0)))  # Default to "stay in place"
            
            # Update blackboard with current state
            self.bb.set("unit_position", unit_pos)
            self.bb.set("unit_energy", unit_energy)
            self.bb.set("relic_nodes", relic_nodes)
            self.bb.set("relic_nodes_mask", relic_nodes_mask)
            self.bb.set("enemy_positions", enemy_positions)
            self.bb.set("enemy_mask", enemy_mask)
            self.bb.set("step", step)
            
            # Get current state representation for DQN
            current_state = self._state_representation(
                unit_pos, 
                unit_energy, 
                relic_nodes, 
                enemy_positions, 
                step, 
                relic_nodes_mask, 
                enemy_mask
            )
            
            # Run behavior tree
            self.tree.tick()
            
            # Get action from blackboard (will use default if none was set)
            bt_action = self.bb.get("action")
            
            # Convert behavior tree action to game action format
            move, sap = bt_action
            if sap != (0, 0):
                # Sap action: targeting enemy
                target = (unit_pos[0] + sap[0], unit_pos[1] + sap[1])
                chosen_action = [5, target[0], target[1]]
            else:
                # Move action from either DQN or behavior tree
                move_dx, move_dy = move
                direction = direction_to(
                    np.array(unit_pos), 
                    np.array([unit_pos[0] + move_dx, unit_pos[1] + move_dy])
                )
                chosen_action = [direction, 0, 0]
            
            actions[unit_id] = chosen_action
            
            # Store experience in replay buffer if training
            if self.training:
                next_state = current_state  # In a proper implementation, this would be the actual next state
                reward = self.get_reward(obs, unit_pos, relic_nodes, relic_nodes_mask)
                action_for_memory = chosen_action[0] if chosen_action[0] <= 4 else 4
                self.memory.push(
                    current_state,
                    action_for_memory,
                    reward,
                    next_state,
                    0
                )
            
            # Update discovered relic nodes
            if relic_nodes_mask.any():
                visible_indices = np.where(relic_nodes_mask)[0]
                for idx in visible_indices:
                    pos = relic_nodes[idx]
                    if pos.tolist() not in [r.tolist() for r in self.discovered_relic_nodes]:
                        self.discovered_relic_nodes.append(pos)
        
        # Set no-op action for inactive units
        inactive_units = np.where(~team_units_mask)[0]
        for unit_id in inactive_units:
            actions[unit_id] = [0, 0, 0]
        
        return actions

    def _state_representation(self, unit_pos, unit_energy, relic_nodes, enemy_positions, step, relic_mask, enemy_mask):
        state_features = []
        unit_pos = np.array(unit_pos)
        
        # Basic position and energy
        state_features.extend(unit_pos)
        state_features.append(unit_energy / 100.0)  # Normalize energy
        state_features.append(step / 505.0)  # Normalize step
        
        # Relic features
        visible_relics = relic_nodes[relic_mask] if relic_mask.any() else np.array([])
        if len(visible_relics) > 0:
            distances = np.linalg.norm(visible_relics - unit_pos, axis=1)
            closest_relic = visible_relics[np.argmin(distances)]
            min_distance = np.min(distances)
            state_features.extend(closest_relic)
            state_features.append(min_distance / 100.0)  # Normalize distance
            state_features.append(len(visible_relics) / len(relic_nodes))  # Ratio of visible relics
        else:
            state_features.extend([-1, -1, 1.0, 0.0])  # No relics visible
        
        # Enemy features
        visible_enemies = enemy_positions[enemy_mask] if enemy_mask.any() else np.array([])
        if len(visible_enemies) > 0:
            distances = np.linalg.norm(visible_enemies - unit_pos, axis=1)
            closest_enemy = visible_enemies[np.argmin(distances)]
            min_distance = np.min(distances)
            state_features.extend(closest_enemy)
            state_features.append(min_distance / 100.0)
            state_features.append(len(visible_enemies) / len(enemy_positions))
        else:
            state_features.extend([-1, -1, 1.0, 0.0])
        
        return torch.FloatTensor(state_features).to(self.device)

    def get_reward(self, obs, unit_pos, relic_nodes, relic_mask):
        reward = obs["team_points"][self.team_id]
        
        # Additional reward components
        if relic_mask.any():
            visible_relics = relic_nodes[relic_mask]
            min_distance = np.min(np.linalg.norm(visible_relics - unit_pos, axis=1))
            proximity_reward = 1.0 / (1.0 + min_distance)  # Reward for being close to relics
            reward += proximity_reward
        
        return reward

    def learn(self, step, last_obs, actions, obs, rewards, dones):
        if not self.training or len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, acts, rews, next_states, done_flags = zip(*batch)
        
        states = torch.stack(states)
        actions_tensor = torch.LongTensor(acts).to(self.device)
        rewards_tensor = torch.FloatTensor(rews).to(self.device)
        next_states = torch.stack(next_states)
        dones_tensor = torch.FloatTensor(done_flags).to(self.device)
        
        # Double DQN implementation
        with torch.no_grad():
            next_action_values = self.policy_net(next_states)
            next_actions = next_action_values.max(1)[1].unsqueeze(1)
            next_state_values = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_state_values
        
        current_q_values = self.policy_net(states).gather(1, actions_tensor.unsqueeze(1)).squeeze()
        
        # Huber loss for more stable training
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update of target network
        if step % 10 == 0:
            with torch.no_grad():
                for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(0.001 * policy_param.data + 0.999 * target_param.data)
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f'dqn_model_{self.player}.pth')

    def load_model(self):
        try:
            checkpoint = torch.load(f'dqn_model_{self.player}.pth', map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            raise FileNotFoundError(f"No trained model found for {self.player}")

    # A stub state representation if needed by external code.
    def _state_representation_stub(self, *args, **kwargs):
        return None

# =============================================================================
# Environment Interface Code (Using the Provided Wrapper)
# =============================================================================

from luxai_s3.wrappers import LuxAIS3GymEnv

def evaluate_agents(agent_1_cls, agent_2_cls, seed=42, training=True, games_to_play=3):
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=seed)
    env_cfg = info["params"]

    player_0 = agent_1_cls("player_0", info["params"], training=training)
    player_1 = agent_2_cls("player_1", info["params"], training=training)

    for i in range(games_to_play):
        obs, info = env.reset()
        game_done = False
        step = 0
        last_obs = None
        last_actions = None
        print(f"Starting game {i}")
        while not game_done:
            actions = {}
            if training:
                last_obs = {
                    "player_0": obs["player_0"].copy(),
                    "player_1": obs["player_1"].copy()
                }
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
            if training:
                last_actions = actions.copy()
            obs, rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] or truncated[k] for k in terminated}
            rewards = {
                "player_0": obs["player_0"]["team_points"][player_0.team_id],
                "player_1": obs["player_1"]["team_points"][player_1.team_id]
            }
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            if training:
                player_0.learn(step, last_obs["player_0"], actions["player_0"], obs["player_0"], rewards["player_0"], dones["player_0"])
                player_1.learn(step, last_obs["player_1"], actions["player_1"], obs["player_1"], rewards["player_1"], dones["player_1"])
            step += 1
        if training:
            player_0.save_model()
            player_1.save_model()
    env.close()
    if training:
        player_0.save_model()
        player_1.save_model()

# =============================================================================
# Run Training and Evaluation
# =============================================================================

if __name__ == "__main__":
    # Training phase: play 10 games.
    evaluate_agents(
        agent_1_cls=PyTreeAgent,
        agent_2_cls=PyTreeAgent,
        seed=42,
        training=True,
        games_to_play=10
    )
    # Evaluation phase: play 5 games without training (ε = 0).
    evaluate_agents(
        agent_1_cls=PyTreeAgent,
        agent_2_cls=PyTreeAgent,
        seed=42,
        training=False,
        games_to_play=5
    )