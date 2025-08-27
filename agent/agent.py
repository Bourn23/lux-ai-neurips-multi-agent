import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


# -------------------------------------------------
# New GuidedExploreAction Behavior Tree Node
# -------------------------------------------------
class GuidedExploreAction(py_trees.behaviour.Behaviour):
    """
    Instead of choosing a random move, this behavior uses the agent's
    guided exploration callback (which incorporates DQN Q-values and a visited map)
    to return an action.
    """
    def __init__(self, agent, name="GuidedExploreAction"):
        super(GuidedExploreAction, self).__init__(name)
        self.agent = agent

    def update(self):
        bb = blackboard.Blackboard()
        unit_pos = bb.get("unit_position")
        unit_energy = bb.get("unit_energy")
        relic_nodes = bb.get("relic_nodes")
        relic_nodes_mask = bb.get("relic_nodes_mask")
        step = bb.get("step")
        if unit_pos is None or unit_energy is None or step is None:
            action = ((0,0),(0,0))
        else:
            action = self.agent.choose_guided_explore_action(unit_pos, unit_energy, relic_nodes, relic_nodes_mask, step)
        bb.set("action", action)
        return common.Status.SUCCESS
# =============================================================================
# PyTreeAgent Implementation (Integrating Behavior Tree and DQN mechanisms)
# =============================================================================

# -------------------------------------------------
# Updated PyTreeAgent Class (with guided exploration)
# -------------------------------------------------
class PyTreeAgent:
    def __init__(self, player: str, env_cfg, training=False):
        self.player = player
        self.env_cfg = env_cfg
        self.training = training
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id

        # DQN parameters and network initialization (as before)
        self.state_size = 6   # e.g., unit_pos (2) + closest_relic (2) + unit_energy (1) + step (1)
        self.action_size = 6  # 0:stay, 1:up, 2:right, 3:down, 4:left, 5:sap
        self.hidden_size = 128
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # (Assume DQN, target_net, optimizer, and memory are created as in your DQN code.)
        self.policy_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)
        if not self.training:
            self.load_model()
            self.epsilon = 0.0

        # Initialize a visited map for guiding exploration.
        # Note: env_cfg should have "map_width" and "map_height".
        self.visited_map = np.zeros((env_cfg["map_width"], env_cfg["map_height"]), dtype=np.float32)

        # For persisting discovered relic nodes.
        self.discovered_relic_nodes = []

        # Build the behavior tree.
        # Tree structure (Selector):
        #   1. EnemySeq: If enemy visible -> SapEnemyAction.
        #   2. LowEnergySeq: If energy is low -> WaitAction.
        #   3. RelicSeq: If a relic is visible -> MoveToRelicAction.
        #   4. GuidedExploreAction: Otherwise, use our guided exploration (not random).
        self.root = composites.Selector("RootSelector", memory=True)
        
        enemy_seq = composites.Sequence("EnemySeq", memory=False)
        enemy_seq.add_child(EnemyVisibleCondition())
        enemy_seq.add_child(SapEnemyAction())
        
        low_energy_seq = composites.Sequence("LowEnergySeq", memory=False)
        low_energy_seq.add_child(LowEnergyCondition(threshold=50))
        low_energy_seq.add_child(WaitAction())
        
        relic_seq = composites.Sequence("RelicSeq", memory=True)
        relic_seq.add_child(RelicVisibleCondition())
        relic_seq.add_child(MoveToRelicAction())
        
        guided_explore = GuidedExploreAction(agent=self)
        
        self.root.add_child(enemy_seq)
        self.root.add_child(low_energy_seq)
        self.root.add_child(relic_seq)
        self.root.add_child(guided_explore)
        
        self.tree = py_trees.trees.BehaviourTree(self.root)
        self.bb = blackboard.Blackboard()

    def choose_guided_explore_action(self, unit_pos, unit_energy, relic_nodes, relic_nodes_mask, step):
        """
        Uses the DQN to get Q-values for move actions (0 to 4), then applies a penalty
        based on the visited_map. Returns a tuple: ((move_dx, move_dy), (0,0)).
        """
        # Get Q-values from the network for the current state.
        with torch.no_grad():
            state = self._state_representation(unit_pos, unit_energy, relic_nodes, step, relic_nodes_mask)
            q_values = self.policy_net(state).cpu().numpy()  # shape: (action_size,)
        # Consider only movement actions (0 to 4).
        candidate_moves = {
            0: (0, 0),
            1: (0, -1),  # up
            2: (1, 0),   # right
            3: (0, 1),   # down
            4: (-1, 0)   # left
        }
        penalty_weight = 0.1  # Tunable hyperparameter.
        best_action = None
        best_score = -float('inf')
        for action in range(5):
            move = candidate_moves[action]
            new_x = unit_pos[0] + move[0]
            new_y = unit_pos[1] + move[1]
            # Check map boundaries.
            if new_x < 0 or new_x >= self.env_cfg["map_width"] or new_y < 0 or new_y >= self.env_cfg["map_height"]:
                continue
            penalty = penalty_weight * self.visited_map[new_x, new_y]
            score = q_values[action] - penalty
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            best_action = 0
        chosen_move = candidate_moves[best_action]
        return (chosen_move, (0, 0))

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        For each active unit, select an action.
        Each action is a list: [action_type, arg1, arg2].
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
            
            # Update visited_map: increment visit count at this unit's location.
            x, y = int(unit_pos[0]), int(unit_pos[1])
            if 0 <= x < self.env_cfg["map_width"] and 0 <= y < self.env_cfg["map_height"]:
                self.visited_map[x, y] += 1

            # Update blackboard entries.
            self.bb.set("unit_position", unit_pos)
            self.bb.set("unit_energy", unit_energy)
            self.bb.set("relic_nodes", relic_nodes)
            self.bb.set("relic_nodes_mask", relic_nodes_mask)
            self.bb.set("enemy_positions", enemy_positions)
            self.bb.set("enemy_mask", enemy_mask)
            self.bb.set("step", step)
            
            # ε–greedy: sometimes choose a guided random (exploratory) action.
            if self.training and random.random() < self.epsilon:
                if len(self.discovered_relic_nodes) > 0:
                    # Choose the closest discovered relic.
                    distances = [np.linalg.norm(np.array(unit_pos) - np.array(target)) 
                                 for target in self.discovered_relic_nodes]
                    target = self.discovered_relic_nodes[np.argmin(distances)]
                    action_type = direction_to(np.array(unit_pos), np.array(target))
                    chosen_action = [action_type, 0, 0]
                else:
                    # Use the guided exploration strategy.
                    guided_action = self.choose_guided_explore_action(unit_pos, unit_energy, relic_nodes, relic_nodes_mask, step)
                    # Map the chosen move to a discrete direction.
                    chosen_action = [direction_to(np.array(unit_pos),
                                                  np.array([unit_pos[0] + guided_action[0][0],
                                                            unit_pos[1] + guided_action[0][1]])), 0, 0]
            else:
                # Use the behavior tree.
                self.tree.tick()
                bt_action = self.bb.get("action")
                move, sap = bt_action
                if sap != (0, 0):
                    target = (unit_pos[0] + sap[0], unit_pos[1] + sap[1])
                    chosen_action = [5, int(target[0]), int(target[1])]
                else:
                    chosen_action = [direction_to(np.array(unit_pos),
                                                  np.array([unit_pos[0] + move[0], unit_pos[1] + move[1]])), 0, 0]
            actions[unit_id] = chosen_action

            # Record newly discovered relic nodes.
            if relic_nodes_mask.any():
                visible_indices = np.where(relic_nodes_mask)[0]
                for idx in visible_indices:
                    pos = relic_nodes[idx]
                    if pos.tolist() not in [r.tolist() for r in self.discovered_relic_nodes]:
                        self.discovered_relic_nodes.append(pos)
            # (If training, store experience, etc.)
            state = self._state_representation(unit_pos, unit_energy, relic_nodes, step, relic_nodes_mask)
            next_state = state  # In a proper implementation, use the next observation.
            self.memory.push(state, chosen_action[0], self.get_reward(obs), next_state, 0)
        
        # For inactive units, assign a do-nothing action.
        for unit_id, active in enumerate(team_units_mask):
            if not active:
                actions[unit_id] = [0, 0, 0]
        return actions

    def _state_representation(self, unit_pos, unit_energy, relic_nodes, step, relic_mask):
        # If no relic is visible, use [-1, -1] as placeholder.
        if not relic_mask.any():
            closest_relic = np.array([-1, -1])
        else:
            visible_relics = relic_nodes[relic_mask]
            distances = np.linalg.norm(visible_relics - unit_pos, axis=1)
            closest_relic = visible_relics[np.argmin(distances)]
        state = np.concatenate([
            np.array(unit_pos),
            np.array(closest_relic),
            [unit_energy],
            [step / 505.0]  # Normalized step (assuming ~505 steps max)
        ])
        return torch.FloatTensor(state).to(self.device)

    def get_reward(self, obs):
        # A simple reward is the team's relic points.
        return obs["team_points"][self.team_id]

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
        
        current_q_values = self.policy_net(states).gather(1, actions_tensor.unsqueeze(1)).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if step % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
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
        games_to_play=1001,
    )
    # Evaluation phase: play 5 games without training (ε = 0).
    # evaluate_agents(
    #     agent_1_cls=PyTreeAgent,
    #     agent_2_cls=PyTreeAgent,
    #     seed=42,
    #     training=False,
    #     games_to_play=5
    # )