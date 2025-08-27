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


def DQNMovementAction(behaviour.Behaviour):
    def __init__(self, agent):
        super().__init__("DQNMovementAction")
        self.agent = agent

    def update(self):
        # get current state from blackboard
        bb = blackboard.Blackboard()
        state = self.agent._state_representation(
            bb.get('unit_position'),
            bb.get('unit_energy'),
            bb.get('relic_nodes'),
            bb.get('enemy_positions'),
            bb.get('enemy_mask'),
            bb.get('step'),
            bb.get('relic_nodes_mask')
        )

        # get action from DQN
        if self.agent.training and random.random() < self.agent.epsilon:
            action = random.randint(0, 4)
        else:
            with torch.no_grad():
                action = self.agent.policy_net(state.unsqueeze(0)).max(1)[1].item()

        move_directions = {
            0: (0, 0),    # stay
            1: (0, 1),    # up
            2: (1, 0),    # right
            3: (0, -1),   # down
            4: (-1, 0)    # left
        }

        # Set action in blackboard
        bb.set('action', (move_direction[action], (0, 0))) # move, sap

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
        self.state_size = 11   # unit_pos (2) + closest_relic (2) + closest_enemy(2) + 
                                # unit_energy (1) + step (1) + relic_visibility (1) +
                                # enemy_visibility (1) + low_energy (1)
        self.action_size = 5  # actions: 0:stay, 1:up, 2:right, 3:down, 4:left
        self.hidden_size = 256
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
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
        
        enemy_seq = composites.Sequence("EnemySeq", memory=False)
        enemy_seq.add_child(EnemyVisibleCondition())
        enemy_seq.add_child(SapEnemyAction())
        
        low_energy_seq = composites.Sequence("LowEnergySeq", memory=False)
        low_energy_seq.add_child(LowEnergyCondition(threshold=50))
        low_energy_seq.add_child(WaitAction())
        
        relic_seq = composites.Sequence("RelicSeq", memory=False)
        relic_seq.add_child(RelicVisibleCondition())
        relic_seq.add_child(MoveToRelicAction())
        
        explore = ExploreAction()
        
        self.root.add_child(enemy_seq)
        self.root.add_child(low_energy_seq)
        self.root.add_child(relic_seq)
        self.root.add_child(explore)
        
        self.tree = py_trees.trees.BehaviourTree(self.root)
        self.bb = blackboard.Blackboard()
        
        # For remembering discovered relic nodes (persist across matches)
        self.discovered_relic_nodes = []

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

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Given the observation for this team, return a dict mapping unit_id to actions.
        Each action is represented as a list [action_type, arg1, arg2] where:
         - For move actions, action_type is in {0:stay, 1:up, 2:right, 3:down, 4:left}
         - For sap actions, action_type is 5 and arg1,arg2 denote target coordinates.
        """
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        team_units_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        relic_nodes = np.array(obs.get("relic_nodes", []))
        relic_nodes_mask = np.array(obs.get("relic_nodes_mask", []))
        # Opponent units info.
        enemy_positions = np.array(obs["units"]["position"][self.opp_team_id])
        enemy_mask = np.array(obs["units_mask"][self.opp_team_id])
        
        available_units = np.where(team_units_mask)[0]
        for unit_id in available_units:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            
            # Update blackboard for behavior tree.
            self.bb.set("unit_position", unit_pos)
            self.bb.set("unit_energy", unit_energy)
            self.bb.set("relic_nodes", relic_nodes)
            self.bb.set("relic_nodes_mask", relic_nodes_mask)
            self.bb.set("enemy_positions", enemy_positions)
            self.bb.set("enemy_mask", enemy_mask)
            
            # ε–greedy: sometimes choose a random action.
            if self.training and random.random() < self.epsilon:
                # Random move: if any relic discovered, move toward it; else random exploration.
                if len(self.discovered_relic_nodes) > 0:
                    target = self.discovered_relic_nodes[0]
                    action_type = direction_to(np.array(unit_pos), np.array(target))
                    chosen_action = [action_type, 0, 0]
                else:
                    chosen_action = [random.randint(0, 4), 0, 0]
            else:
                # Tick the behavior tree.
                self.tree.tick()
                # Retrieve action from blackboard.
                bt_action = self.bb.get("action")
                # bt_action is a tuple: ((move_dx, move_dy), (sap_dx, sap_dy))
                move, sap = bt_action
                # Decide the final action based on the returned tuple.
                if sap != (0, 0):
                    # Sap action: we encode action type 5 and provide target coordinates.
                    # For target, we add the sap offset to the unit position.
                    target = (unit_pos[0] + sap[0], unit_pos[1] + sap[1])
                    chosen_action = [5, target[0], target[1]]
                else:
                    # Otherwise, move action.
                    # Map (dx,dy) to a direction (0:stay, 1:up, 2:right, 3:down, 4:left).
                    chosen_action = [direction_to(np.array(unit_pos), np.array([unit_pos[0]+move[0], unit_pos[1]+move[1]])), 0, 0]
            actions[unit_id] = chosen_action

            # If a new relic is visible, record it for future matches.
            if relic_nodes_mask.any():
                visible_indices = np.where(relic_nodes_mask)[0]
                for idx in visible_indices:
                    pos = relic_nodes[idx]
                    if pos.tolist() not in [r.tolist() for r in self.discovered_relic_nodes]:
                        self.discovered_relic_nodes.append(pos)
            
            # (If training, store experience in replay buffer.)
            state = self._state_representation(unit_pos, unit_energy, relic_nodes, step, relic_nodes_mask)
            # For next_state we assume the same (a proper implementation would wait for the next observation).
            next_state = state  
            # For action storage, we convert the chosen action to a scalar (e.g., action_type).
            # Here we use the first element as the action.
            self.memory.push(state, chosen_action[0], self.get_reward(obs), next_state, 0)
            
        # For inactive units, set action to do nothing.
        for unit_id, active in enumerate(team_units_mask):
            if not active:
                actions[unit_id] = [0, 0, 0]
        return actions

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