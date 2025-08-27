from typing import Any, Dict

class Agent:
    def __init__(self, player: str, env_cfg: Dict[str, Any]) -> None:
        self.player = player
        self.team_id = 0 if player == 'player_0' else 1
        self.opp_team_id = 1 - self.team_id
        np.random.seed(0)

        # Extract environment parameters safely with default
        self.env_cfg = env_cfg
        self.map_width = env_cfg.get('map_width', 24)
        self.map_height = env_cfg.get('map_height', 24)
        self.max_units = env_cfg.get('max_units', 16)
        self.max_steps_in_match = env_cfg.get('max_steps', 100)

        # Costs and parameters extracted or defaulted
        self.move_cost = env_cfg.get('unit_move_cost', 2)
        self.sap_cost = env_cfg.get('unit_sap_cost', 30)
        self.sap_range = env_cfg.get('unit_sap_range', 4)
        self.sr = env_cfg.get('unit_sensor_range', 2)
        self.nebula_vision_red = env_cfg.get('nebula_tile_vision_reduction', 0)
        self.nebula_energy_red = env_cfg.get('nebula_tile_energy_reduction', 0)
        self.energy_void_factor = env_cfg.get('unit_energy_void_factor', 0.125)
        self.sap_dropoff_factor = env_cfg.get('unit_sap_dropoff_factor', 0.5)

        # Track discovered relic nodes and scoring patterns
        self.discovered_relic_nodes_ids = set()
        self.relic_node_positions = []

        ## rest of the code from : https://www.kaggle.com/code/stephenmurph/neurips-heuristic-logics-added

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Main decision function called every turn.
        Reads current step and observations, updates internal knowledge.
        and decides actions for all units.
        """

        self.global_step = step
        match_id = step // self.max_steps_in_match

        # Increment match count when a new match starts
        if step % self.max_steps_in_match == 0 and step > 0:
            self.match_count += 1

        unit_mask = np.array(obs['unit_mask'][self.team_id])
        unit_positions = np.array(obs['units']['position'][self.team_id])
        unit_energys = np.array(obs['units']['energy'][self.team_id])

        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.max_units, 3), dtype=int)
        known_relics = self.relic_node_positions






