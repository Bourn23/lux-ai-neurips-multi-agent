import json
from argparse import Namespace
from agent import PyTreeAgent  # Import our PyTreeAgent from the agent module
# from lux.kit import from_json   # Adjust this import if needed for your environment
from PPO_agent import Agent
from Borna_agent import BornaAgent

# Define a simple from_json helper (since lux.kit is not available)
def from_json(x):
    if isinstance(x, str):
        return json.loads(x)
    return x

# Dictionary to hold agent instances for each player.
agent_dict = dict()

def agent_fn(observation, configurations):
    """
    Agent function for the kaggle submission.
    """
    global agent_dict
    obs = observation.obs
    if isinstance(obs, str):
        obs = json.loads(obs)
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime

    # Initialize the agent at the first step of the game.
    if step == 0:
        # For submission, we use training=False to load pre-trained model parameters.
        # agent_dict[player] = PyTreeAgent(player, configurations["env_cfg"], training=False)
        agent_dict[player] = Agent(player, configurations["env_cfg"], train_mode=False)

    agent = agent_dict[player]
    # Convert observation using our from_json helper.
    actions = agent.act(step, from_json(obs), remainingOverageTime)
    
    # Convert the NumPy array (or similar) to a list of lists with plain ints.
    actions_list = [[int(x) for x in row] for row in actions.tolist()]

    return dict(action=actions_list)

if __name__ == "__main__":
    def read_input():
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)

    i = 0
    env_cfg = None

    while True:
        raw = read_input()
        raw_input = json.loads(raw)
        step = raw_input["step"]
        obs = raw_input["obs"]
        remaining_time = raw_input["remainingOverageTime"]
        player = raw_input["player"]
        info = raw_input["info"]

        # Initialize the environment configuration on the first step.
        if i == 0:
            env_cfg = info["env_cfg"]
        i += 1

        # Create an observation object.
        observation = Namespace(
            step=step,
            obs=obs,
            remainingOverageTime=remaining_time,
            player=player,
            info=info,
        )
        print("Observation: ", observation)
        # Get the agent's actions.
        actions = agent_fn(observation, dict(env_cfg=env_cfg))

        # Print the actions as JSON.
        print(json.dumps(actions))