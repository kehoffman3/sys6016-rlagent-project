# Getting Started

1. Create a new virtual environment: `python -m venv env`
2. Activate the env: `source env/bin/activate`
3. Install packages: `pip install -r requirements.txt`

# Running the DQN Agent
 - To run the agent with the default settings, run `python dqn_agent.py`. Docker must be running.
 - Below are the following commandline flags the script accepts:
 - `--silent`:Whether or not the graphical interface should be displayed.
 - `--episode_length_seconds=360`: Length of epsidoe
 - `--learning_rate=.0001`: Learning rate of the network
 - `--epsilon=.1` Epsilon of the network)
 - `--discount=.99` Discount of the network)
