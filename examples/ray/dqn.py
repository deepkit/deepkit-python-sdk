import deepkit
import ray
from ray.rllib.agents import dqn

# note: ray overwrites sys.path[0], Dunno why, but that breaks deepkit looking for the project link
experiment = deepkit.experiment(account='localhost', project='deepkit-python-sdk')

# Initialize Ray with host that makes docker happy
ray.init(webui_host='127.0.0.1')

# Initialize DQN Trainer with default config and built-in gym cart-pole environment.
trainer = dqn.DQNTrainer(config=dqn.DEFAULT_CONFIG, env="CartPole-v0")

# Extract several layers of models
ray_policy = trainer.get_policy()
ray_model = ray_policy.model
# This is the one I think we should "watch"
keras_model = ray_model.base_model

experiment.watch_keras_model(keras_model)

experiment.log('lets go')

# Manually train for a couple of iterations
for i in range(20):
    result = trainer.train()

experiment.log('Done')
