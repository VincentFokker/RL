import gym

class CustomEnvironment(gym.Env):
	def __init__(self, config=None, **kwargs):
		"""
		        A simple grid environment where the agent has to navigate towards a goal
		        Must be a subclass of gym.Env
		        """
		self.config = config['environment']

		# Grid, positions
		self.grid_size = self.config['grid_size']
		self.grid = np.zeros([self.grid_size, self.grid_size])
		self.agent_start_location = [self.grid_size // 2, self.grid_size // 2]  # Start at the middle of the grid
		self.position = self.agent_start_location
		self.goal_position = []
		self.window_name = 'Test env'

		# Gym-related part
		self.r = 0  # Total episode reward
		self.done = False  # Termination
		self.episode = 0  # Episode number
		self.steps = 0  # Current step in the episode
		self.max_steps = self.config['step_limit']
		self.goals_reached = 0

		self.create_window()

		# Action and observation spaces
		self.action_space = gym.spaces.Discrete(4)

		if self.config['image_as_state']:  # Image based (CNN)
			self.observation_space = gym.spaces.Box(shape=(self.grid_size, self.grid_size, 1), high=1, low=0,
													dtype=np.uint8)
		else:  # Vector based (MLP)
			self.observation_space = gym.spaces.Box(shape=(4,), high=10, low=0, dtype=np.uint8)

	def make_observation(self):
		"""
        Return the environment's current state
        """
		self.position = list(np.clip(self.position, 0, self.grid_size - 1))
		# Image based (uncomment when using an image based observation space)
		if self.config['show_training']:
			self.render()
		else:
			self.render(mode='rgb_array')

		if self.config['image_as_state']:
			state = np.expand_dims(self.grid, axis=2)

		# Vector based
		else:
			state = np.array([self.position + self.goal_position], dtype=np.uint8).squeeze()
		return state

	def reset(self):  (3)
		return ...

	def step(self, action): (4)
		return ...

	def render():
		return ... (5)