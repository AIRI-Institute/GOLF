from torch_geometric.data import Batch


class BaseSaver:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer

    def get_forces(self, indices=None):
        return self.env.get_forces(indices=indices)

    def save(self, batch, envs_to_store):
        if len(envs_to_store) > 0:
            energies = self.env.get_energies(indices=envs_to_store)
            forces = self.env.get_forces(indices=envs_to_store)
            state_list = batch.to_data_list()
            state_list = [state_list[i] for i in envs_to_store]
            self.replay_buffer.add(Batch.from_data_list(state_list), forces, energies)


class RewardThresholdSaver(BaseSaver):
    def __init__(self, env, replay_buffer, reward_threshold):
        super().__init__(env, replay_buffer)
        self.reward_threshold = reward_threshold

    def __call__(self, batch, rewards, _):
        # Only store states with reward > REWARD_THRESHOLD
        envs_to_store = [
            i for i, reward in enumerate(rewards) if reward > self.reward_threshold
        ]
        super().save(batch, envs_to_store)


class LastConformationSaver(BaseSaver):
    def __init__(self, env, replay_buffer, reward_threshold):
        super().__init__(env, replay_buffer)
        self.reward_threshold = reward_threshold

    def __call__(self, batch, rewards, dones):
        # Save last states of trajectories (and with reward > REWARD_THRESHOLD)
        envs_to_store = [
            i
            for i, (done, reward) in enumerate(zip(dones, rewards))
            if done and reward > self.reward_threshold
        ]
        super().save(batch, envs_to_store)
