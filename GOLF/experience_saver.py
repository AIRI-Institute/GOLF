from schnetpack.data.loader import _atoms_collate_fn

from GOLF.utils import unpad_state


class BaseSaver:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer

    def get_forces(self, indices=None):
        return self.env.get_forces(indices=indices)

    def save(self, states, envs_to_store):
        if len(envs_to_store) > 0:
            energies = self.env.get_energies(indices=envs_to_store)
            forces = self.env.get_forces(indices=envs_to_store)
            state_list = unpad_state(states)
            state_list = [state_list[i] for i in envs_to_store]
            self.replay_buffer.add(_atoms_collate_fn(state_list), forces, energies)


class RewardThresholdSaver(BaseSaver):
    def __init__(self, env, replay_buffer, reward_threshold):
        super().__init__(env, replay_buffer)
        self.reward_threshold = reward_threshold

    def __call__(self, states, rewards, _):
        # Only store states with reward > REWARD_THRESHOLD
        envs_to_store = [
            i for i, reward in enumerate(rewards) if reward > self.reward_threshold
        ]
        super().save(states, envs_to_store)


class LastConformationSaver(BaseSaver):
    def __init__(self, env, replay_buffer, reward_threshold):
        super().__init__(env, replay_buffer)
        self.reward_threshold = reward_threshold

    def __call__(self, states, rewards, dones):
        # Save last states of trajectories (and with reward > REWARD_THRESHOLD)
        envs_to_store = [
            i
            for i, (done, reward) in enumerate(zip(dones, rewards))
            if done and reward > self.reward_threshold
        ]
        super().save(states, envs_to_store)
