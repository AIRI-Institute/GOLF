from math import floor

class ActionScaleScheduler():
    def __init__(self,  action_scale_init, action_scale_end, n_step_end, mode="discrete"):
        self.as_init = action_scale_init
        self.as_end = action_scale_end
        self.n_step_end = n_step_end

        assert mode in ["constant", "discrete", "continuous"], "Unknown ActionScaleSheduler mode!"
        self.mode  = mode
        # For discrete mode
        if mode == "discrete":
            n_updates = (self.as_end - self.as_init) / 0.01
            self.update_interval = self.n_step_end / n_updates

    def __call__(self, n_step):
        if self.mode == "constant":
            current_action_scale = self.as_init
        elif self.mode == "discrete":
            current_action_scale = self.as_init + floor(n_step / self.update_interval) * 0.01
        else:
            p = max((self.n_step_end - n_step) / self.n_step_end, 0)
            current_action_scale = p * (self.as_init - self.as_end) + self.as_end
        return current_action_scale
