from AL.experience_saver import RewardThresholdSaver, InitialAndLastSaver
from utils.utils import ignore_extra_args


savers = {
    "reward_threshold": ignore_extra_args(RewardThresholdSaver),
    "initial_and_last": ignore_extra_args(InitialAndLastSaver),
}


def make_saver(args, env, replay_buffer, reward_thresh):
    if args.reward == "dft":
        thresh = reward_thresh / 627.5
    else:
        thresh = reward_thresh

    return savers[args.experience_saver](
        env=env, replay_buffer=replay_buffer, reward_threshold=thresh
    )
