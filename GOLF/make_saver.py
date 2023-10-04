from GOLF.experience_saver import (
    RewardThresholdSaver,
    LastConformationSaver,
)
from utils.utils import ignore_extra_args


savers = {
    "reward_threshold": ignore_extra_args(RewardThresholdSaver),
    "last": ignore_extra_args(LastConformationSaver),
}


def make_saver(args, env, replay_buffer, actor, reward_thresh):
    if args.reward == "dft":
        thresh = reward_thresh / 627.5
    else:
        thresh = reward_thresh

    return savers[args.experience_saver](
        env=env,
        replay_buffer=replay_buffer,
        reward_threshold=thresh,
        actor=actor,
    )
