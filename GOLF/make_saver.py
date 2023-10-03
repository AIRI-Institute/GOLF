from AL.experience_saver import (
    RewardThresholdSaver,
    InitialAndLastSaver,
    GradientMissmatchSaver,
)
from utils.utils import ignore_extra_args


savers = {
    "reward_threshold": ignore_extra_args(RewardThresholdSaver),
    "initial_and_last": ignore_extra_args(InitialAndLastSaver),
    "gradient_missmatch": ignore_extra_args(GradientMissmatchSaver),
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
        gradient_dif_threshold=args.grad_missmatch_threshold,
    )
