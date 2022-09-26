import torch
import torch.nn as nn
import torch.optim as optim

from rl.utils import calculate_gradient_norm


class PPO():
    def __init__(self,
                 policy,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=1e-5,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = policy

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(policy.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        metrics = dict()
        advantages = rollouts.returns[:-1] - rollouts.values
        advantages = (advantages - advantages.mean()) / (
           advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        policy_grad_norm_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)
            i = 0
            for sample in data_generator:
                state_batch, actions_batch, value_preds_batch, \
                return_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs = self.actor_critic.evaluate_actions(state_batch, actions_batch)

                dist_entropy = - action_log_probs.mean()

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                self.optimizer.zero_grad()
                loss.backward()
                policy_grad_norm_epoch += calculate_gradient_norm(self.actor_critic).item()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                i += 1

        num_updates = self.ppo_epoch * self.num_mini_batch

        metrics.update({
            "critic_loss": value_loss_epoch / num_updates,
            "actor_loss": action_loss_epoch / num_updates,
            "actor_entropy": dist_entropy_epoch / num_updates,
            "policy_grad_norm": policy_grad_norm_epoch / num_updates
        })

        return metrics