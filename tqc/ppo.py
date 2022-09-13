import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqc.utils import quantile_huber_loss_f, calculate_gradient_norm


class PPO():
    def __init__(self,
            *,
            actor,
            critic,
            clip_param,
            value_loss_coef,
            entropy_coef,
            actor_lr,
            critic_lr,
            max_grad_norm=None,
            use_clipped_value_loss=True,
            **kwargs):

        self.actor = actor
        self.critic = critic

        self.clip_param = clip_param

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    def evaluate_actions(self, inputs, action):
        value = self.critic(inputs, action)
        self.actor(inputs)
        dist = self.actor.scaled_normal
        if '_atoms_mask' not in inputs:
            atoms_mask = torch.ones(inputs['_positions'].shape[:2]).to(DEVICE)
        else:
            atoms_mask = inputs['_atoms_mask']
        log_probs = dist.log_prob(action)
        log_probs *= atoms_mask[..., None]
        log_probs = log_probs.sum(dim=(1, 2)).unsqueeze(-1)
        dist_entropy = dist.entropy().sum(dim=(1, 2)).mean()
        return value, log_probs, dist_entropy
    
    def update(self, buffer, batch_size=256):
        metrics = dict()
        advantages = buffer.returns - buffer.values
        adv_mean, adv_std = advantages.mean(), advantages.std()

        obs_batch, actions_batch, \
            _, _, _, old_action_log_probs_batch, value_preds_batch, \
                return_batch = buffer.sample(batch_size)
        
        adv_targ = return_batch - value_preds_batch
        adv_targ = (adv_targ - adv_mean) / (adv_std + 1e-5)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.evaluate_actions(
            obs_batch, actions_batch)

        ratio = torch.exp(action_log_probs -
                            old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (
                value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses,
                                            value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
            dist_entropy * self.entropy_coef).backward()
        metrics['critic_grad_norm'] = calculate_gradient_norm(self.critic).item()
        metrics['actor_grad_norm'] = calculate_gradient_norm(self.actor).item()
        nn.utils.clip_grad_norm_(self.critic.parameters(),
                                    self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor.parameters(),
                                    self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        metrics['actor_entropy'] = - action_log_probs.mean().item()
        metrics['value_loss'] = value_loss.item()
        metrics['action_loss'] = action_loss.item()
        metrics['dist_entropy'] = dist_entropy.item()
        return metrics
