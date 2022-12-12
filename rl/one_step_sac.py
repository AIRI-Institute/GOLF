import torch
import torch.nn.functional as F

from rl import DEVICE
from rl.utils import calculate_gradient_norm


class OneStepSAC(object):
	def __init__(
		self,
		policy,
		discount,
		log_alpha,
		actor_lr,
		critic_lr,
		alpha_lr,
		per_atom_target_entropy,
		batch_size=256,
		actor_clip_value=None,
		critic_clip_value=None,
	):
		self.actor = policy.actor
		self.critic = policy.critic
		self.log_alpha = torch.tensor([log_alpha], requires_grad=True, device=DEVICE)

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

		self.discount = discount
		self.per_atom_target_entropy = per_atom_target_entropy
		self.batch_size = batch_size
		self.actor_clip_value = actor_clip_value
		self.critic_clip_value = critic_clip_value

		self.total_it = 0

	def update(self, replay_buffer, update_actor, _):
		metrics = dict()
		state, action, _, reward, _ = replay_buffer.sample(self.batch_size)
		alpha = torch.exp(self.log_alpha)
		metrics['alpha'] = alpha.item()

		# --- Critic loss ---
		cur_Q = self.critic(state, action)
		critic_loss = F.mse_loss(cur_Q, reward)
		metrics['critic_loss'] = critic_loss.item()

		# --- Update critic --- 
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		metrics['critic_grad_norm'] = calculate_gradient_norm(self.critic).item()
		if self.critic_clip_value is not None:
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_clip_value)
		self.critic_optimizer.step()

		# --- Policy loss ---
		new_action, log_pi = self.actor(state)
		metrics['actor_entropy'] = - log_pi.mean().item()
		actor_loss = (alpha * log_pi.squeeze() - self.critic(state, new_action)).mean()
		metrics['actor_loss'] = actor_loss.item()

		# --- Update actor ---
		if update_actor:
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			metrics['actor_grad_norm'] = calculate_gradient_norm(self.actor).item()
			if self.actor_clip_value is not None:
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip_value)
			self.actor_optimizer.step()

			# --- Alpha loss ---
			target_entropy = self.per_atom_target_entropy * state['_atom_mask'].sum(-1)
			alpha_loss = -self.log_alpha * (log_pi + target_entropy).detach().mean()

			# --- Update alpha ---
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
		else:
			metrics['actor_grad_norm'] = 0.0

		self.total_it += 1
		return metrics

	def save(self, filename):
		filename = str(filename)
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		torch.save(self.log_alpha, filename + '_log_alpha')
		torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

	def light_save(self, filename):
		filename = str(filename)
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.log_alpha, filename + "_log_alpha")

	def load(self, filename):
		filename = str(filename)
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.log_alpha = torch.load(filename + '_log_alpha')
		self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))

	def light_load(self, filename):
		filename = str(filename)
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.log_alpha = torch.load(filename + '_log_alpha')
