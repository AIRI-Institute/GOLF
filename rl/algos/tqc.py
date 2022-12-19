import torch

from rl import DEVICE
from rl.utils import calculate_gradient_norm, quantile_huber_loss_f


class TQC(object):
	def __init__(
		self,
		policy,
		discount,
		tau,
		log_alpha,
		actor_lr,
		critic_lr,
		alpha_lr,
		top_quantiles_to_drop,
		per_atom_target_entropy,
		batch_size=256,
		actor_clip_value=None,
		critic_clip_value=None,
		use_one_cycle_lr=False,
		total_steps=0
	):
		self.actor = policy.actor
		self.critic = policy.critic
		self.critic_target = policy.critic_target
		self.log_alpha = torch.tensor([log_alpha], requires_grad=True, device=DEVICE)

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
		self.use_one_cycle_lr = use_one_cycle_lr
		if use_one_cycle_lr:
			self.actor_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.actor_optimizer, max_lr=25 * actor_lr,
																		  final_div_factor=1e+3, total_steps=total_steps)
			self.critic_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.critic_optimizer, max_lr=25 * critic_lr,
																	 	  final_div_factor=1e+3, total_steps=total_steps)

		self.discount = discount
		self.tau = tau
		self.top_quantiles_to_drop = top_quantiles_to_drop
		self.per_atom_target_entropy = per_atom_target_entropy
		self.batch_size = batch_size
		self.actor_clip_value = actor_clip_value
		self.critic_clip_value = critic_clip_value

		self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets

		self.total_it = 0

	def add_next_z_metrics(self, metrics, next_z):
		for t in range(1, self.critic.n_quantiles + 1):
			total_quantiles_to_keep = t * self.critic.n_nets
			metrics[f'Target_Q/Q_value_t={t}'] = next_z[:, :total_quantiles_to_keep].mean().__float__()

	def update(self, replay_buffer, update_actor, greedy):
		metrics = dict()
		state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)
		alpha = torch.exp(self.log_alpha)
		metrics['alpha'] = alpha.item()

		# --- Compute critic target ---
		# First select Q functions
		indices = self.critic_target.select_critics()
		if not greedy:
			with torch.no_grad():
				# get policy action
				new_next_action, next_log_pi = self.actor(next_state)
				# compute and cut quantiles at the next state
				next_z = self.critic_target(next_state, new_next_action)
				sorted_z, _ = torch.sort(next_z.reshape(self.batch_size, -1))
				self.add_next_z_metrics(metrics, sorted_z)
				sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]
				target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)
		else:
			target = reward
		
		# --- Critic loss ---
		# Set current Q functions to be the same as in target critic
		self.critic.set_critics(indices)
		cur_z = self.critic(state, action)
		critic_loss = quantile_huber_loss_f(cur_z, target)
		metrics['critic_loss'] = critic_loss.item()

		# --- Update critic --- 
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		metrics['critic_grad_norm'] = calculate_gradient_norm(self.critic).item()
		if self.critic_clip_value is not None:
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_clip_value)
		self.critic_optimizer.step()
		if self.use_one_cycle_lr:
			self.critic_lr_scheduler.step()

		# --- Policy loss ---
		new_action, log_pi = self.actor(state)
		metrics['actor_entropy'] = - log_pi.mean().item()
		actor_loss = (alpha * log_pi.squeeze() - self.critic(state, new_action).mean(dim=(1, 2))).mean()
		metrics['actor_loss'] = actor_loss.item()

		# --- Update actor ---
		if update_actor:
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			metrics['actor_grad_norm'] = calculate_gradient_norm(self.actor).item()
			if self.actor_clip_value is not None:
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip_value)
			self.actor_optimizer.step()
			if self.use_one_cycle_lr:
				self.actor_lr_scheduler.step()

			# --- Alpha loss ---
			target_entropy = self.per_atom_target_entropy * state['_atom_mask'].sum(-1)
			alpha_loss = -self.log_alpha * (log_pi + target_entropy).detach().mean()

			# --- Update alpha ---
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
		else:
			metrics['actor_grad_norm'] = 0.0

		# --- Update target net ---
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		self.total_it += 1
		return metrics

	def save(self, filename):
		filename = str(filename)
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_target.state_dict(), filename + "_critic_target")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		torch.save(self.log_alpha, filename + '_log_alpha')
		torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

	def light_save(self, filename):
		filename = str(filename)
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_target.state_dict(), filename + "_critic_target")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.log_alpha, filename + "_log_alpha")

	def load(self, filename):
		filename = str(filename)
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.log_alpha = torch.load(filename + '_log_alpha')
		self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))

	def light_load(self, filename):
		filename = str(filename)
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.log_alpha = torch.load(filename + '_log_alpha')
