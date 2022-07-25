import torch

from tqc.utils import quantile_huber_loss_f, calculate_gradient_norm
from tqc import DEVICE


class Trainer(object):
	def __init__(
		self,
		*,
		actor,
		critic,
		critic_target,
		discount,
		tau,
		log_alpha,
		actor_lr,
		critic_lr,
		alpha_lr,
		top_quantiles_to_drop,
		per_atom_target_entropy,
	):
		self.actor = actor
		self.critic = critic
		self.critic_target = critic_target
		self.log_alpha = torch.tensor([log_alpha], requires_grad=True, device=DEVICE)

		# TODO: check hyperparams
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

		self.discount = discount
		self.tau = tau
		self.top_quantiles_to_drop = top_quantiles_to_drop
		self.per_atom_target_entropy = per_atom_target_entropy

		self.quantiles_total = critic.n_quantiles * critic.n_nets

		self.total_it = 0

	def add_next_z_metrics(self, metrics, next_z):
		for t in range(1, self.critic.n_quantiles + 1):
			total_quantiles_to_keep = t * self.critic.n_nets
			metrics[f'Target_Q/Q_value_t={t}'] = next_z[:, :total_quantiles_to_keep].mean().__float__()

	def train(self, replay_buffer, batch_size=256):
		metrics = dict()
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		alpha = torch.exp(self.log_alpha)
		metrics['alpha'] = alpha.item()

		# --- Compute critic target ---
		with torch.no_grad():
			# get policy action
			new_next_action, next_log_pi = self.actor(next_state)
			# compute and cut quantiles at the next state
			next_z = self.critic_target(next_state, new_next_action)
			sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
			self.add_next_z_metrics(metrics, sorted_z)
			sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]
			target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)
		
		# --- Critic loss ---
		cur_z = self.critic(state, action)
		critic_loss = quantile_huber_loss_f(cur_z, target)
		metrics['critic_loss'] = critic_loss.item()

		# --- Update critic --- 
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		metrics['critic_grad_norm'] = calculate_gradient_norm(self.critic).item()
		self.critic_optimizer.step()

		# --- Policy loss ---
		new_action, log_pi = self.actor(state)
		metrics['actor_entropy'] = - log_pi.mean().item()
		actor_loss = (alpha * log_pi.squeeze() - self.critic(state, new_action).mean(dim=(1, 2))).mean()
		metrics['actor_loss'] = actor_loss.item()

		# --- Update actor ---
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		metrics['actor_grad_norm'] = calculate_gradient_norm(self.actor).item()
		self.actor_optimizer.step()

		# --- Alpha loss ---
		target_entropy = self.per_atom_target_entropy * state['_atoms_count']
		alpha_loss = -self.log_alpha * (log_pi + target_entropy).detach().mean()

		# --- Update alpha ---
		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

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
