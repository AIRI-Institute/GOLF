import torch

from tqc.functions import quantile_huber_loss_f
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
		top_quantiles_to_drop,
		target_entropy,
	):
		self.actor = actor
		self.critic = critic
		self.critic_target = critic_target
		self.log_alpha = torch.zeros((1,), requires_grad=True, device=DEVICE)

		# TODO: check hyperparams
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

		self.discount = discount
		self.tau = tau
		self.top_quantiles_to_drop = top_quantiles_to_drop
		self.target_entropy = target_entropy

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

		# --- Q loss ---
		with torch.no_grad():
			# get policy action
			new_next_action, next_log_pi = self.actor(next_state)
			# compute and cut quantiles at the next state
			next_z = self.critic_target(next_state, new_next_action)
			sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
			self.add_next_z_metrics(metrics, sorted_z)
			sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]

			# compute target
			print("---Target---")
			print("reward[0]: {} sorted_z_part[0]: {} alpha: {} next_log_pi[0]: {}".format(reward[0], sorted_z_part[0], alpha, next_log_pi[0]))
			target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)
			print("target[0]: ", target[0])
		
		# --- Critic loss ---
		cur_z = self.critic(state, action)
		print("---Critic---")

		print("cur_z[0]", cur_z[0])
		critic_loss = quantile_huber_loss_f(cur_z, target)
		metrics['critic_loss'] = critic_loss.item()

		# --- Policy and alpha loss ---

		# --- Update --- 

		# --- zero_grad ---
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()


		new_action, log_pi = self.actor(state)
		metrics['actor_entropy'] = - log_pi.mean().item()
		actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()
		metrics['actor_loss'] = actor_loss.item()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.actor_optimizer.step()

		# --- update target net ---
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
