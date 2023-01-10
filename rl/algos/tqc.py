import torch

from torch.nn.functional import mse_loss

from rl import DEVICE
from rl.utils import calculate_gradient_norm, quantile_huber_loss_f, get_lr_scheduler


critic_losses = {
	"TQC": quantile_huber_loss_f,
	"SAC": mse_loss,
}


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
		critic_type="TQC",
		batch_size=256,
		actor_clip_value=None,
		critic_clip_value=None,
		lr_scheduler=None,
		total_steps=0
	):
		self.critic_type = critic_type
		self.actor = policy.actor
		self.critic = policy.critic
		self.critic_target = policy.critic_target
		self.log_alpha = torch.tensor([log_alpha], requires_grad=True, device=DEVICE)

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

		self.use_lr_scheduler = lr_scheduler is not None
		if self.use_lr_scheduler:
			lr_kwargs = {
				"gamma": 0.1,
				"total_steps": total_steps,
				"final_div_factor": 1e+3,
			}
			lr_kwargs['initial_lr'] = actor_lr
			self.actor_lr_scheduler = get_lr_scheduler(lr_scheduler, self.actor_optimizer, **lr_kwargs)
			lr_kwargs['initial_lr'] = critic_lr
			self.critic_lr_scheduler = get_lr_scheduler(lr_scheduler, self.critic_optimizer, **lr_kwargs)

		self.discount = discount
		self.tau = tau
		self.top_quantiles_to_drop = top_quantiles_to_drop
		self.per_atom_target_entropy = per_atom_target_entropy
		self.batch_size = batch_size
		self.actor_clip_value = actor_clip_value
		self.critic_clip_value = critic_clip_value

		if self.critic_type == "TQC":
			self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets

		self.total_it = 0

	def add_next_z_metrics(self, metrics, next_z):
		for t in range(1, self.critic.n_quantiles + 1):
			total_quantiles_to_keep = t * self.critic.n_nets
			metrics[f'Target_Q/Q_value_t={t}'] = next_z[:, :total_quantiles_to_keep].mean().__float__()

	def update(self, replay_buffer, update_actor, greedy):
		metrics = dict()

		# For PaiNN debugging
		# recent_tensors = {}

		state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)
		
		# For PaiNN debugging
		# recent_tensors['state'] = state
		# recent_tensors['action'] = action
		# recent_tensors['next_state'] = next_state
		# recent_tensors['reward'] = reward
		# recent_tensors['not_done'] = not_done

		alpha = torch.exp(self.log_alpha)
		metrics['alpha'] = alpha.item()

		# --- Compute critic target ---
		# First select Q functions
		indices = self.critic_target.select_critics()
		if not greedy:
			with torch.no_grad():
					# Get fresh policy action
					new_next_action, next_log_pi = self.actor(next_state)
					# Get critic prediction for the next_state
					next_Q = self.critic_target(next_state, new_next_action)
					
					# For PaiNN debugging
					# recent_tensors['new_next_action'] = new_next_action
					# recent_tensors['next_log_pi'] = next_log_pi
					# recent_tensors['next_Q'] = next_Q
					if self.critic_type == "TQC":
						# Sort all quantiles
						next_Q, _ = torch.sort(next_Q.reshape(self.batch_size, -1))
						self.add_next_z_metrics(metrics, next_Q)
						# Cut top quantiles
						next_Q = next_Q[:, :self.quantiles_total - self.top_quantiles_to_drop]
					elif self.critic_type == "SAC":
						# Take minimum of Q functions
						next_Q, _ = torch.min(next_Q, dim=1)
					target = reward + not_done * self.discount * (next_Q - alpha * next_log_pi)
		else:
			target = reward
		
		if self.critic_type == "SAC":
			target = target.expand((-1, self.critic.m_nets)).unsqueeze(2)
		
		# For PaiNN debugging
		# recent_tensors['target'] = target

		# --- Critic loss ---
		# Set current Q functions to be the same as in target critic
		self.critic.set_critics(indices)
		cur_Q = self.critic(state, action)
		critic_loss = critic_losses[self.critic_type](cur_Q, target)

		# recent_tensors['critic_loss'] = critic_loss

		# If loss has inf value, its grad will be Nan which will cause an error.
		# As a dirty fix we suggest to just skip such training steps.
		if torch.isinf(critic_loss):
			print("Inf in critic loss")
			return metrics
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

		# For PaiNN debugging
		# recent_tensors['new_action'] = new_action
		# recent_tensors['log_pi'] = log_pi
		
		# Set dimensions to take mean along
		if self.critic_type == "TQC":
			dims = (1, 2)
		elif self.critic_type == "SAC":
			dims = (1, )

		# Calculate actor loss
		critic_out = self.critic(state, new_action).mean(dim=dims)
		entropy = alpha * log_pi.squeeze()
		actor_loss = (entropy - critic_out).mean()
		# If loss has inf value, its grad will be Nan which will cause an error.
		# As a dirty fix we suggest to just skip such training steps.
		if torch.isinf(actor_loss):
			print("Inf in actor loss")
			return metrics
		metrics['actor_loss'] = actor_loss.item()

		# For PaiNN debugging
		# recent_tensors['critic_out'] = critic_out
		# recent_tensors['entropy'] = entropy

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
			# If loss has inf value, its grad will be Nan which will cause an error.
			# As a dirty fix we suggest to just skip such training steps.
			if torch.isinf(alpha_loss):
				print("Inf in alpha loss")
				return metrics
			# For PaiNN debugging
			# recent_tensors['alpha_raw_loss'] = -self.log_alpha * (log_pi + target_entropy).detach()
			

			# --- Update alpha ---
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
		else:
			metrics['actor_grad_norm'] = 0.0
		
		if self.use_lr_scheduler:
			self.actor_lr_scheduler.step()
			self.critic_lr_scheduler.step()

		# --- Update target net ---
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		self.total_it += 1
		return metrics #, recent_tensors

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
