from agents.VanillaDQN import *


class SoftDQN(VanillaDQN):
    '''
    Implementation of Soft DQN, i.e. discrete SAC with target network and replay buffer
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        # Create target Q value network
        self.Q_net_target = [None]
        self.Q_net_target[0] = self.createNN(cfg['env']['input_type']).to(self.device)
        # Load target Q value network
        self.Q_net_target[0].load_state_dict(self.Q_net[0].state_dict())
        self.Q_net_target[0].eval()
        self.tau = cfg['agent']['ent_temp']

    def learn(self):
        super().learn()
        # Update target network
        if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
          self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

    def compute_q_target(self, batch):

        with torch.no_grad():
            # tau * ln pi_k+1(s'): (bs, act_dims)
            q_next = self.Q_net_target[0](batch.next_state)
            tau_log_pi_next_s = self.stable_scaled_log_softmax(q_next)
            # pi_k+1(s'): (bs, act_dims)
            next_a_prob = self.stable_softmax(q_next)

            # expected q_next add policy entropy: (bs,)
            expected_q_plus_entropy = (next_a_prob * (q_next - tau_log_pi_next_s)).sum(dim=1, keepdim=False)

            q_target= (batch.reward + self.discount * expected_q_plus_entropy *  batch.mask)

        return q_target

    def stable_scaled_log_softmax(self, q_value):
        """

        :param q_value: shape should be (bs, act_dims)
        :return: tau * ln softmax(q/tau)
        """
        max_q = torch.max(q_value, dim=1, keepdim=True)[0] # (bs, 1)
        q_sub_max = q_value - max_q # (bs, act_dims)
        tau_lse = max_q + self.tau * torch.log(torch.sum(torch.exp(q_sub_max / self.tau), dim=1, keepdim=True)) #(bs, 1)
        return q_value - tau_lse # (bs, act_dims)

    def stable_softmax(self, q_value):
        """

        :param q_value:
        :return: softmax(q_value/tau)
        """
        max_q = torch.max(q_value, dim=1, keepdim=True)[0]
        q_sub_max = q_value - max_q
        return F.softmax(q_sub_max / self.tau, dim=1) # (bs, act_dims)