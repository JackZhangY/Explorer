from agents.VanillaDQN import *


class CVIDQN(VanillaDQN):
    '''
    Implementation of Conservative Value Iteration with target network and replay buffer
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        # Create target Q value network
        self.Q_net_target = [None]
        self.Q_net_target[0] = self.createNN(cfg['env']['input_type']).to(self.device)
        # Load target Q value network
        self.Q_net_target[0].load_state_dict(self.Q_net[0].state_dict())
        self.Q_net_target[0].eval()
        # alpha for soft action gap
        self.alpha = cfg['agent']['alpha']
        # beta for mellowmax
        self.beta = cfg['agent']['beta']

    def learn(self):
        super().learn()
        # Update target network
        if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
          self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

    def compute_q_target(self, batch):

        with torch.no_grad():
            q_next = self.Q_net_target[0](batch.next_state) # (bs, act_dims)
            mm_q_next = self.mellowmax(q_next) # (bs, )

            # additional term
            q_cur = self.Q_net_target[0](batch.state)
            mm_q_cur = self.mellowmax(q_cur) # (bs, )
            action_idx = batch.action.unsqueeze(1)
            q_s_a = q_cur.gather(1, action_idx.long()).squeeze() # (bs, )

            q_target = batch.reward + self.discount * mm_q_next * batch.mask + self.alpha * (q_s_a - mm_q_cur)

        return q_target

    def mellowmax(self, q_value):
        """

        :param q_value: shape should be (bs, act_dims)
        :return: 1/beta * log(1/act_dims * sum(exp(beta*q)))
        """
        q_max = torch.max(q_value, dim=1, keepdim=True)[0] #(bs, 1)
        mean_exp_q_value = torch.mean(torch.exp(self.beta * (q_value - q_max)), dim=1) # (bs,)
        log_mean_exp_q = torch.log(mean_exp_q_value) / self.beta + q_max.squeeze()

        return log_mean_exp_q # (bs,)


