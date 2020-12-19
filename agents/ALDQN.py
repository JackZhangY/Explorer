from agents.VanillaDQN import *


class ALDQN(VanillaDQN):
    '''
    Implementation of Advantage Learning DQN with target network and replay buffer
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        # Create target Q value network
        self.Q_net_target = [None]
        self.Q_net_target[0] = self.createNN(cfg['env']['input_type']).to(self.device)
        # Load target Q value network
        self.Q_net_target[0].load_state_dict(self.Q_net[0].state_dict())
        self.Q_net_target[0].eval()
        self.alpha = cfg['agent']['alpha']

    def learn(self):
        super().learn()
        # Update target network
        if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
            self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

    def compute_q_target(self, batch):
        with torch.no_grad():
            q_cur = self.Q_net_target[0](batch.state)
            q_s_a = q_cur.gather(1, batch.action.long()).squeeze() # (bs, )
            q_cur_max = q_cur.max(1)[0] # (bs, )
            # Bellman optimal operator
            q_next = self.Q_net_target[0](batch.next_state).max(1)[0]
            q_target = batch.reward + self.discount * q_next * batch.mask
            # add AL augmented term
            q_target += self.alpha * (q_s_a - q_cur_max)
        return q_target

class PALDQN(VanillaDQN):
    '''
    Implementation of Persistent Advantage Learning DQN with target network and replay buffer
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        # Create target Q value network
        self.Q_net_target = [None]
        self.Q_net_target[0] = self.createNN(cfg['env']['input_type']).to(self.device)
        # Load target Q value network
        self.Q_net_target[0].load_state_dict(self.Q_net[0].state_dict())
        self.Q_net_target[0].eval()
        self.alpha = cfg['agent']['alpha']

    def learn(self):
        super().learn()
        # Update target network
        if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
            self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

    def compute_q_target(self, batch):
        with torch.no_grad():
            q_cur = self.Q_net_target[0](batch.state)
            # q(s, a)
            q_s_a = q_cur.gather(1, batch.action.long()).squeeze()
            # v(s)
            q_cur_max = q_cur.max(1)[0]

            q_next = self.Q_net_target[0](batch.next_state)
            # q(s', a)
            q_next_a = q_next.gather(1, batch.action.long()).squeeze()
            # v(s')
            q_next_max = q_next.max(1)[0]

            # Bellman optimal operator
            q_target = batch.reward + self.discount * q_next_max * batch.mask
            # add the PAL augmented term
            q_target += torch.max(-self.alpha * (q_cur_max - q_s_a), -self.discount * (q_next_max - q_next_a))

        return q_target
