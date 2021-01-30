from agents.VanillaDQN import *

class clipALDQN(VanillaDQN):
    '''
    Implementation of clipped Advantage Learning DQN with target network and replay buffer
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
        # the percentage of the maximum, when surpassing, subtract the action gap, else not
        self.clip_r= cfg['agent']['clipratio']

    def learn(self):
        super().learn()
        # Update target network
        if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
            self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

    def compute_q_target(self, batch):

        with torch.no_grad():
            q_cur = self.Q_net_target[0](batch.state)
            action_idx = batch.action.unsqueeze(1)
            q_s_a = q_cur.gather(1, action_idx.long()).squeeze() # (bs, )
            q_cur_max = q_cur.max(1)[0] # (bs, )
            # clip mask
            q_ratio = q_s_a / q_cur_max
            mask = (q_ratio > self.clip_r).float()
            # Bellman optimal operator
            q_next = self.Q_net_target[0](batch.next_state).max(1)[0]
            q_target = batch.reward + self.discount * q_next * batch.mask
            # add clip AL augmented term
            q_target += self.alpha * (q_s_a - q_cur_max) * mask
        return q_target

class clipPALDQN(VanillaDQN):
    '''
    Implementation of generalized clipped Advantage Learning DQN with target network and replay buffer
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
        # the percentage of the maximum, when surpassing, subtract the action gap, else not
        self.clip_r= cfg['agent']['clipratio']

    def learn(self):
        super().learn()
        # Update target network
        if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
            self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

    def compute_q_target(self, batch):

        with torch.no_grad():
            q_cur = self.Q_net_target[0](batch.state)
            action_idx = batch.action.unsqueeze(1)
            q_s_a = q_cur.gather(1, action_idx.long()).squeeze() # (bs, )
            q_cur_max = q_cur.max(1)[0] # (bs, )
            # clip mask
            q_ratio = q_s_a / q_cur_max
            mask = (q_ratio > self.clip_r).float()
            # PAL augment term
            q_next = self.Q_net_target[0](batch.next_state)
            # q(s', a)
            q_next_a = q_next.gather(1, action_idx.long()).squeeze()
            # v(s')
            q_next_max = q_next.max(1)[0]

            # Bellman optimal operator
            q_next = self.Q_net_target[0](batch.next_state).max(1)[0]
            q_target = batch.reward + self.discount * q_next * batch.mask
            # add clip AL augmented term
            q_target += torch.max(self.alpha * (q_s_a - q_cur_max), self.discount * (q_next_a - q_next_max)) * mask

        return q_target

class genALDQN(VanillaDQN):
    '''
    Implementation of generalized clipped Advantage Learning DQN with target network and replay buffer
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
        # the percentage of the maximum, when surpassing, subtract the action gap, else not
        self.clip_r= cfg['agent']['clipratio']
        self.beta = cfg['agent']['beta']

    def learn(self):
        super().learn()
        # Update target network
        if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
            self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

    def compute_q_target(self, batch):

        with torch.no_grad():
            q_cur = self.Q_net_target[0](batch.state)
            action_idx = batch.action.unsqueeze(1)
            q_s_a = q_cur.gather(1, action_idx.long()).squeeze() # (bs, )
            q_cur_max = q_cur.max(1)[0] # (bs, )
            # clip mask(indicate the part closest to the optimal action value )
            q_ratio = q_s_a / q_cur_max
            mask = (q_ratio > self.clip_r).float()
            # Bellman optimal operator
            q_next = self.Q_net_target[0](batch.next_state).max(1)[0]
            q_target = batch.reward + self.discount * q_next * batch.mask

            # add generalized cip AL augmented term
            delta_q = q_cur_max - q_s_a # positive gap
            gap_term = (1 - self.beta) * delta_q * mask + self.beta * delta_q
            q_target -= self.alpha * gap_term
        return q_target

class decayALDQN(VanillaDQN):
    '''
    Implementation of decay (two stage slope)clipped Advantage Learning DQN with target network and replay buffer
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
        # the percentage of the maximum, when surpassing, subtract the advantage action gap,
        # otherwise, subtract the decay action gap
        self.clip_r= cfg['agent']['clipratio']
        # the slope of the second stage(Q>=c*V), should be negative
        self.beta = cfg['agent']['beta']

    def learn(self):
        super().learn()
        # Update target network
        if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
            self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

    def compute_q_target(self, batch):

        with torch.no_grad():
            q_cur = self.Q_net_target[0](batch.state)
            action_idx = batch.action.unsqueeze(1)
            q_s_a = q_cur.gather(1, action_idx.long()).squeeze() # (bs, )
            q_cur_max = q_cur.max(1)[0] # (bs, )
            # find the threshold, and create clip mask(indicate the part closest to the optimal action value )
            q_threshold = q_cur_max * self.clip_r
            delta_q_threshold = q_cur_max - q_threshold
            # first keep the same as AL, the slope is 1
            mask_1_stage = (q_s_a > q_threshold).float()
            mask_2_stage = 1 - mask_1_stage
            # add generalized cip AL augmented term
            delta_q = q_cur_max - q_s_a # positive gap
            decay_delta_q = torch.clamp((delta_q - delta_q_threshold) * self.beta + delta_q_threshold, 0)

            # Bellman optimal operator
            q_next = self.Q_net_target[0](batch.next_state).max(1)[0]
            q_target = batch.reward + self.discount * q_next * batch.mask

            gap_term = delta_q * mask_1_stage + decay_delta_q * mask_2_stage
            q_target -= self.alpha * gap_term
        return q_target

