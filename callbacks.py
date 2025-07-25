from stable_baselines3.common.callbacks import BaseCallback

class EpisodeReturnLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.a_returns, self.b_returns, self.tot_returns, self.timesteps = [], [], [], []
        self._ra = self._rb = self._rt = 0.0

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            self._ra += info.get('reward_a', 0.0)
            self._rb += info.get('reward_b', 0.0)
            self._rt += info.get('total', 0.0)
            if 'episode' in info:
                self.a_returns.append(self._ra)
                self.b_returns.append(self._rb)
                self.tot_returns.append(self._rt)
                self.timesteps.append(self.num_timesteps)
                self._ra = self._rb = self._rt = 0.0
        return True

class CongestionLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.congested1 = []
        self.congested2 = []
        self.congested3 = []
        self.congested_counts = []
        self.episode_lengths = []
        self._c1 = self._c2 = self._c3 = 0
        self._count = self._steps = 0

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        self._c1 += int(info.get('congested1', 0))
        self._c2 += int(info.get('congested2', 0))
        self._c3 += int(info.get('congested3', 0))
        self._count += int(info.get('congested1', 0))
        self._steps += 1

        if 'episode' in info:
            self.congested1.append(self._c1)
            self.congested2.append(self._c2)
            self.congested3.append(self._c3)
            self.congested_counts.append(self._count)
            self.episode_lengths.append(self._steps)
            self._c1 = self._c2 = self._c3 = self._count = self._steps = 0
        return True

class JointActionLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.joint_actions = []

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        ja = info.get('joint_action')
        if ja is not None:
            self.joint_actions.append(int(ja))
        return True
