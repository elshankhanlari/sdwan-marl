
# ---------------- Callback Definitions ----------------
# 1) Define the callback to log completed episodes:
class AgentAndTotalLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.agent_rewards = []  # Rewards for each agent
        self.total_rewards = []  # Total rewards for the episode
        self.timesteps     = []  # Timesteps for each episode
        self.episode_rewards = [] # Rewards for each episode
        self._running_total = 0.0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            # accumulate the env’s lambda-weighted step reward
            self._running_total += info.get("total", 0.0)
            # when an episode ends, Monitor will add “episode” to info
            if "episode" in info:
                self.agent_rewards.append(info["episode"]["r"])
                self.total_rewards.append(self._running_total)
                self.timesteps.append(self.num_timesteps)
                self.episode_rewards.append(info["episode"]["r"])
                self._running_total = 0.0
        return True

class CongestionLogger(BaseCallback):
    """
    Logs per‐episode congestion counts for Overlay1, Overlay2, Overlay3,
    and prints final ranking if you keep bw_history.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.congested1 = []
        self.congested2 = []
        self.congested3 = []
        self.episode_lengths = []
        self.bw_history = []  # optional, remove if unused

        # running counters
        self._c1 = self._c2 = self._c3 = 0
        self._steps = 0

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        # accumulate each step
        self._c1 += int(info.get('congested1', 0.0))
        self._c2 += int(info.get('congested2', 0.0))
        self._c3 += int(info.get('congested3', 0.0))
        self._steps += 1

        # store last bandwidths if you still want the printout
        last_bw = (info['bw1'], info['bw2'], info['bw3'])
        if len(self.bw_history) < len(self.congested1) + 1:
            self.bw_history.append(last_bw)

        # at episode end
        if 'episode' in info:
            self.congested1.append(self._c1)
            self.congested2.append(self._c2)
            self.congested3.append(self._c3)
            self.episode_lengths.append(self._steps)

            # optional: print ranked status
            stats = [
                ('Overlay1', last_bw[0], bool(info['congested1'])),
                ('Overlay2', last_bw[1], bool(info['congested2'])),
                ('Overlay3', last_bw[2], bool(info['congested3'])),
            ]
            stats.sort(key=lambda x: x[1])
            ep = len(self.congested1)
            print(f"\n[Episode {ep:3d}] Overlay final status:")
            for rank, (name, bw, cong) in enumerate(stats, 1):
                print(f"  {rank}. {name:<8s} | BW: {bw:8.2f} | {'CONGESTED' if cong else 'OK'}")
            print('-'*40)

            # reset for next episode
            self._c1 = self._c2 = self._c3 = 0
            self._steps = 0

        return True

class JointActionLogger(BaseCallback):
    """Logs the combined joint action code (0–3) each step."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.joint_actions = []

    def _on_step(self) -> bool:
        # read 'joint_action' from the info dict
        info = self.locals["infos"][0]
        ja = info.get("joint_action")
        if ja is not None:
            self.joint_actions.append(int(ja))
        return True