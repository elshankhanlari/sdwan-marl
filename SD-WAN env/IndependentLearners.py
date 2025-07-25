# ---------------- Environment Definition ----------------
class SDWANEnv(gym.Env):
    def __init__(self, max_steps=300):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0
        self.overlays = {
            'Overlay1': {'service_rate': 100, 'latency':10},    # Bandwidth(50 Mbps)
            'Overlay2': {'service_rate': 20, 'latency':30},    # Bandwidth(20 Mbps)
            'Overlay3': {'service_rate': 50, 'latency':20}     # Bandwidth(30 Mbps)
        }
        self.overlay_queues = {name: [] for name in self.overlays}
        self._build_state()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, np.inf, (12,), np.float32)

    def _build_state(self):

        # reset step count and per‑step overlay fields
        self.step_count = 0
        self.overlays_step = {
            name: {
                'service_rate': cfg['service_rate'],
                'available_capacity': cfg['service_rate'],
                'latency': cfg['latency'],
                'Loss': 0
            } for name, cfg in self.overlays.items()
        }
        # clear shared queues
        for q in self.overlay_queues.values():
            q.clear()

    def reset(self, **kwargs):
      # resets everything for a new episode
      self._build_state()
      return self._get_observation(), {}

    def _generate_requests(self, overlay, queue, p, q_len, possion_mean):

      request_size = np.random.poisson(possion_mean) / 10  # flow size piosson between 0.2 and 5.0 MB # Convert to MB, range ~[0.1, 5.0]
      if overlay['service_rate'] > 0:
        # compute how many seconds (steps) needed
        steps_needed = max(1, math.ceil((request_size * 8) / overlay['service_rate'])) # bytes / bps = seconds
        if len(queue) < q_len:
            queue.append({
                'size': request_size,
                'remaining_steps': steps_needed,
                'started': False,
            })
        else:
            overlay['Loss'] += 1  # Queue full → drop request - branch buffer overflow

    def _process_requests(self, overlay, queue):
        comps = 0
        for req in list(queue):
            if req['remaining_steps'] > 0:
                req['started'] = True
                # determine how much capacity we need this step # Convert size to bits before dividing → get Mbps usage per step
                amt_needed = (req['size'] * 8) / req['remaining_steps']  # Mbps

                if amt_needed <= overlay['available_capacity']:
                    overlay['available_capacity'] -= amt_needed
                    # Convert Mbps to MB for subtraction from req['size']
                    req['size'] -= (amt_needed / 8)  # MB
                    req['remaining_steps'] -= 1

                    if req['size'] <= 0 or req['remaining_steps'] <= 0:
                        comps += 1
                        queue.remove(req)
                else:
                    overlay['Loss'] += 1

        return comps

    def calculate_individual_reward(self, ov_a, ov_b, action_a, action_b, comp_a, comp_b):
        α_bw, α_lat, α_loss, α_comp, lam = 1.0, 1.0, 1.0, 2, 0.8
        B0_A = self.overlays['Overlay1']['service_rate'] if action_a==0 else self.overlays['Overlay2']['service_rate']
        B0_B = self.overlays['Overlay1']['service_rate'] if action_b==0 else self.overlays['Overlay3']['service_rate']

        def single_reward(ov, B0, comp):
            #bw_n = α_bw * (1-(ov['available_capacity']/B0))
            #lat_n = α_lat * ((max_lat - ov['latency'])/max_lat)
            #lat_n = α_lat * (( -ov['latency']))
            bw_n = α_bw * (ov['available_capacity'])
            loss = α_loss * ov['Loss']
            return bw_n - loss + (α_comp  * comp)

        r_a = single_reward(ov_a, B0_A, comp_a)
        r_b = single_reward(ov_b, B0_B, comp_b)
        r_tot = r_a + r_b

        return r_a, r_b, r_tot

    def _get_observation(self):
      # Build a flat vector: [bw1, lat1, Loss1, bw2, lat2, Loss2, bw3, lat3, Loss3, Q_1, Q_2, Q_3]
        flat = []
        for k in ('Overlay1','Overlay2','Overlay3'):
            o = self.overlays_step[k]
            flat += [o['available_capacity'], o['latency'], o['Loss']]
        flat += [len(self.overlay_queues['Overlay1']), len(self.overlay_queues['Overlay2']), len(self.overlay_queues['Overlay3'])]
        return np.array(flat, np.float32)

    def step(self, action):
        self.step_count += 1

        # 1) At start of step, refill available_capacity for each overlay
        for ov in self.overlays_step.values():
            ov['available_capacity'] = ov['service_rate']
            ov['Loss'] = 0

        # decode joint action
        mapping = {
            0: ("Overlay1","Overlay1"),
            1: ("Overlay1","Overlay3"),
            2: ("Overlay2","Overlay1"),
            3: ("Overlay2","Overlay3")
        }
        name_a, name_b = mapping[action]

        λ_a, λ_b = 9, 6  # Mean requests per second for Branch A and B
        mean_a, mean_b = 20, 10  # Mean flow sizes (for Poisson size)

        num_arrivals_a = np.random.poisson(λ_a)
        for _ in range(num_arrivals_a):
          self._generate_requests(self.overlays_step[name_a], self.overlay_queues[name_a], 0.8, 50, mean_a)

        num_arrivals_b = np.random.poisson(λ_b)
        for _ in range(num_arrivals_b):
          self._generate_requests(self.overlays_step[name_b], self.overlay_queues[name_b], 0.6 ,50, mean_b)

        # 4) Process each overlay once
        comps = {}
        for name in self.overlays_step:
            comps[name] = self._process_requests(self.overlays_step[name], self.overlay_queues[name])

        # 5) Extract per‑agent completions
        comp_a = comps[name_a]
        comp_b = comps[name_b]

        # Map each branch’s discrete choice (0 or 1) for normalization
        # Branch A: 0→Overlay1, 1→Overlay2
        action_a = 0 if name_a=="Overlay1" else 1
        # Branch B: 0→Overlay1, 1→Overlay3
        action_b = 0 if name_b=="Overlay1" else 1

        # Calculate Rewards
        r_a, r_b, tot = self.calculate_individual_reward(
            self.overlays_step[name_a], self.overlays_step[name_b], action_a, action_b, comp_a, comp_b
        )


        # check congestion
        congested1 = float(self.overlays_step['Overlay1']['available_capacity'] <= 10)
        congested2 = float(self.overlays_step['Overlay2']['available_capacity'] <= 10)
        congested3 = float(self.overlays_step['Overlay3']['available_capacity'] <= 10)

        # capture raw bandwidth of each overlay
        bw1 = self.overlays_step['Overlay1']['available_capacity']
        bw2 = self.overlays_step['Overlay2']['available_capacity']
        bw3 = self.overlays_step['Overlay3']['available_capacity']

        done = self.step_count >= self.max_steps
        # Update last action and overlays
        self.last_action_a = action
        self.last_action_b = action
        self.last_overlay_a = self.overlays_step[name_a]
        self.last_overlay_b = self.overlays_step[name_b]
        obs = self._get_observation()
        info = {
            'reward_a':  r_a,
            'reward_b':  r_b,
            'total':     tot,
            'congested1': congested1,
            'congested2': congested2,
            'congested3': congested3,
            'bw1':       bw1,
            'bw2':       bw2,
            'bw3':       bw3,
        }
        return obs, tot, done, False, info

# Wrapper for independent learning
class IndependentBranchEnv(gym.Env):
    def __init__(self, branch, base_env=None, max_steps=300):
        super().__init__()
        self.branch = branch  # 'A' or 'B'
        self.env = base_env or SDWANEnv(max_steps)
        # 2 actions: 0->Overlay1, 1->Overlay2 (for A) or Overlay3 (for B)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # choose random for other branch
        other_action = random.choices([0, 1], weights=[0.6, 0.2])[0]
        # map each to joint action
        if self.branch == 'A':
            a = action; b = other_action
            joint = 0 if (a==0 and b==0) else \
                    1 if (a==0 and b==1) else \
                    2 if (a==1 and b==0) else 3
        else:
            a = random.choices([0, 1], weights=[0.2, 0.6])[0]; b = action
            joint = 0 if (a==0 and b==0) else \
                    1 if (a==0 and b==1) else \
                    2 if (a==1 and b==0) else 3
        obs, tot_reward, done, _, info = self.env.step(joint)

        # tag it for our logger
        info['joint_action'] = joint

        r = info['reward_a'] if self.branch=='A' else info['reward_b']
        return obs, r, done, False, info
