import gymnasium as gym
import numpy as np
import math

class SDWANEnv(gym.Env):
    def __init__(self, max_steps=300):
        super().__init__()
        self.max_steps = max_steps
        self.overlays = {
            'Overlay1': {'service_rate': 100, 'latency':10},
            'Overlay2': {'service_rate': 20, 'latency':30},
            'Overlay3': {'service_rate': 50, 'latency':20}
        }
        self.overlay_queues = {name: [] for name in self.overlays}
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, np.inf, (12,), np.float32)
        self._build_state()

    def _build_state(self):
        self.step_count = 0
        self.overlays_step = {
            name: {
                'service_rate': cfg['service_rate'],
                'available_capacity': cfg['service_rate'],
                'latency': cfg['latency'],
                'Loss': 0
            } for name, cfg in self.overlays.items()
        }
        for q in self.overlay_queues.values():
            q.clear()

    def reset(self, **kwargs):
        self._build_state()
        return self._get_observation(), {}

    def _generate_requests(self, overlay, queue, p, q_len, possion_mean):
        request_size = np.random.poisson(possion_mean) / 10
        if overlay['service_rate'] > 0:
            steps_needed = max(1, math.ceil((request_size * 8) / overlay['service_rate']))
            if len(queue) < q_len:
                queue.append({
                    'size': request_size,
                    'remaining_steps': steps_needed,
                    'started': False,
                })
            else:
                overlay['Loss'] += 1

    def _process_requests(self, overlay, queue):
        comps = 0
        for req in list(queue):
            if req['remaining_steps'] > 0:
                req['started'] = True
                amt_needed = (req['size'] * 8) / req['remaining_steps']
                if amt_needed <= overlay['available_capacity']:
                    overlay['available_capacity'] -= amt_needed
                    req['size'] -= (amt_needed / 8)
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

        def sr(ov, B0, comp):
            bw_n = α_bw * (ov['available_capacity'])
            loss = α_loss * ov['Loss']
            return bw_n - loss + (α_comp * comp)

        r_a = sr(ov_a, B0_A, comp_a)
        r_b = sr(ov_b, B0_B, comp_b)
        r_tot = lam * r_a + (1 - lam) * r_b
        return r_a, r_b, r_tot

    def _get_observation(self):
        flat = []
        for k in ('Overlay1','Overlay2','Overlay3'):
            o = self.overlays_step[k]
            flat += [o['available_capacity'], o['latency'], o['Loss']]
        flat += [len(self.overlay_queues['Overlay1']), len(self.overlay_queues['Overlay2']), len(self.overlay_queues['Overlay3'])]
        return np.array(flat, np.float32)

    def step(self, action):
        self.step_count += 1
        for ov in self.overlays_step.values():
            ov['available_capacity'] = ov['service_rate']
            ov['Loss'] = 0

        mapping = {
            0: ('Overlay1','Overlay1'),
            1: ('Overlay1','Overlay3'),
            2: ('Overlay2','Overlay1'),
            3: ('Overlay2','Overlay3')
        }
        na, nb = mapping[action]

        λ_a, λ_b = 9, 6
        mean_a, mean_b = 20, 10

        for _ in range(np.random.poisson(λ_a)):
            self._generate_requests(self.overlays_step[na], self.overlay_queues[na], 0.8, 50, mean_a)
        for _ in range(np.random.poisson(λ_b)):
            self._generate_requests(self.overlays_step[nb], self.overlay_queues[nb], 0.6, 50, mean_b)

        comps = {}
        for name in self.overlays_step:
            comps[name] = self._process_requests(self.overlays_step[name], self.overlay_queues[name])

        comp_a = comps[na]
        comp_b = comps[nb]

        act_a = 0 if na == 'Overlay1' else 1
        act_b = 0 if nb == 'Overlay1' else 1

        r_a, r_b, tot = self.calculate_individual_reward(
            self.overlays_step[na], self.overlays_step[nb], act_a, act_b, comp_a, comp_b
        )

        bw1 = self.overlays_step['Overlay1']['available_capacity']
        bw2 = self.overlays_step['Overlay2']['available_capacity']
        bw3 = self.overlays_step['Overlay3']['available_capacity']
        info = {
            'reward_a':  r_a,
            'reward_b':  r_b,
            'total':     tot,
            'bw1':       bw1,
            'bw2':       bw2,
            'bw3':       bw3,
            'congested1': float(bw1 <= 10),
            'congested2': float(bw2 <= 10),
            'congested3': float(bw3 <= 10),
            'joint_action': action
        }

        done = self.step_count >= self.max_steps
        return self._get_observation(), tot, done, False, info
