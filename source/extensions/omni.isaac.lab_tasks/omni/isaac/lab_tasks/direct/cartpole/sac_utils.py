import torch
from gymnasium import Wrapper

class SACCompatibilityWrapper(Wrapper):
    """
    Ensures observations are returned as a dict {'policy': <tensor>}.
    RL Games SAC expects a dict when resetting, but a raw tensor is also okay
    if we re-inject it as {'policy': obs} upon step.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, **kwargs):
        # Isaac Gym / Gymnasium can return (obs, info). We handle that.
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            if isinstance(obs, dict) and 'policy' in obs:
                return obs, info
            else:
                # Convert raw obs (tensor/array/anything) into {'policy': obs}
                return {'policy': obs}, info

        # If you get just a single obs (older Gym?), do the same
        if isinstance(result, dict) and 'policy' in result:
            return result
        else:
            return {'policy': result}

    def step(self, action):
        # New gym returns 5 items: (obs, reward, terminated, truncated, info).
        results = self.env.step(action)
        if len(results) == 5:
            obs, reward, done, truncated, info = results
            if isinstance(obs, dict) and 'policy' in obs:
                return obs, reward, done, truncated, info
            elif isinstance(obs, torch.Tensor):
                return {'policy': obs}, reward, done, truncated, info
            else:
                # If your env returns a dict without 'policy',
                # or some other structure, you can wrap it here:
                return {'policy': obs}, reward, done, truncated, info
        else:
            # Fallback (if old env returns 4 items or an unexpected format)
            return results
        
def patch_rlgames_sac():
    """
    Monkey-patch RL Games' SACAgent to handle dict observations:
    - If obs is a dict with 'policy', we use obs['policy'] as the actual tensor.
    """
    import rl_games.algos_torch.sac_agent
    SACAgent = rl_games.algos_torch.sac_agent.SACAgent

    # 1) Patch env_reset
    original_env_reset = SACAgent.env_reset
    def patched_env_reset(self):
        result = self.vec_env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, _info = result
        else:
            obs = result
        
        if isinstance(obs, dict) and 'policy' in obs:
            return obs['policy']
        return obs
    SACAgent.env_reset = patched_env_reset

    # 2) Patch env_step
    original_env_step = SACAgent.env_step
    def patched_env_step(self, actions):
        # RL Games calls self.vec_env.step(actions)
        result = self.vec_env.step(actions)

        # If your env is Gymnasium style -> (obs, reward, term, trunc, info)
        # RL Games merges `terminated` and `truncated` into `dones`,
        # but we can assume modern Gym by default:
        if len(result) == 5:
            obs, rewards, terminated, truncated, infos = result
            dones = terminated | truncated
        else:  # older style or unusual env
            obs, rewards, dones, infos = result

        # Extract obs['policy'] if it’s a dict
        if isinstance(obs, dict) and 'policy' in obs:
            obs = obs['policy']

        return obs, rewards, dones, infos
    SACAgent.env_step = patched_env_step

    print("[INFO] RL Games SAC has been patched for dict observations.")

def fix_rlgames_wrappers():
    """
    Optional patch for omni.isaac.lab_tasks.utils.wrappers.rl_games.RlGamesVecEnvWrapper
    to handle Gymnasium's (obs, info) reset output and ensure actions are tensors.
    """
    from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesVecEnvWrapper
    
    original_reset = RlGamesVecEnvWrapper.reset
    def patched_reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # If (obs, info) is returned, just process obs
        if isinstance(result, tuple) and len(result) == 2:
            obs, _info = result
            return self._process_obs(obs)
        return self._process_obs(result)

    original_step = RlGamesVecEnvWrapper.step
    def patched_step(self, actions):
        import torch
        import numpy as np
        # Convert actions to torch tensors on the correct device
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(device=self._sim_device)
        elif not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self._sim_device)
        return original_step(self, actions)

    RlGamesVecEnvWrapper.reset = patched_reset
    RlGamesVecEnvWrapper.step = patched_step
    print("[INFO] Patched RlGamesVecEnvWrapper for tuple resets and tensor actions.")