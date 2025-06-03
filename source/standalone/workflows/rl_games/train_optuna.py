# train_optuna.py

import argparse
import sys
import os
import io
import re
import subprocess
import optuna
import yaml
from datetime import datetime
import traceback
import logging
import mlflow
MLFLOW_AVAILABLE = True
from omegaconf import OmegaConf

###############################################################################
#                                 Logger setup
###############################################################################

# Set up logging for train_optuna.py
logger = logging.getLogger("train_optuna")
logger.setLevel(logging.DEBUG)

# Ensure we don't add duplicate handlers if the file is imported multiple times
if not logger.handlers:
    # Handler 1: Write to file
    file_handler = logging.FileHandler("train_optuna.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Handler 2: Write to real stdout
    stdout_handler = logging.StreamHandler(sys.__stdout__)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)

logger.debug("Logger initialization test message")
logger.info("Starting training run")
logger.handlers[0].flush()

from omni.isaac.lab.app import AppLauncher


###############################################################################
#                           PARSE CMDLINE ARGS
###############################################################################
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games; optionally do Optuna HPO.")

# Primary flags for normal training
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (e.g. Isaac-Cartpole-v0).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--distributed", action="store_true", default=False, help="Multi-GPU or multi-node training.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="Initial policy std.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# Additional flags for hyperparam search
parser.add_argument("--optimize", action="store_true", default=False,
                    help="If set, run Optuna hyperparam search instead of normal training.")

# Let AppLauncher add its own CLI arguments
AppLauncher.add_app_launcher_args(parser)

# Parse the CLI
args_cli, hydra_args = parser.parse_known_args()

# If video is requested, enable offscreen cameras:
if args_cli.video:
    args_cli.enable_cameras = True

# For Hydra, strip out everything except the unrecognized extras:
sys.argv = [sys.argv[0]] + hydra_args

###############################################################################
# UTILITY: STORE METRIC
###############################################################################
def store_metric_in_yaml(metrics, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        if isinstance(metrics, dict):
            yaml.dump(metrics, f, default_flow_style=False)
        else:
            data = {"final_metric": float(metrics)}
            yaml.dump(data, f, default_flow_style=False)

###############################################################################
# TRAINING LOGIC
###############################################################################
def run_single_training(args_cli):
    """
    Launch Isaac Sim, run the training with RL-Games, parse the final average reward,
    and store it in a .yaml file for either normal training or hyperparam search.
    """
    # Launch the App
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import hydra
    from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

    @hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
    def _train_rl(env_cfg, agent_cfg):
        import math
        import random
        import os
        import re
        import gymnasium as gym
        import torch
        from datetime import datetime

        from rl_games.common import env_configurations, vecenv
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.torch_runner import Runner

        # Isaac Lab imports
        from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
        from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from omni.isaac.lab.utils.dict import print_dict
        from omni.isaac.lab.utils.io import dump_yaml, dump_pickle
        from omni.isaac.lab.utils.assets import retrieve_file_path
        from omni.isaac.lab_tasks.direct.cartpole.sac_utils import SACCompatibilityWrapper, patch_rlgames_sac, fix_rlgames_wrappers

        # --------------------------
        # Overwrite from CLI
        # --------------------------
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 999999)
        agent_cfg["params"]["seed"] = (
            args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
        )
        agent_cfg["params"]["config"]["max_epochs"] = (
            args_cli.max_iterations
            if args_cli.max_iterations is not None
            else agent_cfg["params"]["config"]["max_epochs"]
        )

        # If we have a checkpoint
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
            agent_cfg["params"]["load_checkpoint"] = False
            agent_cfg["params"]["load_path"] = ""
            print(f"[INFO] Loading checkpoint: {resume_path}")

        train_sigma = float(args_cli.sigma) if args_cli.sigma else None

        # Distributed training
        if args_cli.distributed:
            seed_offset = app_launcher.global_rank
            local_device_id = app_launcher.local_rank
            agent_cfg["params"]["seed"] += seed_offset
            agent_cfg["params"]["config"]["device"] = f"cuda:{local_device_id}"
            agent_cfg["params"]["config"]["device_name"] = f"cuda:{local_device_id}"
            agent_cfg["params"]["config"]["multi_gpu"] = True
            env_cfg.sim.device = f"cuda:{local_device_id}"

        env_cfg.seed = agent_cfg["params"]["seed"]


        # Update only the reward-related keys in env_cfg.
        reward_keys = [
            "rew_scale_alive",
            "rew_scale_pole_angle",
            "rew_scale_cart_vel",
            "rew_scale_pole_vel",
            "rew_scale_cart_center",
            "rew_scale_pole_straight",
            "rew_scale_pole_upwards",
            "rew_scale_exp_vel",
            "vel_scale_factor",
            "rew_scale_bonus_first",
            "rew_scale_bonus_second",
            "rew_scale_bonus_third"
        ]

        # Check if agent_cfg["params"] has an "env_cfg" section.
        if "env_cfg" in agent_cfg["params"]:
            env_overrides = agent_cfg["params"]["env_cfg"]
            for key in reward_keys:
                if key in env_overrides:
                    # For OmegaConf DictConfig, update using dictionary indexing.
                    setattr(env_cfg, key, env_overrides[key])
                    logger.info(f"Overwriting {key} to {getattr(env_cfg, key)}")

        # Verify updated values
        logger.info("Final reward parameters in env_cfg:")
        for key in reward_keys:
            logger.info(f"{key}: {getattr(env_cfg, key)}")

        # (Optional) Log a few updated values for verification:
        logger.info("Final env config reward parameters:")
        logger.info(f"rew_scale_alive: {env_cfg.rew_scale_alive}")
        logger.info(f"rew_scale_pole_angle: {env_cfg.rew_scale_pole_angle}")
        logger.info(f"rew_scale_cart_vel: {env_cfg.rew_scale_cart_vel}")

        # --------------------------
        # Logging directory
        # --------------------------
        log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = agent_cfg["params"]["config"].get(
            "full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        print(f"[CHILD] root_path: {log_root_path}")
        print(f"[CHILD] full_experiment_name: {log_dir}")

        os.makedirs(os.path.join(log_root_path, log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

        # --------------------------
        # Create gym environment
        # --------------------------
        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None
        )

        # Convert multi-agent if needed
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # Optional video capture
        if args_cli.video:
            from gymnasium.wrappers import RecordVideo
            video_kwargs = {
                "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = RecordVideo(env, **video_kwargs)


        # Check if we're using SAC vs PPO
        is_sac = agent_cfg["params"]["algo"]["name"].lower() == "sac"

        # Apply patches if using SAC
        if is_sac:
            print("[INFO] Applying SAC compatibility patches")
            # Apply the patch to RLGamesVecEnvWrapper first
            fix_rlgames_wrappers()
            # Then patch the SAC agent
            patch_rlgames_sac()
            # Finally wrap the environment
            env = SACCompatibilityWrapper(env)

        # RL-Games VecEnv wrapper
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
        vecenv.register("IsaacRlgWrapper", lambda c_name, num_actors, **kw:
                        RlGamesGpuEnv(c_name, num_actors, **kw))
        env_configurations.register("rlgpu", {
            "vecenv_type": "IsaacRlgWrapper",
            "env_creator": lambda **kw: env
        })
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

        # --------------------------
        # RL-Games Runner
        # --------------------------
                # RL-Games Runner
        runner = Runner(IsaacAlgoObserver())
        runner.load(agent_cfg)
        runner.reset()

        #Cature Rl-games stdout in a buffer
        old_stdout = sys.stdout
        log_buffer = io.StringIO()
        sys.stdout = log_buffer


        if args_cli.checkpoint:
            stats = runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
        else:
            stats = runner.run({"train": True, "play": False, "sigma": train_sigma})

        metrics = {
            "final_balance_count": env.unwrapped.episode_info.get("final_epoch_height", 0.0)  # Access through unwrapped env
        }


        # Restore stdout
        sys.stdout = old_stdout

        logs = log_buffer.getvalue()
        print("[DEBUG] RL-Games logs:\n", logs)

        checkpoint_pattern = re.compile(
            r"=> saving checkpoint '(?P<path>[^']*rew__([0-9.]+)_\.pth)'"
        )
        match = checkpoint_pattern.search(logs)
        if match:
            full_path = match.group("path")
            # parse out the reward
            # If group(2) is the decimal group, you can do:
            final_reward_str = match.group(2)
            final_average_reward = float(final_reward_str)
            print(f"[INFO] Final reward from checkpoint name: {final_average_reward}")
        else:
            print("[WARNING] Could not parse checkpoint reward from logs => fallback=0.0")
            final_average_reward = 0.123

        # ------------------------------------------------------------------
        # Store final metric (average reward) in results.yaml
        # ------------------------------------------------------------------
        output_metric_file = os.path.join(log_root_path, log_dir, "results.yaml")
        metrics = {
            "final_metric": final_average_reward,
            "final_balance_count": env.unwrapped.episode_info.get("final_balance_count", 0.0)
        }
        logger.debug(f"Final metrics: {metrics}")
        store_metric_in_yaml(metrics, output_metric_file)

        if MLFLOW_AVAILABLE:
            try:
                import mlflow
                experiment_name = "triple_cart_test_1"
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                with mlflow.start_run(nested=True):
                    mlflow.log_params({
                        # Network params
                        #"units_per_layer": agent_cfg["params"]["network"]["mlp"]["units"][0],
                        #"n_layers": len(agent_cfg["params"]["network"]["mlp"]["units"]),
                        "activation": agent_cfg["params"]["network"]["mlp"]["activation"],
                        
                        # PPO params
                        "gamma": agent_cfg["params"]["config"]["gamma"],
                        "tau": agent_cfg["params"]["config"]["tau"],
                        "learning_rate": agent_cfg["params"]["config"]["learning_rate"],
                        "entropy_coef": agent_cfg["params"]["config"]["entropy_coef"],
                        "critic_coef": agent_cfg["params"]["config"]["critic_coef"],
                        
                        # Reward scales
                        #"rew_scale_alive": agent_cfg["params"]["env_cfg"]["rew_scale_alive"],
                        #"rew_scale_pole_angle": agent_cfg["params"]["env_cfg"]["rew_scale_pole_angle"],
                        #"rew_scale_cart_vel": agent_cfg["params"]["env_cfg"]["rew_scale_cart_vel"], 
                        #"rew_scale_pole_vel": agent_cfg["params"]["env_cfg"]["rew_scale_pole_vel"],
                        #"rew_scale_pole_straight": agent_cfg["params"]["env_cfg"]["rew_scale_pole_straight"],
                        #"rew_scale_pole_upwards": agent_cfg["params"]["env_cfg"]["rew_scale_pole_upwards"],
                        #"rew_scale_exp_vel": agent_cfg["params"]["env_cfg"]["rew_scale_exp_vel"],
                        #"vel_scale_factor": agent_cfg["params"]["env_cfg"]["vel_scale_factor"],
                        #"rew_scale_cart_center": agent_cfg["params"]["env_cfg"]["rew_scale_cart_center"],
                        #"rew_scale_bonus_first": agent_cfg["params"]["env_cfg"]["rew_scale_bonus_first"],
                        #"rew_scale_bonus_second": agent_cfg["params"]["env_cfg"]["rew_scale_bonus_second"],
                        #"rew_scale_bonus_third": agent_cfg["params"]["env_cfg"]["rew_scale_bonus_third"]
                    })
                    mlflow.log_metrics({
                        "final_reward": final_average_reward,
                        "final_balance_count": metrics["final_balance_count"]
                    })
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        env.close()

    _train_rl()
    simulation_app.close()

###############################################################################
# SUBPROCESS + OPTUNA
###############################################################################
def run_training_subprocess(overrides_dict):
    """
    Spawns a subprocess to re-run this script with specific Hydra overrides,
    returning the final metric from results.yaml or None on failure.
    """
    cli_args = []
    for k, v in overrides_dict.items():
        val_str = "true" if isinstance(v, bool) and v else "false" if isinstance(v, bool) else str(v)
        cli_args.append(f"{k}={val_str}")
    
    cmd = [
        sys.executable, __file__,
        "--headless",
        f"--task={args_cli.task if args_cli.task else 'Isaac-Cartpole-v0'}",
    ]
    cmd += cli_args

    print("[Optuna Subprocess] Launching:", cmd)
    try:
        completed_proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        print("[Optuna Subprocess] ERROR launching subprocess:", e)
        return None

    if completed_proc.returncode != 0:
        print("[Optuna Subprocess] Non-zero exit code.")
        print("[Optuna Subprocess] stdout:\n", completed_proc.stdout)
        print("[Optuna Subprocess] stderr:\n", completed_proc.stderr)
        return None

    lines = completed_proc.stdout.splitlines()
    child_root = None
    child_subfolder = None

    for line in lines:
        if "[CHILD] root_path:" in line:
            child_root = line.split("root_path:")[1].strip()
        elif "[CHILD] full_experiment_name:" in line:
            child_subfolder = line.split("full_experiment_name:")[1].strip()

    if child_root and child_subfolder:
        results_path = os.path.join(child_root, child_subfolder, "results.yaml")
        print("[Optuna Subprocess] Looking for results.yaml at:", results_path)
        try:
            with open(results_path, "r") as f:
                data = yaml.safe_load(f)
            final_balance_count = data.get("final_balance_count", None)
            if final_balance_count is not None:
                final_balance_count = float(final_balance_count)
            print("[Optuna Subprocess] Found final height:", final_balance_count)
            return final_balance_count
        except Exception as e:
            print("[Optuna Subprocess] Could not read metrics:", e)
    
    print("[Optuna Subprocess] Could not parse child lines => fallback None")
    return None

def run_multiple_seeds_for_trial(overrides_dict, n_seeds=1):
    """
    Repeat the same hyperparam config multiple times with different seeds,
    returning the average final metric across seeds.
    """
    import random
    metrics = []
    for _ in range(n_seeds):
        seed = random.randint(0, 999999)
        local_overrides = dict(overrides_dict)
        local_overrides["agent.params.seed"] = seed
        metric = run_training_subprocess(local_overrides)
        if metric is not None:
            metrics.append(metric)

    if len(metrics) == 0:
        return None
    return sum(metrics) / len(metrics)

def optuna_objective(trial):
    # Network hyperparams
    n_layers = trial.suggest_int("n_layers", 4, 6)
    #units_per_layer = trial.suggest_int("units_per_layer", 64, 128, step=32)
    units_per_layer = 256
    #activation = trial.suggest_categorical("activation", ["relu", "elu"])
    units = [units_per_layer] * n_layers

    # PPO hyperparams
    #gamma = trial.suggest_float("agent.params.config.gamma", 0.97, 0.995)
    #tau = trial.suggest_float("agent.params.config.tau", 0.96, 0.99)
    #learning_rate = trial.suggest_float("agent.params.config.learning_rate", 8e-5, 5e-4, log=True)
    #entropy_coef = trial.suggest_float("agent.params.config.entropy_coef", 0.005, 0.015, step=0.001)
    critic_coef = trial.suggest_float("agent.params.config.critic_coef", 4, 12)
   
    # Reward scale hyperparams
    #rew_alive = trial.suggest_float("agent.params.env_cfg.rew_scale_alive", 1, 2.5, step=0.1)
    #rew_angle = trial.suggest_float("agent.params.env_cfg.rew_scale_pole_angle", 2, 4.0, step=0.2)
    #rew_cart_vel = trial.suggest_float("agent.params.env_cfg.rew_scale_cart_vel", -0.2, 0, step=0.01)
    #rew_pole_vel = trial.suggest_float("agent.params.env_cfg.rew_scale_pole_vel", -0.2, 0, step=0.05)
    #rew_straight = trial.suggest_float("agent.params.env_cfg.rew_scale_pole_straight", 0.0, 0.5, step=0.5)
    #rew_upwards = trial.suggest_float("agent.params.env_cfg.rew_scale_pole_upwards", 0, 4.0, step=0.5)
    #rew_scale_exp_vel = trial.suggest_float("agent.params.env_cfg.rew_scale_exp_vel", -1, 0.0, step=0.1)
    #vel_scale_factor = trial.suggest_float("agent.params.env_cfg.vel_scale_factor", 10, 25.0, step=1)
    #rew_cart_center = trial.suggest_float("agent.params.env_cfg.rew_scale_cart_center", 0, 1.0, step=0.1)
    #rew_bonus_first = trial.suggest_float("agent.params.env_cfg.rew_scale_bonus_first", 0, 300, step=25)
    #rew_bonus_second = trial.suggest_float("agent.params.env_cfg.rew_scale_bonus_second", 0, 300.0, step=25)
    #rew_bonus_third = trial.suggest_float("agent.params.env_cfg.rew_scale_bonus_third", 0, 300.0, step=25)


    overrides = {

        "agent.params.network.mlp.units": units,
        #"agent.params.config.gamma": gamma,
        #"agent.params.config.tau": tau,
        #"agent.params.config.learning_rate": learning_rate,
        #"agent.params.config.entropy_coef": entropy_coef,
        "agent.params.config.critic_coef": critic_coef,

        #"agent.params.env_cfg.rew_scale_alive": rew_alive,
        #"agent.params.env_cfg.rew_scale_pole_angle": rew_angle,
        #"agent.params.env_cfg.rew_scale_cart_vel": rew_cart_vel,
        #"agent.params.env_cfg.rew_scale_pole_vel": rew_pole_vel,
        #"agent.params.env_cfg.rew_scale_pole_straight": rew_straight,
        #"agent.params.env_cfg.rew_scale_pole_upwards": rew_upwards,
        #"agent.params.env_cfg.rew_scale_exp_vel": rew_scale_exp_vel,
        #"agent.params.env_cfg.vel_scale_factor": vel_scale_factor,
        #"agent.params.env_cfg.rew_scale_cart_center": rew_cart_center,
        #"agent.params.env_cfg.rew_scale_bonus_first": rew_bonus_first,
        #"agent.params.env_cfg.rew_scale_bonus_second": rew_bonus_second,
        #"agent.params.env_cfg.rew_scale_bonus_third": rew_bonus_third,

        "agent.params.config.name": (
            f"critic{critic_coef:.2f}"
        )
    }

    final_metric = run_multiple_seeds_for_trial(overrides, n_seeds=1)
    return float("nan") if final_metric is None else final_metric

def run_optuna_search():
    """
    Launch an Optuna study to find hyperparams that maximize the RL agent's 
    average reward. 
    """
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("triple_cart_test_1")

    study = optuna.create_study(
    study_name="triple_cart_test_1",
    storage="sqlite:///triple_cart_test_1.db",
    load_if_exists=True,
    direction="maximize"
    )
    
    #Good start trial
    """
    study.enqueue_trial({
        "agent.params.env_cfg.rew_scale_alive": 2,
        "agent.params.env_cfg.rew_scale_pole_angle": 2.5,
        "agent.params.env_cfg.rew_scale_cart_vel": -0.02,
        "agent.params.env_cfg.rew_scale_pole_vel": -0.05,
        "agent.params.env_cfg.rew_scale_pole_straight": 0.0,
        "agent.params.env_cfg.rew_scale_pole_upwards": 0.0,
        "agent.params.env_cfg.rew_scale_exp_vel": -0.0,
        "agent.params.env_cfg.vel_scale_factor": 15,
        "agent.params.env_cfg.rew_scale_cart_center": 0.2,
        "agent.params.env_cfg.rew_scale_bonus_first": 100,
        "agent.params.env_cfg.rew_scale_bonus_second": 200,
        "agent.params.env_cfg.rew_scale_bonus_third": 300,
    })
    """

    n_trials = 20
    print(f"[Optuna] Starting study with {n_trials} trials...")
    study.optimize(optuna_objective, n_trials=n_trials)
    print("[Optuna] Study complete!")
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    print("\nTop 10 trials:")
    trials_df = study.trials_dataframe()
    trials_df_sorted = trials_df.sort_values('value', ascending=False)
    print(trials_df_sorted.head(10))

###############################################################################
# ENTRY POINT
###############################################################################
def entry_point():
    """
    Decide whether to do a normal run or an Optuna hyperparam search.
    """
    if hasattr(args_cli, "optimize") and args_cli.optimize:
        run_optuna_search()
    else:
        run_single_training(args_cli)

if __name__ == "__main__":
    entry_point()