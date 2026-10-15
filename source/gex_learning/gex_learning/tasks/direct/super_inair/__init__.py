"""GX4M Super In-Air tennis rotation task registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="Gx4m-SuperInAir-Rotation-Direct-v0",
    entry_point=f"{__name__}.rotation_env:Gx4mSuperInAirRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotation_env:Gx4mSuperInAirRotationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gx4mSuperInAirPPORunnerCfg",
    },
)
