"""GX4M ducky grasp task registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="Gx4m-DexAnyGrasp-Direct-v0",
    entry_point=f"{__name__}.dexanygrasp_env:DexAnyGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexanygrasp_env:DexAnyGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexAnyGraspPPORunnerCfg",
    },
)
