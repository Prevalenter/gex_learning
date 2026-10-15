"""GX4M in-hand reorientation task registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="Gx4m-InHandReorient-Direct-v0",
    entry_point=f"{__name__}.inhand_reorient_env:Gx4mInHandReorientEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_reorient_env:Gx4mInHandReorientEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gx4mInHandReorientPPORunnerCfg",
    },
)
