"""DexAnyTwist task registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="Gx4m-DexAnyTwist-Direct-v0",
    entry_point=f"{__name__}.dexanytwist_env:DexAnyTwistEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexanytwist_env:DexAnyTwistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexAnyTwistPPORunnerCfg",
    },
)
