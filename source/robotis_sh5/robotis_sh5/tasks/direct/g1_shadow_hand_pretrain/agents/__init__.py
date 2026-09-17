"""skrl agent configs for the floating-hand dexterous pretrain (stage 1).

Why this package exists instead of reusing the sonic-residual one
----------------------------------------------------------------
skrl reads `experiment.directory` straight out of the yaml, so sharing the parent's file
would write stage-1 runs into `logs/skrl/g1_shadow_sonic_residual/` next to the full-body
training runs. Separate directory = separate tensorboard tree.

The PPO hyperparameters are a COPY of
`g1_shadow_sonic_residual/agents/skrl_ppo_cfg.yaml` taken when its md5 prefix was
`8ccbe4a88a6f`. If that file is retuned, re-sync this one by hand (the copy is
recorded so the drift is detectable: compare the md5 prefix in skrl_ppo_cfg.yaml's header).
Only `experiment.directory` differs.
"""
