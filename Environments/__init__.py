from gym.envs.registration import register

register( 
    id="Crosswalk_hybrid_multi_coop_4cars-v0",
    entry_point="Environments.Env_hybrid_multi_coop_4cars:Crosswalk_hybrid_multi_coop_4cars",
    max_episode_steps=100,
    reward_threshold=100.0,
)

register( 
    id="Crosswalk_hybrid_multi_coop_4cars2-v0",
    entry_point="Environments.Env_hybrid_multi_coop_4cars2:Crosswalk_hybrid_multi_coop_4cars2",
    max_episode_steps=100,
    reward_threshold=100.0,
)

register( 
    id="Crosswalk_hybrid_multi_coop-v0",
    entry_point="Environments.Env_hybrid_multi_coop:Crosswalk_hybrid_multi_coop",
    max_episode_steps=100,
    reward_threshold=100.0,
)#scalable
register( 
    id="Crosswalk_hybrid_multi_coop_scalable-v0",
    entry_point="Environments.Env_hybrid_multi_coop_scalable:Crosswalk_hybrid_multi_coop_scalable",
    max_episode_steps=100,
    reward_threshold=100.0,
)

register( # et coop
    id="Crosswalk_hybrid_multi_stop-v0",
    entry_point="Environments.Env_hybrid_multi_stop:Crosswalk_hybrid_multi_stop",
    max_episode_steps=100,
    reward_threshold=100.0,
)

register( 
    id="Crosswalk_hybrid_multi_naif-v0",
    entry_point="Environments.Env_hybrid_multi_naif:Crosswalk_hybrid_multi_naif",
    max_episode_steps=100,
    reward_threshold=100.0,
)