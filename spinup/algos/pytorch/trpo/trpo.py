from .core import mlp_actor_critic


def trpo(
    env_fn,
    actor_critic=mlp_actor_critic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    delta=0.01,
    vf_lr=1e-3,
    train_v_iters=80,
    damping_coeff=0.1,
    cg_iters=10,
    backtrack_iters=10,
    backtrack_coeff=0.8,
    lam=0.97,
    max_ep_len=1000,
    logger_kwargs=dict(),
    save_freq=10,
    algo="trpo",
):
    pass
