Initialize policy/value network parameters θ (and maybe separate φ)
Initialize optimizer(θ)
Initialize vectorized environments envs (n_envs)

for update = 1,2,3,...:

    # ============================================================
    # (A) ROLLOUT: collect on-policy data with π_old = πθ
    # ============================================================
    Clear buffers: obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf

    obs = envs.reset_if_needed()

    for t = 0 .. n_steps-1:
        with no_grad:
            # sample action from current policy
            action, logp, entropy, value = agent.action_and_value(obs)

        next_obs, reward, done, info = envs.step(action)

        store obs_buf[t]  = obs
        store act_buf[t]  = action
        store rew_buf[t]  = reward
        store done_buf[t] = done
        store val_buf[t]  = value           # V_old(s_t)
        store logp_buf[t] = logp            # log π_old(a_t|s_t)

        obs = next_obs

    with no_grad:
        next_value = agent.value(obs)       # V_old(s_{T}) for bootstrap


    # ============================================================
    # (B) COMPUTE TARGETS: GAE(General Advantage Estimation) 
    #                      V_target = advantages + returns
    # ============================================================
    advantages = zeros(n_steps, n_envs)
    returns    = zeros(n_steps, n_envs)

    gae = 0
    for t = n_steps-1 down to 0:
        nonterminal = 1 - done_buf[t]
        delta = rew_buf[t] + γ * next_value * nonterminal - val_buf[t]
        gae   = delta + γ * λ * nonterminal * gae
        advantages[t] = gae
        returns[t]    = advantages[t] + val_buf[t]    # V_target

        next_value = val_buf[t]    # shift bootstrap base (or keep separate, either works)


    # ============================================================
    # (C) FLATTEN BATCH: (n_steps,n_envs,...) -> (batch_size,...)
    # ============================================================
    b_obs  = flatten(obs_buf)
    b_act  = flatten(act_buf)
    b_logp = flatten(logp_buf)        # old logprob
    b_val  = flatten(val_buf)         # old value V_old
    b_adv  = flatten(advantages)
    b_ret  = flatten(returns)         # target returns V_target

    if norm_adv:
        b_adv = (b_adv - mean(b_adv)) / (std(b_adv) + 1e-8)


    # ============================================================
    # (D) LEARN: PPO update on same batch for K epochs
    # ============================================================
    inds = [0..batch_size-1]

    for epoch = 1..n_epochs:
        shuffle(inds)

        for each minibatch mb_inds from inds with size minibatch_size:

            # ---- evaluate current policy/value on the OLD actions(forward pass) ----
            newlogp, entropy, newvalue = agent.evaluate(b_obs[mb_inds], b_act[mb_inds])

            # ---- ratio for PPO ----
            logratio = newlogp - b_logp[mb_inds]
            ratio    = exp(logratio)

            # ---- approximate KL (monitor) ----
            approx_kl = mean( b_logp[mb_inds] - newlogp )
            clipfrac  = mean( abs(ratio - 1) > ε )

            # ---- policy loss (clipped surrogate) ----
            unclipped = ratio * b_adv[mb_inds]
            clipped   = clip(ratio, 1-ε, 1+ε) * b_adv[mb_inds]
            pg_loss   = -mean( min(unclipped, clipped) )

            # ---- value loss (optional value clipping) ----
            if clip_vloss:
                v_unclipped = (newvalue - b_ret[mb_inds])^2
                v_clipped_value = b_val[mb_inds] + clip(newvalue - b_val[mb_inds], -ε_v, +ε_v)
                v_clipped  = (v_clipped_value - b_ret[mb_inds])^2
                v_loss     = 0.5 * mean( max(v_unclipped, v_clipped) )
            else:
                v_loss     = 0.5 * mean( (newvalue - b_ret[mb_inds])^2 )

            # ---- entropy bonus ----
            ent_loss = mean(entropy)

            # ---- total loss ----
            loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss

            # ---- gradient step ----
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(θ, max_grad_norm)
            optimizer.step()

        # ---- early stop on KL (trust region safety) ----
        if target_kl is not None and approx_kl > target_kl:
            break


    # ============================================================
    # (E) LOGGING / METRICS (optional)
    # ============================================================
    explained_var = 1 - Var(b_ret - b_val) / Var(b_ret)
    log pg_loss, v_loss, entropy, approx_kl, clipfrac, explained_var, reward stats, etc.

end for
