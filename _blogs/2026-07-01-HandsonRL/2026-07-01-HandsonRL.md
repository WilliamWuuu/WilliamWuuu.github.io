---
layout: blog
title: 'My Bizarre Adventure in RL: Hands On'
date: 2026-07-01
permalink: /posts/2026/7/HandsonRL/
image_path: /blog-assets/2026-06-28-HandsonRL/img/
tags:
  - Reinforcement Learning
---

Just a rush of blood to the head, I've decided to try my hand at some RL training for LLMs.

## Code

Most of the code comes from the following two repositories:
- [verl: A Flexible and Efficient RL Post-Training Framework](https://github.com/verl-project/verl)
- [JustRL: Scaling a 1.5B LLM with a Simple RL Recipe](https://github.com/thunlp/JustRL)

### PPO

Recall that the optimization objective of PPO is formed as:

$$
\begin{equation}
\begin{align*}
\mathcal{J}_{PPO}(\theta) 
= & \mathbb{E}_{q\sim P(Q),o\sim \pi_{\theta_{old}}(O\vert q)}
\frac{1}{\vert o\vert}\sum_{t=1}^{\vert o\vert} \\
& \min\left[
\frac{\pi_\theta(o_t\vert q, o_{<t})}{\pi_{\theta_{old}}(o_t\vert q, o_{<t})} A_t,
\mathrm{clip}\left(\frac{\pi_\theta(o_t\vert q, o_{<t})}{\pi_{\theta_{old}}(o_t\vert q, o_{<t})}, 1-\varepsilon, 1+\varepsilon\right) A_t
\right],
\end{align*}
\end{equation}
$$

where $q, o$ are questions and outputs sampled from the question dataset and old policy $\pi_{\theta_{old}}$, resepectively.

We'll start by implementing the advantage calculation. Consider computing advantage $A_t$ with generalized advantage estimation (GAE):

$$
\begin{equation}
A_t=\delta_t+(\gamma\lambda)\delta_{t+1}+\dots+(\gamma\lambda)^{T-t+1}\delta_{T-1}
\quad\text{where}\quad
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
\end{equation}
$$

we can see that the recursive formula for the advantage can be expressed as:

$$
\begin{equation}
A_t = \delta_t + (\gamma\lambda)A_{t+1}.
\end{equation}
$$

Thus we can write the function as

```Python
import verl.utils.torch_functional as verl_F

def compute_gae_advantage(
    token_level_rewards: torch.Tensor,  # (B,T) Reward of every token.
    values: torch.Tensor,  # (B,T) State value from the critic model.
    response_mask: torch.Tensor,  #(B,T) [EOS] mask.
    gamma: torch.Tensor,  # Discount factor.
    lam: torch.Tensor,  # Lambda.
):
    with torch.no_grad():
        last_gae_lam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        # t = T-1, T-2, ..., 0
        for t in reversed(range(gen_len)):
            next_values = values[:, t + 1] if t < gen_len - 1 else 0.0
            # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = token_level_rewards[:, t] + gamma * next_values - values[:, t]
            # A_t = delta_t + gamma * lambda * A_{t+1}
            last_gae_lam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(last_gae_lam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)  # (B,T)

        # R_t = A_t + V(s_t)
        returns = advantages + values
        # Whitening
        ## Essentially equivalent to standardization,
        ## i.e. \hat{A} = (A - mu) / (sigma + epsilon)
        # Mask
        ## In NLP, a batch is typically: [ prompt tokens | response tokens | padding ].
        ## Only response tokens are needed for policy gradient.
        ## So we only calculate the mean/std on tokens where mask=1, 
        ## and normalize the advantage only at those specific positions.
        advantages = .masked_whiten(advantages, response_mask)
    return advantages, returns
```

Then we can move on to the whole policy loss. But before implementation, a trick that is very commonly used in practical PPO training needs to be introduced, Dual-Clip.

Consider the case where the advantage is positive. When $\frac{\pi_\theta}{\pi_{old}}$ increases, loss tends to decrease, which is exactly the optimizer wants to see. Meanwhile, we want to control this ratio to prevent it from getting too large. The clip mechanism does a great job of achieving this.

But let's look at the case when the advantage is negative. Increasing this probability ratio now increases the loss, so the optimizer is incentivized to reduce this ratio, i.e., decrease the probability of the sampled action. This is consistent with discouraging bad actions. However, an important subtlety is that this lower clipping does not symmetrically bound the magnitude of the loss when r grows large. In the region where the ratio is greater than $1+\epsilon$, the unclipped term dominates for $A<0$ and the loss continues to grow approximately linearly with the ratio. As a result, extremely large ratios combined with negative advantages can still produce disproportionately large loss contributions. So we need an extra bound to address this instability, which leads to the introduction of dual-clip.

```Python
import verl.utils.torch_functional as verl_F

def compute_policy_loss(
    old_log_prob,  # log(π_old(a|s))
    log_prob,  # log(π_θ(a|s))
    advantages,
    response_mask,
    cliprange=None,  # ε
    cliprange_low=None,  # Specified lower bound of clip.
    cliprange_high=None,  # Specified higher bound of clip.
    clip_ratio_c=3.0,  # Lower bound ratio of dual-clip
    loss_agg_mode: str = "token-mean",
):
    assert clip_ratio_c > 1.0

    # π_θ(a|s) / π_old(a|s) = e ^ (log(π_θ(a|s)) - log(π_old(a|s)))
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    
    # E[log(π_old(a|s)) - log(π_θ(a|s))]
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CPI loss
    pg_losses1 = -advantages * ratio
    
    # Clip range: clip or dual-clip
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    # Clip loss
    ## - clip(ratio, 1-cliprange, 1+cliprange) * A
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    ## max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    # Calculate the proportion of clipping occurrences. 
    ## Too high: clipping is too strong and learning is being restricted; 
    ## Too low: clipping is too weak.
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), 
        response_mask
    )

    # Dual-Clip loss
    pg_losses3 = -advantages * clip_ratio_c
    ## min(- A * c, L_{CLIP})
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    ## Monitor whether negative advantage is being excessively restricted.
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), 
        response_mask
    )

    # Final loss
    ## Positive A: clip
    ## Negative A: dual-clip
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    # Aggregate loss tensor into scalar value based on the chosen mode.
    pg_loss = agg_loss(
        loss_mat=pg_losses, 
        loss_mask=response_mask, 
        loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
```

Remember the squared error and the entropy terms that were omitted? 

```Python
def compute_value_loss(
    values_pred: torch.Tensor,  # Predicted values.
    values: torch.Tensor,  # Baseline values.
    returns: torch.Tensor,  # Ground truth returns.
    response_mask: torch.Tensor, 
    cliprange_value: float,  # Clip range for predicted values.
    loss_agg_mode: str = "token-mean"
):
    values_pred_clipped = verl_F.clip_by_value(
        values_pred, 
        values - cliprange_value, 
        values + cliprange_value
    )
    vf_losses1 = (values_pred - returns) ** 2
    vf_losses2 = (values_pred_clipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(
        loss_mat=clipped_vf_losses, 
        loss_mask=response_mask, 
        loss_agg_mode=loss_agg_mode
    )
    vf_clipfrac = verl_F.masked_mean(
        torch.gt(vf_losses2, vf_losses1).float(), 
        response_mask
    )
    return vf_loss, vf_clipfrac

def compute_entropy_loss(
    logits, 
    response_mask, 
    loss_agg_mode: str = "token-mean"
):
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(
        loss_mat=token_entropy, 
        loss_mask=response_mask, 
        loss_agg_mode=loss_agg_mode
    )
    return entropy_loss
```

Once all these key components are complete, we can look into building the  trainer for PPO.

{% include widgets/blog_image.html src="design.png" caption="Picture 1" %}

```Python
class PPOTrainer:
    def __init__(self, config, tokenizer, ...):
        """Constructor of the trainer."""
        
        self.config = config
        # ... (omitted)

    def init_workers(self):
        """Worker groups for each role (actor, critic, etc.)."""

        self.critic_wg = all_wg["critic"]
        self.critic_wg.init_model()

        self.ref_policy_wg = all_wg["ref"]
        self.ref_policy_wg.init_model()

        self.rm_wg = all_wg["rm"]
        self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        # ... (omitted)

    def fit(self):
        """Minimal training loop."""

        self.global_steps = 0

        # Load checkpoint.
        self._load_checkpoint()

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # DataProto 
                ## A data structure that aims to provide a standard protocol for data exchange between functions.
                ## Contains a batch (TensorDict) and a meta_info (Dict).
                # TensorDict
                ## Allows you to manipulate a dictionary of Tensors like a single Tensor. 
                ## https://docs.pytorch.org/tensordict/stable/index.html
                batch = DataProto.from_single_dict(batch_dict)
                
                # ===== rollout =====
                gen_inputs = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )
                rollout_output = self.actor_rollout_wg.generate_sequences(gen_inputs)
                batch = batch.union(rollout_output)
                
                batch.batch["response_mask"] = compute_response_mask(batch)

                # ===== reward =====
                reward_tensor, _ = compute_reward(batch, self.reward_fn)
                batch.batch["token_level_rewards"] = reward_tensor

                # ===== reference policy KL penalty =====
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)
                batch = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward)

                # ===== critic value =====
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

                # ===== advantage estimation =====
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                )

                # ===== critic update =====
                critic_out = self.critic_wg.update_critic(batch)

                # ===== actor update =====
                actor_out = self.actor_rollout_wg.update_actor(batch)

                # ===== end of the loop =====
                self.global_steps += 1
                if self.global_steps >= self.total_training_steps:
                    return
```

```Python
class DataParallelPPOActor:
    def __init__(
        self, 
        config, 
        actor_module: nn.Module, 
        actor_optimizer: torch.optim.Optimizer
    ):
        """Constructor of the actor model."""

        self.config = config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        # ... (omitted)

    def compute_log_prob(self, data: DataProto):
        # ... (omitted)

    def update_policy(self, data: DataProto):
        self.actor_module.train()
        
        metrics = {}

        # ===== extract PPO training fields =====
        batch = data.select(
            batch_keys=[
                "responses",
                "attention_mask",
                "old_log_probs",
                "advantages",
                "ref_log_prob"
            ]
        ).batch

        multi_turn = data.meta_info.get("multi_turn", False)

        # ===== minibatch split =====
        mini_batches = batch.split(self.config.ppo_mini_batch_size)

        for epoch in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:

                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for mb in micro_batches:
                    mb = mb.to(get_torch_device().current_device())

                    responses = mb["responses"]
                    advantages = mb["advantages"]
                    old_log_prob = mb["old_log_probs"]

                    response_len = responses.size(1)

                    # ===== response mask =====
                    attention_mask = mb["attention_mask"]
                    if multi_turn:
                        response_mask = mb.get("loss_mask", attention_mask)[:, -response_len:]
                    else:
                        response_mask = attention_mask[:, -response_len:]

                    # ===== forward policy =====
                    _, log_prob = self._forward_micro_batch(mb)

                    # ===== PPO clipped objective =====
                    pg_loss, clipfrac, kl_est = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=self.config.clip_ratio,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )

                    loss = pg_loss

                    # ===== entropy bonus =====
                    if self.config.entropy_coeff > 0:
                        entropy, _ = self._forward_micro_batch(
                            mb,
                            calculate_entropy=True
                        )
                        entropy_loss = agg_loss(
                            entropy,
                            response_mask,
                            self.config.loss_agg_mode
                        )
                        loss -= self.config.entropy_coeff * entropy_loss

                    # ===== KL penalty to reference policy =====
                    if self.config.use_kl_loss:
                        ref_log_prob = mb["ref_log_prob"]
                        kl = kl_penalty(log_prob, ref_log_prob, self.config.kl_loss_type)
                        kl_loss = agg_loss(kl, response_mask, self.config.loss_agg_mode)
                        loss += self.config.kl_loss_coef * kl_loss
                        metrics.setdefault("actor/kl_loss", []).append(kl_loss.item())

                    loss = loss / len(micro_batches)
                    loss.backward()

                    metrics.setdefault("actor/pg_loss", []).append(pg_loss.item())
                    metrics.setdefault("actor/clipfrac", []).append(clipfrac.item())
                    metrics.setdefault("actor/kl_est", []).append(kl_est.item())

                # ===== optimizer step =====
                grad_norm = self._optimizer_step()
                metrics.setdefault("actor/grad_norm", []).append(grad_norm.item())

        self.actor_optimizer.zero_grad()
        return metrics
```

```Python
class PPOCritic:
    def __init__(
        self, 
        config, 
        critic_module: nn.Module, 
        critic_optimizer: optim.Optimizer
    ):
        """Constructor of the critic model."""
        
        self.config = config
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        # ... (omitted)

    def compute_values(self, data: DataProto) -> torch.Tensor:
        # ... (omitted)

    def update_critic(self, data: DataProto):
        self.critic_module.train()
        metrics = {}

        # ===== extract training fields =====
        batch = data.select(
            batch_keys=["responses", "attention_mask", "values", "returns"]
        ).batch

        # ===== minibatch split =====
        mini_batches = batch.split(self.config.ppo_mini_batch_size)

        for epoch in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:

                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()

                for mb in micro_batches:
                    mb = mb.to(get_torch_device().current_device())

                    responses = mb["responses"]
                    attention_mask = mb["attention_mask"]
                    values = mb["values"]
                    returns = mb["returns"]

                    response_mask = attention_mask[:, -responses.size(1):]

                    # ===== forward value function =====
                    vpreds = self._forward_micro_batch(mb)

                    # ===== PPO value loss=====
                    vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                        vpreds=vpreds,
                        values=values,
                        returns=returns,
                        response_mask=response_mask,
                        cliprange_value=self.config.cliprange_value,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )

                    loss = vf_loss / len(micro_batches)
                    loss.backward()

                    metrics.setdefault("critic/vf_loss", []).append(vf_loss.item())
                    metrics.setdefault("critic/vf_clipfrac", []).append(vf_clipfrac.item())

                # ===== optimizer step =====
                grad_norm = self._optimizer_step()
                metrics.setdefault("critic/grad_norm", []).append(grad_norm.item())

        self.critic_optimizer.zero_grad()
        return metrics
```

### GRPO

## Experiments

All the experiments were conducted on a single NVIDIA H800 card with CUDA 12.4.


