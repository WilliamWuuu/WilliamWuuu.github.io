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

## Experiments

All the experiments were conducted on a single NVIDIA H800 card with CUDA 12.4.


