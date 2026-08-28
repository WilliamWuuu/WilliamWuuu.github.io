---
layout: blog
title: 'World Models: Neo Vintage'
date: 2026-08-18
description: 'TODO.'
lang: en
translation_key: world-models-neo-vintage
translation_url: /blogs/2026/world-models-neo-vintage/zh
permalink: /blogs/2026/world-models-neo-vintage/
image_path: /blog-assets/2026-08-18-WorldModels/img/
category: notes
tags:
  - World Models
---

> Linda S. Gottfredson: "Intelligence is a very general mental capability that, among other things, involves the ability to reason, plan, solve problems, think abstractly, comprehend complex ideas, learn quickly and learn from experience. It is not merely book learning, a narrow academic skill, or test-taking smarts. Rather, it reflects a broader and deeper capability for comprehending our surroundings—"catching on," "making sense" of things, or "figuring out" what to do."

In the area of modern machine learning, the term "world model" is becoming an increasingly popular, yet increasingly vague, term. It's easy to get the intuition that a so-called world model is simply a model that can generate a realistic world, if we start from generative models nowadays. However, I think a more interesting and fundamental point about the world model is, can an intelligent agent think through its own internal world before actually taking action?

For example, if a person needs to walk from the table, which is on the left, to the door, his (or her) behavior would definitely not be: take a step to the left $\rightarrow$ bump into the table $\rightarrow$ receive a negative reward $\rightarrow$ update the policy. Instead, he (or she) will first roughly judge in his (or her) mind: the table is there, and I should be able to reach the door by going around to the right. Humans do this kind of simple planning every day. Before we actually take action, we have already "run" the future in a sense.

If machines were to acquire similar capabilities, they would likely need to internally build a model of how the world works, which is precisely the starting point of the research path of world models.

This blog is basically a review of three works on this topic:

- *[Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](http://incompleteideas.net/papers/sutton-90.pdf)* by Richard S. Sutton
- *[World Models](https://arxiv.org/abs/1803.10122)* by David Ha and Jurgen Schmidhuber
- *[A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)* by Yann LeCun

# Why do machines need to imagine?

Just like we've talked, an agent should deduce its best action based on its goal and some internal model capable of simulating how the world works. We call this kind of mechanism *planning*.

In the reinforcement learning literature, an agent decides what to do based on a certain policy, which is essentially a probability distribution of possible actions conditioned on the agent's current state. And this policy is learned via trial-and-error in a real world, i.e., the agent learns from experiences that each actually happens once at least. We call this kind of mechanism *learning*.

Sometimes, we are able to perform instinctive behaviours when we face danger, or act reflectively when consolidating a specific task in repitition, without the need to consciously plan out a course of action. We call this kind of mechanism *reacting*.

*Dyna* is a class of architectures integrating and permitting tradeoffs among these three approaches.

## Dyna-PI

*Dyna-PI* is based on approximating *policy iteration*, which we have fully discussed in the [temporal-difference learning blog](https://williamwuuu.github.io/blogs/2026/temporal-difference-learning/). Recall that *policy iteration* is one of the ways of finding an optimal policy through a sequence:

$$
\pi_0 
\xrightarrow{E} v_{\pi_0} 
\xrightarrow{I} \pi_1
\xrightarrow{E} v_{\pi_1}
\xrightarrow{I} \pi_2
\xrightarrow{E} \cdots
\xrightarrow{I} \pi_*
\xrightarrow{E} v_{\pi_*},
$$

where $\xrightarrow{E}$ denotes a *policy evaluation* and $\xrightarrow{I}$ denotes a *policy improvement*.

{% include widgets/blog_image.html src="Dyna-PI.png" caption="Picture 1: Overview of the proposed Dyna-PI architecture from the paper. The module in the lower left corner is like a single pole, double throw switch. With the 'WORLD' in place as shown we have reinforcement learning; with the 'WORLD MODEL' in place as shown we have planning." %}

The detailed algorithm of Dyna-PI:

$$
\begin{align*}
& \text{1. Decide if this will be a real experience or a hypothetical one;} \\
& \text{2. Pick a state } s \text{. If this is a real experience, use the current state;} \\
& \text{3. Choose an action: } a\sim\pi(\cdot\vert s)\text{;} \\
& \text{4. Perform action } a \text{; obtain next state } s^\prime \text{ and reward } r \text{ from world or world model;} \\
& \text{5. If this is a real experience, update world model } \widehat{\mathcal M} \text{ with } (s,a,s^\prime,r)\text{;} \\
& \text{6. Update evaluation function so that } e(s) \text{ is more like } r+\gamma e(s^\prime)\text{;} \\
& \text{7. Update policy - strengthen or weaken the tendency to perform action } a \text{ in state } s \\
& \quad\,\text{according to the error in the evaluation function: } r+\gamma e(s^\prime)-e(s)\text{;} \\
& \text{8. Go to Step 1.}
\end{align*}
$$

Now let's deduce how Dyna-PI originates from policy iteration. To avoid confusion, we will uniformly adopt a finite discount Markov decision process (MDP):

$$
\mathcal M=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma),\quad 0\leq \gamma<1.
$$

Recall that the Bellman expectation operator is defined as:

$$
(T_\pi V)(s) =
\sum_{a\in\mathcal A}
\pi(a|s)
\sum_{s^\prime\in\mathcal S} \sum_{r\in\mathcal{R}}
p(s^\prime,r\vert s,a)
\left[ r + \gamma V(s^\prime)\right].
$$

Like we've talked about, policy evaluation can be seen as repeatedly applying Bellman expectation operator to the estimated state-value until convergence. And policy improvement is simply choosing a greedy action with respect to estimated state-value for each state. *Policy improvement theorem* gaurantees that accurate evaluation and greedy improvement will eventually lead to an optimal policy.

However, this process imposes two strict requirements on online agents: 

- They must sum over all successor states; 
- They must know the complete environment model. 

Dyna-PI can be understood as the result of successively relaxing these three requirements.

Assume the state at current time $t$ is $S_t=s$. We denote the action sampled from the policy as $A_t\sim \pi(\cdot\vert s)$. By performing the action, we can get a feedback from the world as $(S_{t+1},R_{t+1}) \sim p(\cdot, \cdot \vert s,A_t)$. The TD-error in the evaluation function is then defined as:

$$
\delta_t = R_{t+1}+\gamma V(S_{t+1})-V(S_t).
$$

We can calculate the expectation of it conditioned on the current state:

$$
\begin{align*}
\mathbb{E}_\pi \left[\delta_t \vert S_t=s\right] 
& = \sum_{a}\pi(a\vert s)\sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left[r+\gamma V(s^\prime)-V(s)\right] \\
& = \sum_{a}\pi(a\vert s)\sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left[r+\gamma V(s^\prime)\right] - V(s) \\
& = (T_\pi V)(s) - V(s).
\end{align*}
$$

Thus the update

$$
V(S_t)\leftarrow V(S_t) + \beta \delta_t
$$

is a single-sample stochastic approximation of 

$$
V(S_t)\leftarrow (T_\pi V)(S_t).
$$

It does not require explicitly enumerating all actions and successor states, and only require one transition sample.



In particular if $V=v_\pi$, the expectation of TD-error conditioned on the current state and a certain action can be written as:

$$
\begin{align*}
\mathbb{E}\left[\delta_t\vert S_t=s,A_t=a\right]
& = \mathbb{E}\left[r+\gamma v_\pi(s^\prime)-v_\pi(s)\vert s,a\right] \\
& = \mathbb{E}\left[r+\gamma v_\pi(s^\prime)\vert s,a\right] - v_\pi(s) \\
& = q_\pi(s,a)-v_\pi(s) \\
& =: A_\pi(s,a),
\end{align*}
$$

where $A_\pi$ is called the advantage function, indicating how much better is the action $a$ chosen in state $s$ than the average performance of the current policy. Thus TD-error is also a single-sample approximation of the advantage function.

Assume the policy uses a Boltzmann distribution:

$$
\pi_w(a\vert s)=\frac{\exp(w(s,a))}{\sum_b\exp(w(s,b))},
$$

where $w(s,a)$ is a preference parameter for every state-action pair. We can form the policy update as:

$$
w(S_t,A_t)\leftarrow w(S_t,A_t)+\alpha\delta_t.
$$

If an action produces a better action-value than the current estimated state-value (i.e., $\delta>0$), the preference for that action will increase; if $\delta<0$, the preference will then decrease.

At this point, we have derived a model-free incremental algorithm from the precise policy iteration:

$$
\begin{cases}
\delta_t = R_{t+1}+\gamma V(S_{t+1})-V(S_t) & \leftarrow\text{TD-error}\\
V(S_t)\leftarrow V(S_t) + \beta \delta_t & \leftarrow\text{policy evaluation} \\
w(S_t,A_t)\leftarrow w(S_t,A_t)+\alpha\delta_t & \leftarrow\text{policy improvement}
\end{cases}
$$

So where is the world model? Reinforcement learning gets real experiences from the real world, and the world model is expected to provide experiences close to reality, thus reduce the cost spended on agent-world iteractions. 

Formally, we can update the world model $\widehat{\mathcal{M}}$ with real experiences, like $(s,a,s^\prime,r)$. Then we can use the model to generate a one-step hypothetical experience

$$
(\tilde{s}^\prime,\tilde{r}) \sim \widehat{\mathcal{M}}_t(\cdot,\cdot|s,a).
$$

## Dyna-Q

To be finished.

# Where should machines imagine?

To be finished.

# What should machines imagine?

To be finished.
