---
layout: blog
title: 'My Bizarre Adventure in RL: Q-Learning'
date: 2026-05-06
permalink: /posts/2026/5/Q-Learning/
tags:
  - Reinforcement Learning
---

I thought recording my first time digging into the theory behind reinforcement learning (RL) would be fun.
So boom, there you go! (^_^)

This post begins with the classic **Markov decision process (MDP)** formulation, moves through **value iteration**, and then uses those ideas to arrive at **Q-learning**.

The central question is simple: if an agent can take actions in an environment and receive rewards, how can it learn to choose actions that lead to large long-term reward? The answer starts with making the environment precise.

## Markov Decision Processes

A Markov decision process models how a system changes state when different actions are applied. In its basic form, an MDP is defined by the tuple

$$
(\mathcal{S},\mathcal{A},T,r),
$$

where:

- $\mathcal{S}$ is the state space.
- $\mathcal{A}$ is the action space.
- $T:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\to[0,1]$ is the transition function. It gives the probability of reaching a next state $s^\prime\in\mathcal{S}$ after taking action $a\in\mathcal{A}$ in state $s\in\mathcal{S}$:

$$
T(s,a,s^\prime)=P(s^\prime\vert s,a).
$$

- $r:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ is the reward function.

Starting from an initial state $s_0\in\mathcal{S}$, the agent repeatedly takes actions and observes rewards. This produces a trajectory

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,s_2,a_2,r_2,\dots).
$$

If we simply add all rewards along the trajectory, the return is

$$
R(\tau)=r_0+r_1+r_2+\dots.
\tag{1}
$$

At an intuitive level, reinforcement learning is about finding behavior that produces trajectories with large total reward. However, for an infinite trajectory, the sum in (1) may diverge. To make the objective well-defined, we introduce a discount factor $\gamma<1$:

$$
R(\tau)=r_0 + \gamma r_1 + \gamma^2 r_2+\dots.
\tag{2}
$$

Discounting makes rewards in the near future more important than rewards far away. It also encourages the agent to reach rewarding states in fewer steps, rather than treating all future rewards equally.

## Value Iteration

To decide which action is good, the agent needs a rule for choosing actions. A stochastic policy, denoted by $\pi(a\vert s)$, is a conditional distribution over actions $a\in\mathcal{A}$ given a state $s\in\mathcal{S}$.

Given a policy $\pi$, we define the value function as the expected discounted return when starting from state $s_0$ and then following $\pi$:

$$
V^\pi(s_0)
= E_{a_t\sim\pi(a \vert s_t)}\left[R(\tau)\right]
= E_{a_t\sim\pi(a \vert s_t)}\left[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)\right].
\tag{3}
$$

Here $a_t\sim\pi(a\vert s_t)$ means the action is sampled from the policy at time $t$, and the next state is sampled according to the transition probability $s_{t+1}\sim P(s_{t+1}\vert s_t,a_t)$. In other words, $V^\pi(s_0)$ measures the average discounted reward obtained by all possible trajectories that begin at $s_0$ and follow $\pi$.

The useful part is that this long-term quantity can be decomposed into an immediate reward plus the value of the next state:

$$
V^\pi(s_0) =
E_{a_0\sim\pi(a \vert s_0)}\left[
\underbrace{r(s_0, a_0)}_{\text{immediate reward}} +
\underbrace{\gamma E_{s_1\sim P(s_1\vert s_0,a_0)}\left[V^\pi(s_1)\right]}_{\text{future expected reward}}
\right].
\tag{4}
$$

This recursive relationship is the foundation of dynamic programming in reinforcement learning. It is the **Bellman equation**. Written explicitly using the transition probabilities of the MDP, it becomes

$$
V^\pi(s)=\sum_{a\in\mathcal{A}}\pi(a\vert s)\left[r(s,a)+\gamma\sum_{s^\prime\in\mathcal{S}}P(s^\prime\vert s,a)V^\pi(s^\prime)\right]
\tag{5}
$$

for all $s\in\mathcal{S}$.

The goal is to find an optimal policy, which maximizes the expected return:

$$
\begin{align}
\pi^*
&= \arg\max_\pi V^\pi(s_0) \notag\\
&= \arg\max_\pi E_{a_0\sim\pi(a \vert s_0)}
\left[
r(s_0, a_0) +
\gamma E_{s_1\sim P(s_1\vert s_0,a_0)}\left[V^\pi(s_1)\right]
\right].
\end{align}
\tag{6}
$$

Once the optimal value function $V^*$ is known, the corresponding greedy policy can be recovered by selecting the action with the largest one-step lookahead value:

$$
\pi^*(s)=
\arg\max_{a\in\mathcal{A}}\left[
r(s,a)+\gamma\sum_{s^\prime\in\mathcal{S}}P(s^\prime\vert s,a)V^*(s^\prime)
\right].
\tag{7}
$$

This turns the search for an optimal policy into the search for an optimal value function. **Value iteration** does this by initializing $V_0(s)$ to arbitrary values for all states and repeatedly applying the Bellman optimality update:

$$
V_{k+1}(s)=\max_{a\in\mathcal{A}}\left\{
r(s,a)+\gamma\sum_{s^\prime\in\mathcal{S}}P(s^\prime\vert s,a)V_k(s^\prime)
\right\}.
\tag{8}
$$

As the iteration continues, the value function converges to the optimal value function:

$$
V^*(s)=\lim_{k\to\infty}V_k(s)
\tag{9}
$$

for all states $s\in\mathcal{S}$.

## Q-Learning

### Action Values

Value iteration works with values of states. In practice, it is often more useful to work with values of state-action pairs. This quantity is called the **action-value function**, or the **Q-function**:

$$
Q^\pi(s_0,a_0)=r(s_0,a_0)+E\left[
\sum_{t=1}^\infty\gamma^t r(s_t,a_t)
\right].
\tag{10}
$$

The interpretation is direct: $Q^\pi(s_0,a_0)$ is the expected discounted return after taking action $a_0$ in state $s_0$, and then following policy $\pi$ afterward.

Like the value function, the Q-function also has a Bellman form. Using the transition probabilities, we can write

$$
Q^\pi(s,a)=r(s,a)+
\gamma\sum_{s^\prime\in\mathcal{S}}P(s^\prime\vert s,a)\sum_{a^\prime\in\mathcal{A}}\pi(a^\prime\vert s^\prime)Q^\pi(s^\prime,a^\prime)
\tag{11}
$$

for all $s\in\mathcal{S}$ and $a\in\mathcal{A}$.

The optimal version replaces the expectation over the next action with a maximum over actions:

$$
Q_{k+1}(s,a)=r(s,a)+\gamma\sum_{s^\prime\in\mathcal{S}}P(s^\prime\vert s,a)\max_{a^\prime\in\mathcal{A}}Q_k(s^\prime,a^\prime).
\tag{12}
$$

This is the value-iteration idea written in terms of action values. It also points directly toward Q-learning: instead of requiring full knowledge of the transition probabilities, Q-learning estimates these action values from sampled experience.

### Learning From Data

Suppose a robot takes actions sampled from an exploration policy $\pi_e(a\vert s)$ and collects a dataset of $n$ trajectories, each with $T$ time steps:

$$
\{(s^i_0, a^i_0), (s^i_1, a^i_1),(s^i_2, a^i_2),\dots,(s^i_{T-1}, a^i_{T-1})\},\quad i=1, 2, \dots, n.
$$

We can estimate a Q-function by minimizing the Bellman error:

$$
\hat{Q}=\arg\min_Q\ell(Q),
\tag{13}
$$

where

$$
\ell(Q):=\frac{1}{nT}\sum_{i=1}^{n}\sum_{t=0}^{T-1}\left(\underbrace{
Q(s^i_t,a^i_t)-\left(
r(s_t^i,a_t^i)+\gamma\max_{a^\prime}Q(s^i_{t+1},a^\prime)
\right)}_{\text{Bellman error}}\right)^2.
\tag{14}
$$

This objective tries to make the current estimate $Q(s^i_t,a^i_t)$ match a one-step target: the observed immediate reward plus the discounted best estimated value at the next state.

The optimization problem becomes identical to value iteration under two idealized conditions:

- The data-collecting policy $\pi_e$ is equal to the optimal policy $\pi^*$.
- An infinite amount of data is collected.

In practice, we use sampled updates. For every pair $(s_t^i,a_t^i)$ in the dataset, gradient descent on the Bellman error gives an update of the form

$$
\begin{align}
Q(s_t^i,a_t^i)
&\leftarrow Q(s_t^i,a_t^i)-\eta\nabla_{Q(s_t^i,a_t^i)}\ell(Q) \tag{15} \\
&= (1-\eta)Q(s_t^i,a_t^i)+\eta\left(r(s_t^i,a_t^i)+\gamma\max_{a^\prime}Q(s^i_{t+1},a^\prime)\right),
\end{align}
$$

where $\eta$ is the learning rate. After obtaining an estimate $\hat{Q}$, which approximates the optimal action-value function $Q^*$, we can extract a deterministic greedy policy:

$$
\hat{\pi}(s)=\arg\max_{a}\hat{Q}(s,a).
$$

### Exploration

The quality of $\hat{Q}$ depends heavily on the data used to estimate it. If the exploration policy $\pi_e$ does not visit diverse parts of the state-action space, then $\hat{Q}$ can become a poor approximation of $Q^*$. This problem is not limited to unvisited states. Because Bellman updates propagate information through neighboring states, poor estimates in one region can affect other parts of the value function as well.

One simple approach is to choose a completely random exploration policy that samples actions uniformly from $\mathcal{A}$. Such a policy can eventually visit all states, but it may require a very large number of trajectories.

A more common strategy is to connect exploration to the current estimate of $Q$. One standard choice is the **$\epsilon$-greedy exploration policy**:

$$
\pi_e(a\vert s)=
\begin{cases}
\arg\max_{a^\prime}\hat{Q}(s,a^\prime) & \text{with probability }1-\epsilon, \\
\mathrm{uniform}(\mathcal{A}) & \text{with probability }\epsilon.
\end{cases}
$$

With probability $1-\epsilon$, the agent chooses the currently best action. With probability $\epsilon$, it explores randomly.

Another common choice is the softmax exploration policy:

$$
\pi_e(a\vert s)=\frac{e^{\hat{Q}(s,a)/T}}{\sum_{a^\prime}e^{\hat{Q}(s,a^\prime)/T}},
$$

where the hyperparameter $T$ is called the temperature. A larger $\epsilon$ in $\epsilon$-greedy exploration plays a similar role to a larger temperature $T$ in the softmax policy: both make the agent explore more.

Q-learning therefore combines two ideas. The Bellman equation gives a recursive target for long-term reward, while exploration determines whether the agent collects enough varied experience for that target to become meaningful.

## References

[1] Watkins, C. J., Dayan, P. (1992). Technical Note: Q-learning. Machine learning, 8(3-4), 279-292.

[2] Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola. (2023). Dive into Deep Learning. Cambridge University Press.
