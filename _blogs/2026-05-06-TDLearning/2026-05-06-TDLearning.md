---
layout: blog
title: 'My Bizarre Adventure in RL: Temporal-Difference Learning'
date: 2026-05-06
description: 'A journey from Markov decision process to temporal-difference learning.'
lang: en
translation_key: temporal-difference-learning
translation_url: /blogs/2026/temporal-difference-learning/zh
permalink: /blogs/2026/temporal-difference-learning/
image_path: /blog-assets/2026-05-06-TDLearning/img/
category: notes
tags:
  - Reinforcement Learning
---

I think blogging about my first time digging into reinforcement learning (RL) theory would be fun.
So boom, here it goes!

This post begins with the classic *Markov decision process (MDP)* formulation, moves through *dynamic programming* and *Monte Carlo Methods*, and then uses those ideas to arrive at *temporal-difference learning*.

# Markov Decision Processes

## Problem Framing

The reinforcement learning (RL) problem is meant to be a straightforward framing of the problem of learning from interaction to achieve a goal, or more vividly, "**trial and error**".

The learner and decision-maker is called the *agent*. The thing it interacts with, comprising everything outside the agent, is called the *environment*. These two interact at each of a sequence of discrete time steps, $t=0,1,2,3,\dots$. At each time step $t$, the agent receives some representation of the environment's *state*, $S_t\in\mathcal{S}$, where $\mathcal{S}$ is the set of possible states, and on that basis selects an *action*, $A_t\in\mathcal{A}(S_t)$, where $\mathcal{A}(S_t)$ is the set of actions available in state $S_t$. One time step later, in part as a consequence of its action, the agent receives a numerical *reward*, $R_{t+1}\in\mathcal{R}\subset\mathbb{R}$, and finds itself in a new state, $S_{t+1}$.

{% include widgets/blog_image.html src="agent-env.png" caption="Picture 1: The agent–environment interaction in reinforcement learning." %}

Consider the situation when the system starts at a particular state $S_t\in\mathcal{S}$ and continuously taking actions after time step $t$, resulting in a trajectory like

$$
\tau=(S_t,A_t,R_{t+1},S_{t+1},A_{t+1},R_{t+2},S_{t+2},A_{t+2},R_{t+3},\dots).
$$

In general, we seek to maximize the *expected return*, where $G_t$ can be defined in the simplest case as the cumulative reward the agent receives after time step $t$:

$$
\begin{equation}
G_t=R_{t+1}+R_{t+2}+R_{t+3}+\cdots+R_T,
\end{equation}
$$

where $T$ is the final time step.

Since this expected return will be infinite when $T=\infty$, the learning process above would possibly fail. In order to prevent this from happening, we introduce a parameter called the *discount rate* $0\leq\gamma\leq 1$ to (1) as:

$$
\begin{equation}
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3}+\cdots = \sum\limits_{k=0}^\infty\gamma^k R_{t+k+1}.
\end{equation}
$$

If $\gamma<1$, the infinite sum has a finite value as long as the reward sequence $\{R_k\}$ is bounded. If $\gamma=0$, the agent is "myopic" in being concerned only with maximizing immediate rewards. As $\gamma$ approaches $1$, the objective takes future rewards into account more strongly: the agent becomes more farsighted.

## The Markov Property

In the RL framework, the agent makes its decisions as a function of a signal from the environment’s state. In broad sense, “the state” means whatever information that is available to the agent. And we expect the state satisfies *the Markov property*, which we defined as follow.

Consider the response given by a general environment at time ${t+1}$ to the action taken at time $t$, which may depend on everything that has happened earlier. In this case the dynamics can be defined as:

$$
\begin{equation}
\Pr\{S_{t+1}=s^\prime, R_{t+1}=r \vert S_0,A_0,R_1,\dots,S_{t-1},A_{t-1},R_t,S_t,A_t\},
\end{equation}
$$

for all possible $r$ and $s^\prime$. If the environment’s response at $t+1$ depends only on the state and action representations at $t$, i.e. 

$$
\begin{equation}
p(s^\prime, r \vert s,a)=\Pr\{S_{t+1}=s^\prime, R_{t+1}=r \vert S_t,A_t\},
\end{equation}
$$

for all $s^\prime,r,S_t,A_t$, we say the state signal has the Markov property and is a Markov state.

A reinforcement learning task that satisfies the Markov property is called a *Markov decision process (MDP)*, which models how a system changes state when different actions are applied. Formally, given any state $s$ and action $a$, the dynamics of an MDP can be specified by:

$$
\begin{equation}
p(s^\prime, r \vert s,a)=\Pr\{S_{t+1}=s^\prime, R_{t+1}=r \vert S_t=s,A_t=a\}.
\end{equation}
$$

## Value Functions

The value function of a state (or state-action pair) estimates how good or bad an individual is in a given state (or how good it is to perform a given action in a given state). And the notion of "how good" here is defined in terms of expected returns when starting in a specific state and following a specific behavior thereafter. Formally, we call this kind of behavior as a *policy*, $\pi(a\vert s)$, which is a conditional distribution over the actions $a\in\mathcal{A}$ given the state $s\in\mathcal{S}$.

For MDPs, we can define the *value* of a state $s$ under a policy $\pi$ as

$$
\begin{equation}
v_\pi(s)
= \mathbb{E}_\pi\left[G_t\vert S_t=s\right] 
= \mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\right\vert S_t=s\right],
\end{equation}
$$

where $\mathbb{E}_\pi$ denotes the expected value of a random variable given that the agent follows policy $\pi$, and $t$ is any time step. The function $v_\pi$ is called the *state-value function for policy $\pi$*.

Similarly, we define the value of taking action $a$ in state $s$ under a policy $\pi$ as 

$$
\begin{equation}
q_\pi(s,a)
=\mathbb{E}_\pi\left[G_t\vert S_t=s,A_t=a\right]
= \mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\right\vert S_t=s,A_t=a\right].
\end{equation}
$$

The function $q_\pi$ is called the *action-value function for policy $\pi$*.

A fundamental property of value functions is that they satisfy particular recursive relationships. The value function can be mathematically decomposed into 

$$
\begin{equation}
\begin{align*}
v_\pi(s) 
&= \mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\right\vert S_t=s\right] \\
&= \mathbb{E}_\pi\left[\left. R_{t+1}+\gamma\sum_{k=0}^{\infty}\gamma^k R_{t+k+2}\right\vert S_t=s\right] \\
&= \sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a) \left[ r + \gamma\mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+2}\right\vert S_{t+1}=s^\prime\right]\right] \\
&= \sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a) \left[r + \gamma v_\pi(s^\prime)\right], \\
\end{align*}
\end{equation}
$$

which is the foundation of dynamic programming upon which all RL algorithms are based. This is the *Bellman equation for $v_\pi$*, which expresses a relationship between the value of the state and the values of its successor state.

## Optimal Policy

Solving a RL task roughly means finding an *optimal policy* $\pi^*$ that maximize the expected return, which shares the same state-value function

$$
\begin{equation}
v^*(s)=\max_\pi v_\pi(s),\quad \forall s\in\mathcal{S}
\end{equation}
$$

and the same action value function

$$
\begin{equation}
q^*(s,a)=\max_\pi q_\pi(s,a),\quad \forall s\in\mathcal{S},a\in\mathcal{A}(s).
\end{equation}
$$

Intuitively, the value of a state under an optimal policy must equal the expected return for the best action from that state. Starting from this fact, we can derive the so-called *Bellman optimality equation* for $v^*$:

$$
\begin{equation}
\begin{align*}
v^*(s)
&= \max_{a\in\mathcal{A}(s)}q_{\pi_*}(s,a) \\
&= \max_{a\in\mathcal{A}(s)}\mathbb{E}_{\pi^*}\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\right\vert S_t=s,A_t=a\right] \\
&= \max_{a\in\mathcal{A}(s)}\mathbb{E}_{\pi^*}\left[R_{t+1}+\left.\gamma\sum_{k=0}^{\infty}\gamma^k R_{t+k+2}\right\vert S_t=s,A_t=a\right] \\
&= \max_{a\in\mathcal{A}(s)}\mathbb{E}_{\pi^*}\left[R_{t+1}+\left.\gamma v^*(S_{t+1})\right\vert S_t=s,A_t=a\right] \\
&= \max_{a\in\mathcal{A}(s)}\sum_{s^\prime,r} p(s^\prime,r\vert s,a)[r+\gamma v^*(s^\prime)].
\end{align*}
\end{equation}
$$

Similarly, we can also derive the Bellman optimality equation for $q^*$ is

$$
\begin{equation}
\begin{align*}
q^*(s,a)
&= \mathbb{E}_{\pi^*}\left[\left.R_{t+1}+\gamma\max_{a^\prime} q^*(S_{t+1}, a^\prime)\right\vert S_t=s,A_t=a\right] \\
&= \sum_{s^\prime,r}p(s^\prime,r\vert s, a)\left[r+\gamma\max_{a^\prime}q^*(s^\prime,a^\prime)\right].
\end{align*}
\end{equation}
$$

The Bellman optimality equation is actually a system of equations with $N$ equations and $N$ unknowns. By solving this system of nonlinear equations, we can get $v^*$ and $q^*$, which determine an optimal policy.

# Dynamic Programming

The key idea of RL generally, is the use of value functions to organize and structure the search for good policies. *Dynamic programming (DP)* refers to a collection of algorithms that can be used to compute the value functions defined earlier, given a perfect model of the environment as an MDP.

## Policy Iteration

*Policy iteration* is one of the ways of finding an optimal policy, which mainly consists of two components, evaluation and improvement. Basically, we hope to achieve that through a sequence:

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

### Policy Evaluation

Consider how to compute the state-value function $v_\pi$ for an arbitrary policy $\pi$, which is commonly called *policy evaluation*. Recall that the Bellman equation for $v_\pi$ is formed as

$$
\begin{equation}
\begin{align*}
& v_\pi = \sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a) \left[r + \gamma v_\pi(s^\prime)\right] \\
\Rightarrow \quad & v_\pi-\gamma\sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a)\left[v_\pi(s^\prime)\right] = \underbrace{\sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a)\left[r\right]}_{=:r_\pi}
\end{align*}
\end{equation}
$$

Since both the policy term $\pi(a\vert s)$ and the environment’s dynamics term $p(s^\prime,r\vert s, a)$ are completely known, this form is actually a system of $\vert\mathcal{S}\vert$ linear equations in $\vert\mathcal{S}\vert$ unknowns ($v_\pi(s),s\in\mathcal{S}$). If we arrange state values, rewards and transition probabilities ​​into matrices:

$$
\begin{equation}
\mathbf{v}_\pi=
\left[
\begin{matrix}
v_\pi(s_1) \\
v_\pi(s_2) \\
\cdots \\
v_\pi(s_{\vert\mathcal{S}\vert})
\end{matrix}
\right],\,
\mathbf{r}_\pi=
\left[
\begin{matrix}
r_\pi(s_1) \\
r_\pi(s_2) \\
\cdots \\
r_\pi(s_{\vert\mathcal{S}\vert})
\end{matrix}
\right],\,
P_\pi=
\left[
\begin{matrix}
P(s_1,s_1) & P(s_1,s_2) & \cdots & P(s_1,s_{\vert\mathcal{S}\vert}) \\
P(s_2,s_1) & P(s_2,s_2) & \cdots & P(s_2,s_{\vert\mathcal{S}\vert}) \\
\vdots & \vdots & \ddots & \vdots \\
P(s_{\vert\mathcal{S}\vert},s_1) & P(s_{\vert\mathcal{S}\vert},s_2) & \cdots & P(s_{\vert\mathcal{S}\vert},s_{\vert\mathcal{S}\vert})
\end{matrix}
\right]
\end{equation}
$$

where

$$
\begin{equation}
[P_\pi]_{ij} = P(s_i,s_j) := \sum_{a\in\mathcal{A}(s_i)}\pi(a\vert s_i)\sum_{r\in\mathcal{R}} p(s_j,r\vert s_i,a),
\end{equation}
$$

then the Bellman equation can be written into the matrix form:

$$
\begin{equation}
(I-\gamma P_\pi)\mathbf{v}_\pi=\mathbf{r}_\pi.
\end{equation}
$$

Thus we find the solution as:

$$
\begin{equation}
\mathbf{v}_\pi=(I-\gamma P_\pi)^{-1}\mathbf{r}_\pi.
\end{equation}
$$

Then methods like Gaussian elimination are applied in the linear algebra literature. But RL typically deals with a vast state space ($\vert\mathcal{S}\vert$ is very large) and does not necessarily require an exact solution. 

Let's take a deeper look inside. Define the *Bellman expectation operator* $T_\pi$ as

$$
\begin{equation}
(T_\pi v)(s)=r_\pi(s)+\gamma\sum_{s^\prime}P(s,s^\prime)v(s^\prime),
\end{equation}
$$

where $v:\mathcal{S}\to\mathbb{R}$ is a random value function. If we let $v=v_\pi$, then we have

$$
\begin{equation}
\begin{align*}
(T_\pi v_\pi)(s)
&= r_\pi(s)+\gamma\sum_{s^\prime}P(s,s^\prime)v_\pi(s^\prime) \\
&= \sum_{a} \pi(a\vert s)\sum_{s^\prime}\sum_{r} p(s^\prime,r\vert s, a)\left[r\right] + \gamma\sum_{s^\prime}\left[\left(\sum_{a}\pi(a\vert s)\sum_{r} p(s,r\vert s,a)\right) v_\pi(s^\prime)\right] \\
&= \sum_{a} \pi(a\vert s)\sum_{s^\prime}\sum_{r} p(s^\prime,r\vert s, a)\left[r\right] + \sum_{a} \pi(a\vert s)\sum_{s^\prime}\sum_{r} p(s^\prime,r\vert s, a)\left[v_\pi(s^\prime)\right] \\
&= \sum_{a} \pi(a\vert s)\sum_{s^\prime}\sum_{r} p(s^\prime,r\vert s, a)\left[r+v_\pi(s^\prime)\right] \\
&= v_\pi(s).
\end{align*}
\end{equation}
$$

Obviously, $v_\pi$ is exactly the fixed-point of $T_\pi$. So for our purposes, methods like [fixed-point iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration) are most suitable, since the Bellman equation in matrix form is hoped to converge to the fixed-point $\mathbf{v}_\pi$.

The detailed iteration algorithm is described as follow:

$$
\begin{align*}
&\text{Input }\pi\text{, the policy to be evaluated} \\
&\text{Initialize an array }V(s)=0\text{, for all }s\in\mathcal{S} \\
&\text{Repeat} \\
&\quad \Delta\leftarrow 0 \\
&\quad \text{For each }s\in\mathcal{S} \\
&\quad \quad v\leftarrow V(s) \\
&\quad \quad V(s)\leftarrow \sum_a \pi(a\vert s)\sum_{s^\prime,r} p(s^\prime,r\vert s, a) \left[r + \gamma V(s^\prime)\right] \\
&\quad \quad \Delta\leftarrow\max(\Delta,\vert v-V(s)\vert) \\
&\text{until }\Delta<\theta\text{ (a small positive number)} \\
&\text{Output }V\approx v_\pi
\end{align*}
$$

### Policy Improvement

Intuitively, we hope the policy gets better and better through the "trial and error" process. Since we are able to compute $v_\pi$ for a policy $\pi$ through policy evaluation, we are able to tell a new policy $\pi^\prime$ is better if

$$
\begin{equation}
v_{\pi^\prime}(s)\geq v_\pi(s),\quad \forall s\in\mathcal{S}
\end{equation}
$$

which is to say, starting from any state, the expected return of executing $\pi^\prime$ is no less than that of executing $\pi$. This means a complete iteration algorithm needs to be performed again. The value of the *policy improvement theorem* lies in the fact that we only need to know $v_\pi$ or $q_\pi$ of the old policy to determine whether a certain modification guarantees a better policy. 

**Policy improvement theorem.** Let $\pi$ and $\pi^\prime$ be any pair of policies such that 

$$
\begin{equation}
q_\pi(s,\pi^\prime(s))\geq v_\pi(s),\quad \forall s\in\mathcal{S},
\end{equation}
$$

then the policy $\pi^\prime$ must be as good as, or better than, $\pi$. The proof goes as follow:

$$
\begin{equation}
\begin{align*}
v_\pi(s) 
& \leq q_\pi(s,\pi^\prime(s)) \\
& = \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma v_\pi(S_{t+1})\vert S_t=s\right] \\
& \leq \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma q_\pi(S_{t+1},\pi^\prime(S_{t+1}))\vert S_t=s\right] \\
& = \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma\mathbb{E}_{\pi^\prime}\left[R_{t+2}+\gamma v_\pi(S_{t+2})\right] \vert S_t=s\right] \\
& = \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma R_{t+2}+\gamma^2 v_\pi(S_{t+2}) \vert S_t=s\right] \\
& \leq \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\gamma^3 v_\pi(S_{t+3}) \vert S_t=s\right] \\
& \dots \\
& \leq \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots\vert S_t=s\right] \\
& = v_{\pi^\prime}(s).
\end{align*}
\end{equation}
$$

Within this theorem, we can easily construct a greedy policy $\pi^\prime$ by selecting at each state the action that appears the best:

$$
\begin{equation}
\pi^\prime(s)=\underset{a\in\mathcal{A}}{\operatorname{argmax}}q_\pi(s,a),\quad \forall s\in\mathcal{S}.
\end{equation}
$$

In particular, if the new greedy policy $\pi^\prime$ is as good as the old policy $\pi$, then we have

$$
\begin{equation}
\begin{align*}
v_{\pi}=v_{\pi^\prime} 
& = \max_{a\in\mathcal{A}} q_\pi(s,a) \\
& = \max_{a\in\mathcal{A}} \mathbb{E}\left[R_{t+1}+\gamma v_\pi(S_{t+1})\vert S_t=s,A_t=a\right] \\
& = \max_{a\in\mathcal{A}} \sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left[r+\gamma v_\pi(s^\prime)\right],
\end{align*}
\end{equation}
$$

which is exactly the same as the Bellman optimality equation. This indicates that we've already find the optimal policy.

# Monte Carlo Methods

*Monte Carlo methods* are ways of solving the reinforcement learning problem based on averaging sample returns. Unlike DP, where we assume complete knowledge of the environment, Monte Carlo methods require only *experience* (sample sequences of states, actions, and rewards from actual or simulated interaction with an environment).

## Monte Carlo Prediction

Let's consider the policy evaluation problem again. First we define an *episode* as one complete journey or trial where an agent interacts with its environment, starting from an initial state and ending at a terminal state. And we call each occurrence of a certain state s in an episode a *visit* to s.

Assume we already have a set of $n$ episodes obtained by following $\pi$ and passing through $s$. To estimate $v_\pi(s)$, the value of a certain state $s$ under a given policy $\pi$, we can simply average the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value. In particular, we introduce *first-visit MC method*, which estimates $v_\pi(s)$ as the average of the returns following first visits to $s$. Here is the formal algorithm:

$$
\begin{align*}
& \text{Initialize:} \\
& \quad \pi\leftarrow\text{policy to be evaluated} \\
& \quad V\leftarrow\text{an arbitrary state-value function} \\
& \quad \mathrm{Returns}(s)\leftarrow\text{an empty list, for all } s\in\mathcal{S} \\
& \text{Repeat:} \\
& \quad \text{Generate an episode using: }\pi \\
& \quad \text{For each state }s\text{ in the state space }\mathcal{S}: \\
& \quad \quad G\leftarrow \text{return following the first occurrence of }s\text{ in the episode} \\
& \quad \quad \text{Append }G\text{ to }\mathrm{Returns}(s) \\
& \quad \quad V(s)\leftarrow\mathrm{average}(\mathrm{Returns}(s))
\end{align*}
$$

<details class="proof" markdown="1" open>
<summary>Proof of the almost-sure convergence of first-visit MC method.</summary>

Let $i$-th episode be

$$
S_0^{(i)},A_0^{(i)},R_1^{(i)},S_1^{(i)},A_1^{(i)},R_2^{(i)},\cdots,S_{T_i}^{(i)},
$$

where $T_i$ is the time when the episode reaches the terminal state. Define the first-visit time of state $s$ in this episode:

$$
\begin{equation}
\tau_s^{(i)}=
\begin{cases}
\inf\{t\geq 0:S_t^{(i)}=s\} & \text{if visited}\\
\infty & \text{otherwise}\\
\end{cases}
\end{equation}
$$

If $\tau_s^{(i)}<T$, the accumulated return following the first-visit is formed as:

$$
\begin{equation}
G_{\tau_s^{(i)}}^{(i)}=R_{\tau_s^{(i)}+1}^{(i)}+\gamma R_{\tau_s^{(i)}+2}^{(i)}+\gamma^2 R_{\tau_s^{(i)}+3}^{(i)}+\cdots\gamma^{T-\tau_s^{(i)}-1}R_T^{(i)}.
\end{equation}
$$

And our goal is to prove that, for all the episodes containing $s$, the sample average of $G_{\tau_s}$ converges to $v_\pi(s)$.

By the definition of $\tau_s$, we can see that $\tau_s$ is a [stopping time](https://en.wikipedia.org/wiki/Stopping_time) in the random process literature. Because a process under a fixed policy satisfies the *strong Markov property*, once state $s$ is reached at stopping time $\tau_s$, the conditional distribution of the future trajectory is the same as that of a new process that "starts directly from state $s$ and follows $\pi$". Define the *filtration* $\mathcal{F}_{\tau_s}$ to be all the information we have up to time $\tau_s$. Thus we have

$$
\begin{equation}
\mathbb{E}_\pi\left[G_{\tau_s}\vert\mathcal{F}_{\tau_s}\right]=v_\pi(s).
\end{equation}
$$

For the $i$-th episode, we define an variable $I^{(i)}$ to indicate whether state $s$ is visited in it, i.e. 

$$
\begin{equation}
I^{(i)}=
\begin{cases}
1 & s\text{ is visited} \\
0 & \text{otherwise}
\end{cases}
\end{equation}
$$

Thus the $V(s)$ we get from the first-visit MC estimation can be written as:

$$
\begin{equation}
V(s) 
= \frac{\sum\limits_{i=1}^n I^{(i)} G_{\tau_s^{(i)}}^{(i)}}{\sum\limits_{i=1}^n I^{(i)}} 
= \frac{\mathbb{E}\left[ I^{(i)} G_{\tau_s^{(i)}}^{(i)}\right]}{\mathbb{E}\left[ I^{(i)}\right]}.
\end{equation}
$$

Based on the [law of total expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation), we have

$$
\begin{equation}
V(s) 
= \frac{\mathbb{E}\left[ I^{(i)} \mathbb{E}\left[G_{\tau_s^{(i)}}^{(i)}\vert\mathcal{F}_{s^{(i)}}^{(i)}\right]\right]}{\mathbb{E}\left[ I^{(i)}\right]}
= \frac{\mathbb{E}\left[ I^{(i)} v_\pi(s)\right]}{\mathbb{E}\left[ I^{(i)}\right]}
= \frac{\mathbb{E}\left[ I^{(i)}\right] v_\pi(s)}{\mathbb{E}\left[ I^{(i)}\right]}.
\end{equation}
$$

Since $I^{(i)}$ are i.i.d Bernoulli random variables, as $n\to\infty$, based on the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers), $\mathbb{E}\left[I^{(i)}\right]=\sum_{i=1}^n I^{(i)}$ converges to some positive number. Thus the estimator $V(s)$ converges to $v_\pi(s)$ obviously.

</details>

The estimation of action values $q_\pi$ is quite similar to state values $v_\pi$, which is particularly useful when the model is unavailable. However, if $\pi$ is a deterministic policy, when using first-visit method, some state-action pairs may never be visited. What's expected is all the actions from each state, so that we can choose among the actions available in each state. A general approach to assuring that all state–action pairs are encountered is to consider only stochastic policies with a nonzero probability of selecting all actions in each state.

## Monte Carlo Control
Monte Carlo estimation can be used in control, that is, approximating optimal policies. We use the policy iteration scaffold mentioned in the DP section, but replace the objective of policy evaluation from state values to action values, i.e.

$$
\pi_0 
\xrightarrow{E} q_{\pi_0} 
\xrightarrow{I} \pi_1
\xrightarrow{E} q_{\pi_1}
\xrightarrow{I} \pi_2
\xrightarrow{E} \cdots
\xrightarrow{I} \pi_*
\xrightarrow{E} q_{\pi_*}.
$$

Policy improvement is done by making the policy greedy with respect to the action-value function, i.e.

$$
\begin{equation}
q_{\pi_k}(s,\pi_{k+1}(s))=q_{\pi_k}(s,\underset{a\in\mathcal{A}}{\operatorname{argmax}} q_{\pi_k}(s,a))=\max_{a\in\mathcal{A}} q_{\pi_k}(s,a),
\end{equation}
$$

therefore no model is needed to construct the greedy policy. 

# Temporal-Difference Learning

TD learning is a combination of Monte Carlo ideas and DP ideas. Like MC methods, TD methods can learn directly from raw experience without a model of the environment’s dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).

## TD Prediction

For state-value estimation, a simple every-visit MC method suitable for non-stationary environments is

$$
\begin{equation}
V(S_t)\leftarrow V(S_t)+\alpha\left[G_t-V(S_t)\right],
\end{equation}
$$

which waits until the actual return $G_t$ following the visit is known. $\alpha$ is a constant step size parameter. TD methods, instead, wait only until the next time step. The simplest TD method, known as TD($0$), is

$$
\begin{equation}
V(S_t)\leftarrow V(S_t)+\alpha\left[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\right],
\end{equation}
$$

where we replace the target for update from $R_{t+1}$ to $R_{t+1}+\gamma V(S_{t+1})$. We call this kind of methods which is based on an existing estimate, a *bootstrapping* method.

Recall that in the MDP section, we derive the recursive relationships the value functions satisfies: 

$$
\begin{equation}
\begin{align*}
v_\pi(s) 
&= \mathbb{E}_\pi\left[\left.G_t\right\vert S_t=s\right] \\
&= \mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\right\vert S_t=s\right] \\
&= \mathbb{E}_\pi\left[\left. R_{t+1}+\gamma\sum_{k=0}^{\infty}\gamma^k R_{t+k+2}\right\vert S_t=s\right] \\
&= \sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a) \left[ r + \gamma\mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+2}\right\vert S_{t+1}=s^\prime\right]\right] \\
&= \sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a) \left[r + \gamma v_\pi(s^\prime)\right]. \\
\end{align*}
\end{equation}
$$

Roughly speaking, MC methods use an estimate of the first line as the target, whereas DP methods use an estimate of last line as the target.

The detailed algorithm can be written as follow:

$$
\begin{align*}
& \text{Input: the policy }\pi\text{ to be evaluated} \\
& \text{Initialize }\text{ arbitrarily (e.g., }V(s) = 0, \forall s\in\mathcal{S}\text{)} \\
& \text{Repeat (for each episode):} \\
& \quad \text{Initialize }S \\
& \quad \text{Repeat (for each step of episode):} \\
& \quad \quad A\leftarrow\text{action given by }\pi\text{ for }S \\
& \quad \quad \text{Take action }A \text{; observe reward }R \text{, and next state }S^\prime \\
& \quad \quad V(S) \leftarrow V(S)+\alpha\left[R + \gamma V (S^\prime) − V(S)\right] \\
& \quad \quad S \leftarrow S^\prime \\
& \quad \text{until }S\text{ is terminal}
\end{align*}
$$

The almost sure convergence to $v_\pi(s)$ should be conditioned on $\alpha$ changing with the number of state $s$ is visited. The step size used for the $n$-th access to state $s$ is denoted as $\alpha_n(s)$, s.t.

$$
\begin{equation}
0<\alpha_n(s)\leq 1,\quad
\sum_{n=1}^\infty \alpha_n(s)=\infty,\quad
\sum_{n=1}^\infty\alpha_n^2(s)<\infty.
\end{equation}
$$

The proof of the convergence relies on stochastic approximation theory and contraction mapping properties, which is too long for this blog and therefore omitted.

## Q-Learning: Off-Policy TD Control

The simplest form of *Q-learning*, *one-step Q-learning*, is defined by

$$
\begin{equation}
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\left[R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)\right].
\end{equation}
$$

In this case, the learned action-value function $Q$, directly approximates the optimal action-value function $q^*$.

Let's imagine that a robot takes actions sampled from some policy $\pi_e(a\vert s)$, collecting a dataset of $n$ episodes of $T$ time-steps each

$$
\{
s^{(i)}_0, a^{(i)}_0, 
s^{(i)}_1, a^{(i)}_1, 
\dots,
s^{(i)}_{T-1}, a^{(i)}_{T-1}, 
s^{(i)}_T
\},\quad i=1, 2, \dots, n
$$

We consider a optimization problem as

$$
\begin{equation}
\hat{Q}=\min\ell(Q)
\end{equation}
$$

where we denote $\ell(Q)$ as

$$
\begin{equation}
\ell(Q):=\frac{1}{nT}\sum_{i=1}^{n}\sum_{t=0}^{T-1}\left(\underbrace{
Q(s^{(i)}_t,a^{(i)}_t)-\left(
r(s_t^{(i)},a_t^{(i)})+\gamma\max_{a^\prime}Q(s^{(i)}_{t+1},a^\prime)
\right)}_{\text{Bellman Error}}\right)^2.
\end{equation}
$$

This optimization problem would be identical to the Value iteration if satisfying two ideal conditions:
- The taken policy $\pi_e$ is equal to the optimal policy $\pi^*$;
- An infinite amount of data is collected.

We can minimize the objective using gradient descent. For every pair $(s_t^i,a_t^i)$ in our dataset, we can write

$$
\begin{equation}
\begin{align*}
Q(s_t^{(i)},a_t^{(i)})
&\leftarrow Q(s_t^{(i)},a_t^{(i)})-\eta\nabla_{Q(s_t^{(i)},a_t^{(i)})}\ell(Q) \\
&= (1-\eta)Q(s_t^{(i)},a_t^{(i)})+\eta\left(r(s_t^{(i)},a_t^i)+\gamma\max_{a^\prime}Q(s^{(i)}_{t+1},a^\prime)\right)
\end{align*}
\end{equation}
$$

where $\eta$ is the learning rate.

Given the solution of these updates $\hat{Q}$, which is an approximation of the optimal value function $Q^*$, we can obtain the optimal policy corresponding to this value function easily using

$$
\begin{equation}
\hat{\pi}(s)=\arg\max_{a}\hat{Q}(s,a).
\end{equation}
$$

### Exploration

If the policy $\pi_e$ does not reach diverse parts of the state-action space, then it is easy to imagine our estimate $\hat{Q}$ will be a poor approximation of the optimal $Q^*$. It is also important to note that in such a situation, the estimate of $\hat{Q}$ at all states $s\in\mathcal{S}$ will be bad, not just the ones visited by $\pi_e$. We can mitigate this concern by picking a completely random $\pi_e$ that samples actions uniformly randomly from $\mathcal{A}$. Such a policy would visit all states, but it will take a large number of trajectories before it does so. 

Typically implementations of Q-Learning tie together the current estimate of $Q$ and the exploration policy $\pi_e$ to set

$$
\begin{equation}
\pi_e(a\vert s)=
\begin{cases}
\arg\max_{a^\prime}\hat{Q}(s,a^\prime) & \text{with prob. }1-\epsilon \\
\mathrm{uniform}(\mathcal{A}) & \text{with prob. }\epsilon
\end{cases}
\end{equation}
$$

where $\epsilon$ is called the exploration parameter. This particular $\pi_e$ is called an **$\epsilon$-greedy exploration policy**, which chooses the optimal action with $1-\epsilon$ but explores randomly $\epsilon$. We can also use the softmax exploration policy

$$
\begin{equation}
\pi_e(a\vert s)=\frac{e^{\hat{Q}(s,a)/T}}{\sum_{a^\prime}e^{\hat{Q}(s,a)/T}}
\end{equation}
$$

where the hyper-parameter $T$ is called temperature. A large value of $\epsilon$ in $\epsilon$-greedy policy functions similarly to large value of temperature $T$ for the softmax policy.

# References

[1] Richard S. Sutton, Andrew G. Barto. (2014). Reinforcement Learning: An Introduction. The MIT Press.

[2] Watkins, C. J., Dayan, P. (1992). Technical Note: Q-learning. Machine learning, 8(3-4), 279-292.

[3] Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola. (2023). Dive into Deep Learning. Cambridge University Press.
