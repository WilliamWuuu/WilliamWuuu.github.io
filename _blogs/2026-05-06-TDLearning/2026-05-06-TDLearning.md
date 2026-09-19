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
  - Stochastic Process
  - Markov Decision Process
  - Dynamic Programming
  - Monte Carlo Methods
---

I think blogging about my first time digging into reinforcement learning (RL) theory would be fun.
So boom, here it goes!

This post begins with the classic *Markov decision process (MDP)* formulation, moves through *dynamic programming* and *Monte Carlo Methods*, and then uses those ideas to arrive at *temporal-difference learning*.

# Markov Decision Processes

## Problem Framing

The reinforcement learning (RL) problem is meant to be a straightforward framing of the problem of learning from interaction to achieve a goal, or more vividly, "**trial and error**".

The learner and decision-maker is called the *agent*. The thing it interacts with, comprising everything outside the agent, is called the *environment*. These two interact at each of a sequence of discrete time steps, <span class="math-source" markdown="0">\(t=0,1,2,3,\dots\)</span>. At each time step <span class="math-source" markdown="0">\(t\)</span>, the agent receives some representation of the environment's *state*, <span class="math-source" markdown="0">\(S_t\in\mathcal{S}\)</span>, where <span class="math-source" markdown="0">\(\mathcal{S}\)</span> is the set of possible states, and on that basis selects an *action*, <span class="math-source" markdown="0">\(A_t\in\mathcal{A}(S_t)\)</span>, where <span class="math-source" markdown="0">\(\mathcal{A}(S_t)\)</span> is the set of actions available in state <span class="math-source" markdown="0">\(S_t\)</span>. One time step later, in part as a consequence of its action, the agent receives a numerical *reward*, <span class="math-source" markdown="0">\(R_{t+1}\in\mathcal{R}\subset\mathbb{R}\)</span>, and finds itself in a new state, <span class="math-source" markdown="0">\(S_{t+1}\)</span>.

{% include widgets/blog_image.html src="agent-env.png" caption="Picture 1: The agent–environment interaction in reinforcement learning." %}

Consider the situation when the system starts at a particular state <span class="math-source" markdown="0">\(S_t\in\mathcal{S}\)</span> and continuously taking actions after time step <span class="math-source" markdown="0">\(t\)</span>, resulting in a trajectory like

<div class="math-source" markdown="0">
\[
\tau=(S_t,A_t,R_{t+1},S_{t+1},A_{t+1},R_{t+2},S_{t+2},A_{t+2},R_{t+3},\dots).
\]
</div>

In general, we seek to maximize the *expected return*, where <span class="math-source" markdown="0">\(G_t\)</span> can be defined in the simplest case as the cumulative reward the agent receives after time step <span class="math-source" markdown="0">\(t\)</span>:

<div class="math-source" markdown="0">
\[
\begin{equation}
G_t=R_{t+1}+R_{t+2}+R_{t+3}+\cdots+R_T,
\end{equation}
\]
</div>

where <span class="math-source" markdown="0">\(T\)</span> is the final time step.

Since this expected return will be infinite when <span class="math-source" markdown="0">\(T=\infty\)</span>, the learning process above would possibly fail. In order to prevent this from happening, we introduce a parameter called the *discount rate* <span class="math-source" markdown="0">\(0\leq\gamma\leq 1\)</span> to (1) as:

<div class="math-source" markdown="0">
\[
\begin{equation}
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3}+\cdots = \sum\limits_{k=0}^\infty\gamma^k R_{t+k+1}.
\end{equation}
\]
</div>

If <span class="math-source" markdown="0">\(\gamma&lt;1\)</span>, the infinite sum has a finite value as long as the reward sequence <span class="math-source" markdown="0">\(\{R_k\}\)</span> is bounded. If <span class="math-source" markdown="0">\(\gamma=0\)</span>, the agent is "myopic" in being concerned only with maximizing immediate rewards. As <span class="math-source" markdown="0">\(\gamma\)</span> approaches <span class="math-source" markdown="0">\(1\)</span>, the objective takes future rewards into account more strongly: the agent becomes more farsighted.

## The Markov Property

In the RL framework, the agent makes its decisions as a function of a signal from the environment’s state. In broad sense, “the state” means whatever information that is available to the agent. And we expect the state satisfies *the Markov property*, which we defined as follow.

Consider the response given by a general environment at time <span class="math-source" markdown="0">\({t+1}\)</span> to the action taken at time <span class="math-source" markdown="0">\(t\)</span>, which may depend on everything that has happened earlier. In this case the dynamics can be defined as:

<div class="math-source" markdown="0">
\[
\begin{equation}
\Pr\{S_{t+1}=s^\prime, R_{t+1}=r \vert S_0,A_0,R_1,\dots,S_{t-1},A_{t-1},R_t,S_t,A_t\},
\end{equation}
\]
</div>

for all possible <span class="math-source" markdown="0">\(r\)</span> and <span class="math-source" markdown="0">\(s^\prime\)</span>. If the environment’s response at <span class="math-source" markdown="0">\(t+1\)</span> depends only on the state and action representations at <span class="math-source" markdown="0">\(t\)</span>, i.e.

<div class="math-source" markdown="0">
\[
\begin{equation}
p(s^\prime, r \vert s,a)=\Pr\{S_{t+1}=s^\prime, R_{t+1}=r \vert S_t,A_t\},
\end{equation}
\]
</div>

for all <span class="math-source" markdown="0">\(s^\prime,r,S_t,A_t\)</span>, we say the state signal has the Markov property and is a Markov state.

A reinforcement learning task that satisfies the Markov property is called a *Markov decision process (MDP)*, which models how a system changes state when different actions are applied. Formally, given any state <span class="math-source" markdown="0">\(s\)</span> and action <span class="math-source" markdown="0">\(a\)</span>, the dynamics of an MDP can be specified by:

<div class="math-source" markdown="0">
\[
\begin{equation}
p(s^\prime, r \vert s,a)=\Pr\{S_{t+1}=s^\prime, R_{t+1}=r \vert S_t=s,A_t=a\}.
\end{equation}
\]
</div>

## Value Functions

The value function of a state (or state-action pair) estimates how good or bad an individual is in a given state (or how good it is to perform a given action in a given state). And the notion of "how good" here is defined in terms of expected returns when starting in a specific state and following a specific behavior thereafter. Formally, we call this kind of behavior as a *policy*, <span class="math-source" markdown="0">\(\pi(a\vert s)\)</span>, which is a conditional distribution over the actions <span class="math-source" markdown="0">\(a\in\mathcal{A}\)</span> given the state <span class="math-source" markdown="0">\(s\in\mathcal{S}\)</span>.

For MDPs, we can define the *value* of a state <span class="math-source" markdown="0">\(s\)</span> under a policy <span class="math-source" markdown="0">\(\pi\)</span> as

<div class="math-source" markdown="0">
\[
\begin{equation}
v_\pi(s)
= \mathbb{E}_\pi\left[G_t\vert S_t=s\right]
= \mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\right\vert S_t=s\right],
\end{equation}
\]
</div>

where <span class="math-source" markdown="0">\(\mathbb{E}_\pi\)</span> denotes the expected value of a random variable given that the agent follows policy <span class="math-source" markdown="0">\(\pi\)</span>, and <span class="math-source" markdown="0">\(t\)</span> is any time step. The function <span class="math-source" markdown="0">\(v_\pi\)</span> is called the *state-value function for policy <span class="math-source" markdown="0">\(\pi\)</span>*.

Similarly, we define the value of taking action <span class="math-source" markdown="0">\(a\)</span> in state <span class="math-source" markdown="0">\(s\)</span> under a policy <span class="math-source" markdown="0">\(\pi\)</span> as

<div class="math-source" markdown="0">
\[
\begin{equation}
q_\pi(s,a)
=\mathbb{E}_\pi\left[G_t\vert S_t=s,A_t=a\right]
= \mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\right\vert S_t=s,A_t=a\right].
\end{equation}
\]
</div>

The function <span class="math-source" markdown="0">\(q_\pi\)</span> is called the *action-value function for policy <span class="math-source" markdown="0">\(\pi\)</span>*.

A fundamental property of value functions is that they satisfy particular recursive relationships. The value function can be mathematically decomposed into

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{align*}
v_\pi(s)
&amp;= \mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\right\vert S_t=s\right] \\
&amp;= \mathbb{E}_\pi\left[\left. R_{t+1}+\gamma\sum_{k=0}^{\infty}\gamma^k R_{t+k+2}\right\vert S_t=s\right] \\
&amp;= \sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a) \left[ r + \gamma\mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+2}\right\vert S_{t+1}=s^\prime\right]\right] \\
&amp;= \sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a) \left[r + \gamma v_\pi(s^\prime)\right], \\
\end{align*}
\end{equation}
\]
</div>

which is the foundation of dynamic programming upon which all RL algorithms are based. This is the *Bellman equation for <span class="math-source" markdown="0">\(v_\pi\)</span>*, which expresses a relationship between the value of the state and the values of its successor state.

## Optimal Policy

Solving a RL task roughly means finding an *optimal policy* <span class="math-source" markdown="0">\(\pi^*\)</span> that maximize the expected return, which shares the same state-value function

<div class="math-source" markdown="0">
\[
\begin{equation}
v^*(s)=\max_\pi v_\pi(s),\quad \forall s\in\mathcal{S}
\end{equation}
\]
</div>

and the same action value function

<div class="math-source" markdown="0">
\[
\begin{equation}
q^*(s,a)=\max_\pi q_\pi(s,a),\quad \forall s\in\mathcal{S},a\in\mathcal{A}(s).
\end{equation}
\]
</div>

Intuitively, the value of a state under an optimal policy must equal the expected return for the best action from that state. Starting from this fact, we can derive the so-called *Bellman optimality equation* for <span class="math-source" markdown="0">\(v^*\)</span>:

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{align*}
v^*(s)
&amp;= \max_{a\in\mathcal{A}(s)}q_{\pi_*}(s,a) \\
&amp;= \max_{a\in\mathcal{A}(s)}\mathbb{E}_{\pi^*}\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\right\vert S_t=s,A_t=a\right] \\
&amp;= \max_{a\in\mathcal{A}(s)}\mathbb{E}_{\pi^*}\left[R_{t+1}+\left.\gamma\sum_{k=0}^{\infty}\gamma^k R_{t+k+2}\right\vert S_t=s,A_t=a\right] \\
&amp;= \max_{a\in\mathcal{A}(s)}\mathbb{E}_{\pi^*}\left[R_{t+1}+\left.\gamma v^*(S_{t+1})\right\vert S_t=s,A_t=a\right] \\
&amp;= \max_{a\in\mathcal{A}(s)}\sum_{s^\prime,r} p(s^\prime,r\vert s,a)[r+\gamma v^*(s^\prime)].
\end{align*}
\end{equation}
\]
</div>

Similarly, we can also derive the Bellman optimality equation for <span class="math-source" markdown="0">\(q^*\)</span> is

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{align*}
q^*(s,a)
&amp;= \mathbb{E}_{\pi^*}\left[\left.R_{t+1}+\gamma\max_{a^\prime} q^*(S_{t+1}, a^\prime)\right\vert S_t=s,A_t=a\right] \\
&amp;= \sum_{s^\prime,r}p(s^\prime,r\vert s, a)\left[r+\gamma\max_{a^\prime}q^*(s^\prime,a^\prime)\right].
\end{align*}
\end{equation}
\]
</div>

The Bellman optimality equation is actually a system of equations with <span class="math-source" markdown="0">\(N\)</span> equations and <span class="math-source" markdown="0">\(N\)</span> unknowns. By solving this system of nonlinear equations, we can get <span class="math-source" markdown="0">\(v^*\)</span> and <span class="math-source" markdown="0">\(q^*\)</span>, which determine an optimal policy.

# Dynamic Programming

The key idea of RL generally, is the use of value functions to organize and structure the search for good policies. *Dynamic programming (DP)* refers to a collection of algorithms that can be used to compute the value functions defined earlier, given a perfect model of the environment as an MDP.

## Policy Iteration

*Policy iteration* is one of the ways of finding an optimal policy, which mainly consists of two components, evaluation and improvement. Basically, we hope to achieve that through a sequence:

<div class="math-source" markdown="0">
\[
\pi_0
\xrightarrow{E} v_{\pi_0}
\xrightarrow{I} \pi_1
\xrightarrow{E} v_{\pi_1}
\xrightarrow{I} \pi_2
\xrightarrow{E} \cdots
\xrightarrow{I} \pi_*
\xrightarrow{E} v_{\pi_*},
\]
</div>

where <span class="math-source" markdown="0">\(\xrightarrow{E}\)</span> denotes a *policy evaluation* and <span class="math-source" markdown="0">\(\xrightarrow{I}\)</span> denotes a *policy improvement*.

### Policy Evaluation

Consider how to compute the state-value function <span class="math-source" markdown="0">\(v_\pi\)</span> for an arbitrary policy <span class="math-source" markdown="0">\(\pi\)</span>, which is commonly called *policy evaluation*. Recall that the Bellman equation for <span class="math-source" markdown="0">\(v_\pi\)</span> is formed as

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{align*}
&amp; v_\pi = \sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a) \left[r + \gamma v_\pi(s^\prime)\right] \\
\Rightarrow \quad &amp; v_\pi-\gamma\sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a)\left[v_\pi(s^\prime)\right] = \underbrace{\sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a)\left[r\right]}_{=:r_\pi}
\end{align*}
\end{equation}
\]
</div>

Since both the policy term <span class="math-source" markdown="0">\(\pi(a\vert s)\)</span> and the environment’s dynamics term <span class="math-source" markdown="0">\(p(s^\prime,r\vert s, a)\)</span> are completely known, this form is actually a system of <span class="math-source" markdown="0">\(\vert\mathcal{S}\vert\)</span> linear equations in <span class="math-source" markdown="0">\(\vert\mathcal{S}\vert\)</span> unknowns (<span class="math-source" markdown="0">\(v_\pi(s),s\in\mathcal{S}\)</span>). If we arrange state values, rewards and transition probabilities ​​into matrices:

<div class="math-source" markdown="0">
\[
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
P(s_1,s_1) &amp; P(s_1,s_2) &amp; \cdots &amp; P(s_1,s_{\vert\mathcal{S}\vert}) \\
P(s_2,s_1) &amp; P(s_2,s_2) &amp; \cdots &amp; P(s_2,s_{\vert\mathcal{S}\vert}) \\
\vdots &amp; \vdots &amp; \ddots &amp; \vdots \\
P(s_{\vert\mathcal{S}\vert},s_1) &amp; P(s_{\vert\mathcal{S}\vert},s_2) &amp; \cdots &amp; P(s_{\vert\mathcal{S}\vert},s_{\vert\mathcal{S}\vert})
\end{matrix}
\right]
\end{equation}
\]
</div>

where

<div class="math-source" markdown="0">
\[
\begin{equation}
[P_\pi]_{ij} = P(s_i,s_j) := \sum_{a\in\mathcal{A}(s_i)}\pi(a\vert s_i)\sum_{r\in\mathcal{R}} p(s_j,r\vert s_i,a),
\end{equation}
\]
</div>

then the Bellman equation can be written into the matrix form:

<div class="math-source" markdown="0">
\[
\begin{equation}
(I-\gamma P_\pi)\mathbf{v}_\pi=\mathbf{r}_\pi.
\end{equation}
\]
</div>

Thus we find the solution as:

<div class="math-source" markdown="0">
\[
\begin{equation}
\mathbf{v}_\pi=(I-\gamma P_\pi)^{-1}\mathbf{r}_\pi.
\end{equation}
\]
</div>

Then methods like Gaussian elimination are applied in the linear algebra literature. But RL typically deals with a vast state space (<span class="math-source" markdown="0">\(\vert\mathcal{S}\vert\)</span> is very large) and does not necessarily require an exact solution.

Let's take a deeper look inside. Define the *Bellman expectation operator* <span class="math-source" markdown="0">\(T_\pi\)</span> as

<div class="math-source" markdown="0">
\[
\begin{equation}
(T_\pi v)(s)=r_\pi(s)+\gamma\sum_{s^\prime}P(s,s^\prime)v(s^\prime),
\end{equation}
\]
</div>

where <span class="math-source" markdown="0">\(v:\mathcal{S}\to\mathbb{R}\)</span> is a random value function. If we let <span class="math-source" markdown="0">\(v=v_\pi\)</span>, then we have

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{align*}
(T_\pi v_\pi)(s)
&amp;= r_\pi(s)+\gamma\sum_{s^\prime}P(s,s^\prime)v_\pi(s^\prime) \\
&amp;= \sum_{a} \pi(a\vert s)\sum_{s^\prime}\sum_{r} p(s^\prime,r\vert s, a)\left[r\right] + \gamma\sum_{s^\prime}\left[\left(\sum_{a}\pi(a\vert s)\sum_{r} p(s^\prime,r\vert s,a)\right) v_\pi(s^\prime)\right] \\
&amp;= \sum_{a} \pi(a\vert s)\sum_{s^\prime}\sum_{r} p(s^\prime,r\vert s, a)\left[r\right] + \gamma\sum_{a} \pi(a\vert s)\sum_{s^\prime}\sum_{r} p(s^\prime,r\vert s, a)\left[v_\pi(s^\prime)\right] \\
&amp;= \sum_{a} \pi(a\vert s)\sum_{s^\prime}\sum_{r} p(s^\prime,r\vert s, a)\left[r+\gamma v_\pi(s^\prime)\right] \\
&amp;= v_\pi(s).
\end{align*}
\end{equation}
\]
</div>

Obviously, <span class="math-source" markdown="0">\(v_\pi\)</span> is exactly the fixed-point of <span class="math-source" markdown="0">\(T_\pi\)</span>. So for our purposes, methods like [fixed-point iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration) are most suitable, since the Bellman equation in matrix form is hoped to converge to the fixed-point <span class="math-source" markdown="0">\(\mathbf{v}_\pi\)</span>.

The detailed iteration algorithm is described as follow:

<div class="math-source" markdown="0">
\[
\begin{align*}
&amp;\text{Input }\pi\text{, the policy to be evaluated} \\
&amp;\text{Initialize an array }V(s)=0\text{, for all }s\in\mathcal{S} \\
&amp;\text{Repeat} \\
&amp;\quad \Delta\leftarrow 0 \\
&amp;\quad \text{For each }s\in\mathcal{S} \\
&amp;\quad \quad v\leftarrow V(s) \\
&amp;\quad \quad V(s)\leftarrow \sum_a \pi(a\vert s)\sum_{s^\prime,r} p(s^\prime,r\vert s, a) \left[r + \gamma V(s^\prime)\right] \\
&amp;\quad \quad \Delta\leftarrow\max(\Delta,\vert v-V(s)\vert) \\
&amp;\text{until }\Delta&lt;\theta\text{ (a small positive number)} \\
&amp;\text{Output }V\approx v_\pi
\end{align*}
\]
</div>

### Policy Improvement

Intuitively, we hope the policy gets better and better through the "trial and error" process. Since we are able to compute <span class="math-source" markdown="0">\(v_\pi\)</span> for a policy <span class="math-source" markdown="0">\(\pi\)</span> through policy evaluation, we are able to tell a new policy <span class="math-source" markdown="0">\(\pi^\prime\)</span> is better if

<div class="math-source" markdown="0">
\[
\begin{equation}
v_{\pi^\prime}(s)\geq v_\pi(s),\quad \forall s\in\mathcal{S}
\end{equation}
\]
</div>

which is to say, starting from any state, the expected return of executing <span class="math-source" markdown="0">\(\pi^\prime\)</span> is no less than that of executing <span class="math-source" markdown="0">\(\pi\)</span>. This means a complete iteration algorithm needs to be performed again. The value of the *policy improvement theorem* lies in the fact that we only need to know <span class="math-source" markdown="0">\(v_\pi\)</span> or <span class="math-source" markdown="0">\(q_\pi\)</span> of the old policy to determine whether a certain modification guarantees a better policy.

**Policy improvement theorem.** Let <span class="math-source" markdown="0">\(\pi\)</span> and <span class="math-source" markdown="0">\(\pi^\prime\)</span> be any pair of policies such that

<div class="math-source" markdown="0">
\[
\begin{equation}
q_\pi(s,\pi^\prime(s))\geq v_\pi(s),\quad \forall s\in\mathcal{S},
\end{equation}
\]
</div>

then the policy <span class="math-source" markdown="0">\(\pi^\prime\)</span> must be as good as, or better than, <span class="math-source" markdown="0">\(\pi\)</span>. The proof goes as follow:

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{align*}
v_\pi(s)
&amp; \leq q_\pi(s,\pi^\prime(s)) \\
&amp; = \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma v_\pi(S_{t+1})\vert S_t=s\right] \\
&amp; \leq \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma q_\pi(S_{t+1},\pi^\prime(S_{t+1}))\vert S_t=s\right] \\
&amp; = \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma\mathbb{E}_{\pi^\prime}\left[R_{t+2}+\gamma v_\pi(S_{t+2})\right] \vert S_t=s\right] \\
&amp; = \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma R_{t+2}+\gamma^2 v_\pi(S_{t+2}) \vert S_t=s\right] \\
&amp; \leq \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\gamma^3 v_\pi(S_{t+3}) \vert S_t=s\right] \\
&amp; \dots \\
&amp; \leq \mathbb{E}_{\pi^\prime}\left[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots\vert S_t=s\right] \\
&amp; = v_{\pi^\prime}(s).
\end{align*}
\end{equation}
\]
</div>

Within this theorem, we can easily construct a greedy policy <span class="math-source" markdown="0">\(\pi^\prime\)</span> by selecting at each state the action that appears the best:

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{align*}
\pi^\prime(s)
&amp; = \underset{a\in\mathcal{A}}{\operatorname{argmax}}\, q_\pi(s,a) \\
&amp; = \underset{a\in\mathcal{A}}{\operatorname{argmax}}\, \mathbb{E}\left[R_{t+1}+\gamma v_\pi(S_{t+1})\vert S_t=s,A_t=a\right] \\
&amp; = \underset{a\in\mathcal{A}}{\operatorname{argmax}} \sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left[r+\gamma v_\pi(s^\prime)\right],
\end{align*}
\end{equation}
\]
</div>

which is also known as *one-step lookahead*. In particular, if the new greedy policy <span class="math-source" markdown="0">\(\pi^\prime\)</span> is as good as the old policy <span class="math-source" markdown="0">\(\pi\)</span>, then we have

<div class="math-source" markdown="0">
\[
\begin{equation}
v_{\pi}=v_{\pi^\prime} = \max_{a\in\mathcal{A}} \sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left[r+\gamma v_\pi(s^\prime)\right],
\end{equation}
\]
</div>

which is exactly the same as the Bellman optimality equation. This indicates that we've already find the optimal policy.

## Value Iteration

In essence, policy evaluation is just repeatedly applying Bellman expectation operator to the estimated state-value until convergence, i.e.

<div class="math-source" markdown="0">
\[
\begin{equation}
V \longrightarrow T_{\pi}V \longrightarrow T_{\pi}^2 V \longrightarrow \dots \longrightarrow T_{\pi}^m V,
\end{equation}
\]
</div>

where <span class="math-source" markdown="0">\(m\)</span> tends towards infinity. This process introduces most of the cost during policy iteration. And in fact, it's possible to truncate the evaluation process.

Policy improvement does not require high precision for every state-value. Consider two possible actions at a certain state. Even if the value estimation is still biased, as long as the judgment that one action is better than the other remains unchanged, continuing to calculate <span class="math-source" markdown="0">\(V\)</span> more accurately will not change the next strategy improvement. Therefore, we can pre-set an <span class="math-source" markdown="0">\(m\)</span>, thus only performing a limited number of evaluations. And *value iteration* is a special case of policy iteration that <span class="math-source" markdown="0">\(m\)</span> is set to be <span class="math-source" markdown="0">\(1\)</span>, cutting off policy evaluation to only one round.

We can write the detailed algorithm as follow:

<div class="math-source" markdown="0">
\[
\begin{align*}
&amp;\text{Initialize an array }V(s)=0\text{, for all }s\in\mathcal{S} \\
&amp; \text{Repeat} \\
&amp; \quad \Delta \leftarrow 0 \\
&amp;\quad \text{For each }s\in\mathcal{S} \\
&amp;\quad \quad v\leftarrow V(s) \\
&amp;\quad \quad V(s)\leftarrow \max_a\sum_{s^\prime,r} p(s^\prime,r\vert s, a) \left[r + \gamma V(s^\prime)\right] \\
&amp;\quad \quad \Delta\leftarrow\max(\Delta,\vert v-V(s)\vert) \\
&amp;\text{until }\Delta&lt;\theta\text{ (a small positive number)} \\
&amp; \text{Output a deterministic policy, }\pi\text{, such that} \\
&amp; \quad \pi(s)=\underset{a}{\operatorname{argmax}}\sum_{s^\prime,r}p(s^\prime,r\vert s,a)[r+\gamma V(s^\prime)]
\end{align*}
\]
</div>

# Monte Carlo Methods

*Monte Carlo methods* are ways of solving the reinforcement learning problem based on averaging sample returns. Unlike DP, where we assume complete knowledge of the environment, Monte Carlo methods require only *experience* (sample sequences of states, actions, and rewards from actual or simulated interaction with an environment).

## Monte Carlo Prediction

Let's consider the policy evaluation problem again. First we define an *episode* as one complete journey or trial where an agent interacts with its environment, starting from an initial state and ending at a terminal state. And we call each occurrence of a certain state s in an episode a *visit* to s.

Assume we already have a set of <span class="math-source" markdown="0">\(n\)</span> episodes obtained by following <span class="math-source" markdown="0">\(\pi\)</span> and passing through <span class="math-source" markdown="0">\(s\)</span>. To estimate <span class="math-source" markdown="0">\(v_\pi(s)\)</span>, the value of a certain state <span class="math-source" markdown="0">\(s\)</span> under a given policy <span class="math-source" markdown="0">\(\pi\)</span>, we can simply average the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value. In particular, we introduce *first-visit MC method*, which estimates <span class="math-source" markdown="0">\(v_\pi(s)\)</span> as the average of the returns following first visits to <span class="math-source" markdown="0">\(s\)</span>. Here is the formal algorithm:

<div class="math-source" markdown="0">
\[
\begin{align*}
&amp; \text{Initialize:} \\
&amp; \quad \pi\leftarrow\text{policy to be evaluated} \\
&amp; \quad V\leftarrow\text{an arbitrary state-value function} \\
&amp; \quad \mathrm{Returns}(s)\leftarrow\text{an empty list, for all } s\in\mathcal{S} \\
&amp; \text{Repeat:} \\
&amp; \quad \text{Generate an episode using: }\pi \\
&amp; \quad \text{For each state }s\text{ in the state space }\mathcal{S}: \\
&amp; \quad \quad G\leftarrow \text{return following the first occurrence of }s\text{ in the episode} \\
&amp; \quad \quad \text{Append }G\text{ to }\mathrm{Returns}(s) \\
&amp; \quad \quad V(s)\leftarrow\mathrm{average}(\mathrm{Returns}(s))
\end{align*}
\]
</div>

<details class="proof" markdown="1" open>
<summary>Proof of the almost-sure convergence of first-visit MC method.</summary>

Let <span class="math-source" markdown="0">\(i\)</span>-th episode be

<div class="math-source" markdown="0">
\[
S_0^{(i)},A_0^{(i)},R_1^{(i)},S_1^{(i)},A_1^{(i)},R_2^{(i)},\cdots,S_{T_i}^{(i)},
\]
</div>

where <span class="math-source" markdown="0">\(T_i\)</span> is the time when the episode reaches the terminal state. Define the first-visit time of state <span class="math-source" markdown="0">\(s\)</span> in this episode:

<div class="math-source" markdown="0">
\[
\begin{equation}
\tau_s^{(i)}=
\begin{cases}
\inf\{t\geq 0:S_t^{(i)}=s\} &amp; \text{if visited}\\
\infty &amp; \text{otherwise}\\
\end{cases}
\end{equation}
\]
</div>

If <span class="math-source" markdown="0">\(\tau_s^{(i)}&lt;T\)</span>, the accumulated return following the first-visit is formed as:

<div class="math-source" markdown="0">
\[
\begin{equation}
G_{\tau_s^{(i)}}^{(i)}=R_{\tau_s^{(i)}+1}^{(i)}+\gamma R_{\tau_s^{(i)}+2}^{(i)}+\gamma^2 R_{\tau_s^{(i)}+3}^{(i)}+\cdots\gamma^{T-\tau_s^{(i)}-1}R_T^{(i)}.
\end{equation}
\]
</div>

And our goal is to prove that, for all the episodes containing <span class="math-source" markdown="0">\(s\)</span>, the sample average of <span class="math-source" markdown="0">\(G_{\tau_s}\)</span> converges to <span class="math-source" markdown="0">\(v_\pi(s)\)</span>.

By the definition of <span class="math-source" markdown="0">\(\tau_s\)</span>, we can see that <span class="math-source" markdown="0">\(\tau_s\)</span> is a [stopping time](https://en.wikipedia.org/wiki/Stopping_time) in the random process literature. Because a process under a fixed policy satisfies the *strong Markov property*, once state <span class="math-source" markdown="0">\(s\)</span> is reached at stopping time <span class="math-source" markdown="0">\(\tau_s\)</span>, the conditional distribution of the future trajectory is the same as that of a new process that "starts directly from state <span class="math-source" markdown="0">\(s\)</span> and follows <span class="math-source" markdown="0">\(\pi\)</span>". Define the *filtration* <span class="math-source" markdown="0">\(\mathcal{F}_{\tau_s}\)</span> to be all the information we have up to time <span class="math-source" markdown="0">\(\tau_s\)</span>. Thus we have

<div class="math-source" markdown="0">
\[
\begin{equation}
\mathbb{E}_\pi\left[G_{\tau_s}\vert\mathcal{F}_{\tau_s}\right]=v_\pi(s).
\end{equation}
\]
</div>

For the <span class="math-source" markdown="0">\(i\)</span>-th episode, we define an variable <span class="math-source" markdown="0">\(I^{(i)}\)</span> to indicate whether state <span class="math-source" markdown="0">\(s\)</span> is visited in it, i.e.

<div class="math-source" markdown="0">
\[
\begin{equation}
I^{(i)}=
\begin{cases}
1 &amp; s\text{ is visited} \\
0 &amp; \text{otherwise}
\end{cases}
\end{equation}
\]
</div>

Thus the <span class="math-source" markdown="0">\(V(s)\)</span> we get from the first-visit MC estimation can be written as:

<div class="math-source" markdown="0">
\[
\begin{equation}
V(s)
= \frac{\sum\limits_{i=1}^n I^{(i)} G_{\tau_s^{(i)}}^{(i)}}{\sum\limits_{i=1}^n I^{(i)}}
= \frac{\mathbb{E}\left[ I^{(i)} G_{\tau_s^{(i)}}^{(i)}\right]}{\mathbb{E}\left[ I^{(i)}\right]}.
\end{equation}
\]
</div>

Based on the [law of total expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation), we have

<div class="math-source" markdown="0">
\[
\begin{equation}
V(s)
= \frac{\mathbb{E}\left[ I^{(i)} \mathbb{E}\left[G_{\tau_s^{(i)}}^{(i)}\vert\mathcal{F}_{s^{(i)}}^{(i)}\right]\right]}{\mathbb{E}\left[ I^{(i)}\right]}
= \frac{\mathbb{E}\left[ I^{(i)} v_\pi(s)\right]}{\mathbb{E}\left[ I^{(i)}\right]}
= \frac{\mathbb{E}\left[ I^{(i)}\right] v_\pi(s)}{\mathbb{E}\left[ I^{(i)}\right]}.
\end{equation}
\]
</div>

Since <span class="math-source" markdown="0">\(I^{(i)}\)</span> are i.i.d Bernoulli random variables, as <span class="math-source" markdown="0">\(n\to\infty\)</span>, based on the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers), <span class="math-source" markdown="0">\(\mathbb{E}\left[I^{(i)}\right]=\sum_{i=1}^n I^{(i)}\)</span> converges to some positive number. Thus the estimator <span class="math-source" markdown="0">\(V(s)\)</span> converges to <span class="math-source" markdown="0">\(v_\pi(s)\)</span> obviously.

</details>

The estimation of action values <span class="math-source" markdown="0">\(q_\pi\)</span> is quite similar to state values <span class="math-source" markdown="0">\(v_\pi\)</span>, which is particularly useful when the model is unavailable. However, if <span class="math-source" markdown="0">\(\pi\)</span> is a deterministic policy, when using first-visit method, some state-action pairs may never be visited. What's expected is all the actions from each state, so that we can choose among the actions available in each state. A general approach to assuring that all state–action pairs are encountered is to consider only stochastic policies with a nonzero probability of selecting all actions in each state.

## Monte Carlo Control
Monte Carlo estimation can be used in control, that is, approximating optimal policies. We use the policy iteration scaffold mentioned in the DP section, but replace the objective of policy evaluation from state values to action values, i.e.

<div class="math-source" markdown="0">
\[
\pi_0
\xrightarrow{E} q_{\pi_0}
\xrightarrow{I} \pi_1
\xrightarrow{E} q_{\pi_1}
\xrightarrow{I} \pi_2
\xrightarrow{E} \cdots
\xrightarrow{I} \pi_*
\xrightarrow{E} q_{\pi_*}.
\]
</div>

Policy improvement is done by making the policy greedy with respect to the action-value function, i.e.

<div class="math-source" markdown="0">
\[
\begin{equation}
q_{\pi_k}(s,\pi_{k+1}(s))=q_{\pi_k}(s,\underset{a\in\mathcal{A}}{\operatorname{argmax}} q_{\pi_k}(s,a))=\max_{a\in\mathcal{A}} q_{\pi_k}(s,a),
\end{equation}
\]
</div>

therefore no model is needed to construct the greedy policy.

# Temporal-Difference Learning

TD learning is a combination of Monte Carlo ideas and DP ideas. Like MC methods, TD methods can learn directly from raw experience without a model of the environment’s dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).

## TD Prediction

For state-value estimation, a simple every-visit MC method suitable for non-stationary environments is

<div class="math-source" markdown="0">
\[
\begin{equation}
V(S_t)\leftarrow V(S_t)+\alpha\left[G_t-V(S_t)\right],
\end{equation}
\]
</div>

which waits until the actual return <span class="math-source" markdown="0">\(G_t\)</span> following the visit is known. <span class="math-source" markdown="0">\(\alpha\)</span> is a constant step size parameter. TD methods, instead, wait only until the next time step. The simplest TD method, known as TD(<span class="math-source" markdown="0">\(0\)</span>), is

<div class="math-source" markdown="0">
\[
\begin{equation}
V(S_t)\leftarrow V(S_t)+\alpha\left[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\right],
\end{equation}
\]
</div>

where we replace the target for update from <span class="math-source" markdown="0">\(R_{t+1}\)</span> to <span class="math-source" markdown="0">\(R_{t+1}+\gamma V(S_{t+1})\)</span>. We call this kind of methods which is based on an existing estimate, a *bootstrapping* method.

Recall that in the MDP section, we derive the recursive relationships the value functions satisfies:

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{align*}
v_\pi(s)
&amp;= \mathbb{E}_\pi\left[\left.G_t\right\vert S_t=s\right] \\
&amp;= \mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\right\vert S_t=s\right] \\
&amp;= \mathbb{E}_\pi\left[\left. R_{t+1}+\gamma\sum_{k=0}^{\infty}\gamma^k R_{t+k+2}\right\vert S_t=s\right] \\
&amp;= \sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a) \left[ r + \gamma\mathbb{E}_\pi\left[\left.\sum_{k=0}^{\infty}\gamma^k R_{t+k+2}\right\vert S_{t+1}=s^\prime\right]\right] \\
&amp;= \sum_{a\in\mathcal{A}(s)} \pi(a\vert s)\sum_{s^\prime\in\mathcal{S}}\sum_{r\in\mathcal{R}} p(s^\prime,r\vert s, a) \left[r + \gamma v_\pi(s^\prime)\right]. \\
\end{align*}
\end{equation}
\]
</div>

Roughly speaking, MC methods use an estimate of the first line as the target, whereas DP methods use an estimate of last line as the target.

The detailed algorithm can be written as follow:

<div class="math-source" markdown="0">
\[
\begin{align*}
&amp; \text{Input: the policy }\pi\text{ to be evaluated} \\
&amp; \text{Initialize }\text{ arbitrarily (e.g., }V(s) = 0, \forall s\in\mathcal{S}\text{)} \\
&amp; \text{Repeat (for each episode):} \\
&amp; \quad \text{Initialize }S \\
&amp; \quad \text{Repeat (for each step of episode):} \\
&amp; \quad \quad A\leftarrow\text{action given by }\pi\text{ for }S \\
&amp; \quad \quad \text{Take action }A \text{; observe reward }R \text{, and next state }S^\prime \\
&amp; \quad \quad V(S) \leftarrow V(S)+\alpha\left[R + \gamma V (S^\prime) − V(S)\right] \\
&amp; \quad \quad S \leftarrow S^\prime \\
&amp; \quad \text{until }S\text{ is terminal}
\end{align*}
\]
</div>

The almost sure convergence to <span class="math-source" markdown="0">\(v_\pi(s)\)</span> should be conditioned on <span class="math-source" markdown="0">\(\alpha\)</span> changing with the number of state <span class="math-source" markdown="0">\(s\)</span> is visited. The step size used for the <span class="math-source" markdown="0">\(n\)</span>-th access to state <span class="math-source" markdown="0">\(s\)</span> is denoted as <span class="math-source" markdown="0">\(\alpha_n(s)\)</span>, s.t.

<div class="math-source" markdown="0">
\[
\begin{equation}
0&lt;\alpha_n(s)\leq 1,\quad
\sum_{n=1}^\infty \alpha_n(s)=\infty,\quad
\sum_{n=1}^\infty\alpha_n^2(s)&lt;\infty.
\end{equation}
\]
</div>

The proof of the convergence relies on stochastic approximation theory and contraction mapping properties, which is too long for this blog and therefore omitted.

## Off-policy Learning & On-policy Learning

The methods introduced so far basically estimate the value function <span class="math-source" markdown="0">\(v_\pi\)</span> based on an assumption that, we're given an infinite supply of episodes generated using some given policy <span class="math-source" markdown="0">\(\pi\)</span>. Suppose now that the policy <span class="math-source" markdown="0">\(\pi\)</span> is not available, and all we have are episodes generated from a different policy <span class="math-source" markdown="0">\(\mu\)</span>, where <span class="math-source" markdown="0">\(\mu\neq\pi\)</span>. Then the problem becomes, how do we estimate <span class="math-source" markdown="0">\(v_\pi\)</span>, which is the target of the learning process, using episodes following another policy <span class="math-source" markdown="0">\(\mu\)</span>.

In this sense, <span class="math-source" markdown="0">\(\pi\)</span> and <span class="math-source" markdown="0">\(\mu\)</span> are respectively called *target policy* and *behavior policy*. And there are two possible situations:

- The problem when <span class="math-source" markdown="0">\(\mu=\pi\)</span>, just as we have been discussing all along, is called *on-policy learning*, where the behavior policy and the target policy are the same thing;
- The problem when <span class="math-source" markdown="0">\(\mu\neq\pi\)</span>, is called *oﬀ-policy learning* because it is learning about a policy given only experience “oﬀ” (not following) that policy.

Off-policy algorithms can reuse past training samples, making them more popular.

### Sarsa: On-Policy TD Control

Now let's reconsider our ultimate goal, i.e., the control problem, which is to find the optimal policy. As usual, we follow the pattern of generalized policy iteration, only this time using TD prediction for the policy evaluation part.

Same as learning the state-value <span class="math-source" markdown="0">\(v_\pi\)</span>, we can also use TD(0) as described above to learn the action-value <span class="math-source" markdown="0">\(q_\pi\)</span>:

<div class="math-source" markdown="0">
\[
\begin{equation}
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\left[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\right].
\end{equation}
\]
</div>

This update leverages every element of the quintuple of events:

<div class="math-source" markdown="0">
\[
\begin{equation}
(S_t,A_t,R_{t+1},S_{t+1},A_{t+1}),
\end{equation}
\]
</div>

giving rise to the name *Sarsa* (State-Action-Reward-State-Action) for the algorithm.

Just like all on-policy methods, we continually estimate <span class="math-source" markdown="0">\(q_\pi\)</span> for the behavior policy <span class="math-source" markdown="0">\(\pi\)</span>, and at the same time change π toward greediness with respect to <span class="math-source" markdown="0">\(q_\pi\)</span>. The detailed algorithm is given as follow:

<div class="math-source" markdown="0">
\[
\begin{align*}
&amp; \text{Initialize }Q(s,a)\text{, }\forall s\in\mathcal{S}\text{, }a\in\mathcal{A}(s)\text{, arbitrarily, and }Q(s_\text{terminal},\cdot) = 0 \\
&amp; \text{Repeat (for each episode): }\\
&amp; \quad \text{Initialize }S \\
&amp; \quad \text{Choose }A\text{ from }S\text{ using policy derived from }Q\text{ (e.g., }\epsilon\text{-greedy)} \\
&amp; \quad \text{Repeat (for each step of episode):} \\
&amp; \quad\quad \text{Take action }A\text{, observe }R, S^\prime \\
&amp; \quad\quad \text{Choose }A^\prime\text{ from }S^\prime\text{ using policy derived from }Q\text{ (e.g., }\epsilon\text{-greedy)} \\
&amp; \quad\quad Q(S,A) \leftarrow Q(S,A) + \alpha\left[R+ \gamma Q(S^\prime,A^\prime)−Q(S,A)\right]\\
&amp; \quad \quad S \leftarrow S^\prime;\,A\leftarrow A^\prime; \\
&amp; \quad \text{until }S\text{ is terminal} \\
\end{align*}
\]
</div>

A possible implementation:

```python
class Sarsa:
    """Implementation of Sarsa algorithm."""
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # value table of the Q-function
        self.n_action = n_action  # size of the action space
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # epsilon-greedy policy
	
    def take_action(self, state):
        """Choose the next action."""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
	
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a
	
    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error
```

### Q-Learning: Off-Policy TD Control

The simplest form of *Q-learning*, *one-step Q-learning*, is defined by

<div class="math-source" markdown="0">
\[
\begin{equation}
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\left[R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)\right].
\end{equation}
\]
</div>

In this case, the learned action-value function <span class="math-source" markdown="0">\(Q\)</span>, directly approximates the optimal action-value function <span class="math-source" markdown="0">\(q^*\)</span>.

Let's imagine that a robot takes actions sampled from some policy <span class="math-source" markdown="0">\(\pi_e(a\vert s)\)</span>, collecting a dataset of <span class="math-source" markdown="0">\(n\)</span> episodes of <span class="math-source" markdown="0">\(T\)</span> time-steps each

<div class="math-source" markdown="0">
\[
\{
s^{(i)}_0, a^{(i)}_0,
s^{(i)}_1, a^{(i)}_1,
\dots,
s^{(i)}_{T-1}, a^{(i)}_{T-1},
s^{(i)}_T
\},\quad i=1, 2, \dots, n
\]
</div>

We consider a optimization problem as

<div class="math-source" markdown="0">
\[
\begin{equation}
\hat{Q}=\min\ell(Q)
\end{equation}
\]
</div>

where we denote <span class="math-source" markdown="0">\(\ell(Q)\)</span> as

<div class="math-source" markdown="0">
\[
\begin{equation}
\ell(Q):=\frac{1}{nT}\sum_{i=1}^{n}\sum_{t=0}^{T-1}\left(\underbrace{
Q(s^{(i)}_t,a^{(i)}_t)-\left(
r(s_t^{(i)},a_t^{(i)})+\gamma\max_{a^\prime}Q(s^{(i)}_{t+1},a^\prime)
\right)}_{\text{Bellman Error}}\right)^2.
\end{equation}
\]
</div>

This optimization problem would be identical to the Value iteration if satisfying two ideal conditions:
- The taken policy <span class="math-source" markdown="0">\(\pi_e\)</span> is equal to the optimal policy <span class="math-source" markdown="0">\(\pi^*\)</span>;
- An infinite amount of data is collected.

We can minimize the objective using gradient descent. For every pair <span class="math-source" markdown="0">\((s_t^i,a_t^i)\)</span> in our dataset, we can write

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{align*}
Q(s_t^{(i)},a_t^{(i)})
&amp;\leftarrow Q(s_t^{(i)},a_t^{(i)})-\eta\nabla_{Q(s_t^{(i)},a_t^{(i)})}\ell(Q) \\
&amp;= (1-\eta)Q(s_t^{(i)},a_t^{(i)})+\eta\left(r(s_t^{(i)},a_t^i)+\gamma\max_{a^\prime}Q(s^{(i)}_{t+1},a^\prime)\right)
\end{align*}
\end{equation}
\]
</div>

where <span class="math-source" markdown="0">\(\eta\)</span> is the learning rate.

Given the solution of these updates <span class="math-source" markdown="0">\(\hat{Q}\)</span>, which is an approximation of the optimal value function <span class="math-source" markdown="0">\(Q^*\)</span>, we can obtain the optimal policy corresponding to this value function easily using

<div class="math-source" markdown="0">
\[
\begin{equation}
\hat{\pi}(s)=\arg\max_{a}\hat{Q}(s,a).
\end{equation}
\]
</div>

A possible implementation:

```python
class QLearning:
    """Implementation of Q-learning algorithm."""
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
	
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
	
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a
	
    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error
```

### Coding Practice: The Cliff Walking Example

Consider the grid world shown in the following figure with start and goal states marked as "S" and "G", and the cliff within it is filled with grey.

{% include widgets/blog_image.html src="cliff-walking.png" caption="Picture 2: The cliﬀ-walking task." %}

The action space consists of 4 movement: up, down, right, and left. Reward is <span class="math-source" markdown="0">\(−1\)</span> on all transitions except stepping into the the cliff region, which incurs a reward of <span class="math-source" markdown="0">\(−100\)</span> and sends the agent instantly back to the start.

A possible implementation:

```python
class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        
        # At the initial state S.
        self.x = 0
        self.y = self.nrow - 1
	
    def step(self, action):
		"""One step the agent can take."""
        # change[0]:up, change[1]:down, change[2]:left, change[3]:right
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        
        # position after taking the action
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        
        # flatterned serial number of the current state
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        
        # current location is at the cliff or the target
        if self.y == self.nrow - 1 and self.x > 0:  
            done = True
            # current location is at the cliff
            if self.x != self.ncol - 1:
                reward = -100
        
        return next_state, reward, done
	
    def reset(self):
	    """Return to the initial state."""
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
```

Experiments are conducted in [Google Colab](https://colab.research.google.com/drive/1YaZj9DVwgtc1pczfsdTVWHFH0hg2iO9W?usp=sharing).


# References

[1] Richard S. Sutton, Andrew G. Barto. (2014). Reinforcement Learning: An Introduction. The MIT Press.

[2] Watkins, C. J., Dayan, P. (1992). Technical Note: Q-learning. Machine learning, 8(3-4), 279-292.

[3] Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola. (2023). Dive into Deep Learning. Cambridge University Press.
