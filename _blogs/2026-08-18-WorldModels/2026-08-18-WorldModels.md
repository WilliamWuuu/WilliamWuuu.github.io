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

For example, if a person needs to walk from the table, which is on the left, to the door, his (or her) behavior would definitely not be: take a step to the left <span class="math-source" markdown="0">\(\rightarrow\)</span> bump into the table <span class="math-source" markdown="0">\(\rightarrow\)</span> receive a negative reward <span class="math-source" markdown="0">\(\rightarrow\)</span> update the policy. Instead, he (or she) will first roughly judge in his (or her) mind: the table is there, and I should be able to reach the door by going around to the right. Humans do this kind of simple planning every day. Before we actually take action, we have already "run" the future in a sense.

If machines were to acquire similar capabilities, they would likely need to internally build a model of how the world works, which is precisely the starting point of the research path of world models.

This blog is basically a review of three works on this topic:

- *[Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](http://incompleteideas.net/papers/sutton-90.pdf)* by Richard S. Sutton
- *[World Models](https://arxiv.org/abs/1803.10122)* by David Ha and Jurgen Schmidhuber
- *[A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)* by Yann LeCun

# Why do machines need to imagine?

Just like we've talked, an agent should deduce its best action based on its goal and some internal model capable of simulating how the world works. We call this kind of mechanism *planning*.

In the reinforcement learning literature, an agent decides what to do based on a certain policy, which is essentially a probability distribution of possible actions conditioned on the agent's current state. And this policy is learned via trial-and-error in a real world, i.e., the agent learns from experiences that each actually happens once at least. We call this kind of mechanism *learning*.

Sometimes, we are able to perform instinctive behaviours when we face danger, or act reflectively when consolidating a specific task in repitition, without the need to consciously plan out a course of action. We call this kind of mechanism *reacting*.

*Dyna* is a class of architectures integrating and permitting tradeoffs among these three approaches, including *Dyna-PI* and *Dyna-Q*. This blog mainly introduces the latter one.

## Dyna-PI

*Dyna-PI* is based on approximating *policy iteration*, which we have fully discussed in the [temporal-difference learning blog](https://williamwuuu.github.io/blogs/2026/temporal-difference-learning/). Recall that *policy iteration* is one of the ways of finding an optimal policy through a sequence:

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

{% include widgets/blog_image.html src="Dyna-PI.png" caption="Picture 1: Overview of the proposed Dyna-PI architecture from the paper. The module in the lower left corner is like a single pole, double throw switch. With the 'WORLD' in place as shown we have reinforcement learning; with the 'WORLD MODEL' in place as shown we have planning." %}

The detailed algorithm of Dyna-PI:

<div class="math-source" markdown="0">
\[
\begin{align*}
&amp; \text{1. Decide if this will be a real experience or a hypothetical one;} \\
&amp; \text{2. Pick a state } s \text{. If this is a real experience, use the current state;} \\
&amp; \text{3. Choose an action: } a\sim\pi(\cdot\vert s)\text{;} \\
&amp; \text{4. Perform action } a \text{; obtain next state } s^\prime \text{ and reward } r \text{ from world or world model;} \\
&amp; \text{5. If this is a real experience, update world model } \widehat{\mathcal W} \text{ with } (s,a,s^\prime,r)\text{;} \\
&amp; \text{6. Update evaluation function so that } e(s) \text{ is more like } r+\gamma e(s^\prime)\text{;} \\
&amp; \text{7. Update policy - strengthen or weaken the tendency to perform action } a \text{ in state } s \\
&amp; \quad\,\text{according to the error in the evaluation function: } r+\gamma e(s^\prime)-e(s)\text{;} \\
&amp; \text{8. Go to Step 1.}
\end{align*}
\]
</div>

Now let's deduce how Dyna-PI originates from policy iteration. To avoid confusion, we will uniformly adopt a finite discount Markov decision process (MDP):

<div class="math-source" markdown="0">
\[
\begin{equation}
\mathcal M=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma),\quad 0\leq \gamma&lt;1.
\end{equation}
\]
</div>

Recall that the Bellman expectation operator is defined as:

<div class="math-source" markdown="0">
\[
\begin{equation}
(T_\pi V)(s) =
\sum_{a\in\mathcal A}
\pi(a|s)
\sum_{s^\prime\in\mathcal S} \sum_{r\in\mathcal{R}}
p(s^\prime,r\vert s,a)
\left[ r + \gamma V(s^\prime)\right].
\end{equation}
\]
</div>

Like we've talked about, policy evaluation can be seen as repeatedly applying Bellman expectation operator to the estimated state-value until convergence. And policy improvement is simply choosing a greedy action with respect to estimated state-value for each state. *Policy improvement theorem* gaurantees that accurate evaluation and greedy improvement will eventually lead to an optimal policy.

However, this process imposes two strict requirements on online agents:

- They must sum over all successor states;
- They must know the complete environment model.

Dyna-PI can be understood as the result of successively relaxing these three requirements.

Assume the state at current time <span class="math-source" markdown="0">\(t\)</span> is <span class="math-source" markdown="0">\(S_t=s\)</span>. We denote the action sampled from the policy as <span class="math-source" markdown="0">\(A_t\sim \pi(\cdot\vert s)\)</span>. By performing the action, we can get a feedback from the world as <span class="math-source" markdown="0">\((S_{t+1},R_{t+1}) \sim p(\cdot, \cdot \vert s,A_t)\)</span>. The TD-error is then defined as:

<div class="math-source" markdown="0">
\[
\begin{equation}
\delta_t = R_{t+1}+\gamma V(S_{t+1})-V(S_t).
\end{equation}
\]
</div>

We can calculate the expectation of it conditioned on the current state:

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{align*}
\mathbb{E}_\pi \left[\delta_t \vert S_t=s\right]
&amp; = \sum_{a}\pi(a\vert s)\sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left[r+\gamma V(s^\prime)-V(s)\right] \\
&amp; = \sum_{a}\pi(a\vert s)\sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left[r+\gamma V(s^\prime)\right] - V(s) \\
&amp; = (T_\pi V)(s) - V(s).
\end{align*}
\end{equation}
\]
</div>

Thus the update

<div class="math-source" markdown="0">
\[
\begin{equation}
V(S_t)\leftarrow V(S_t) + \beta \delta_t
\end{equation}
\]
</div>

is a single-sample stochastic approximation of

<div class="math-source" markdown="0">
\[
\begin{equation}
V(S_t)\leftarrow (T_\pi V)(S_t).
\end{equation}
\]
</div>

It does not require explicitly enumerating all actions and successor states, and only require one transition sample.



In particular if <span class="math-source" markdown="0">\(V=v_\pi\)</span>, the expectation of TD-error conditioned on the current state and a certain action can be written as:

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{align*}
\mathbb{E}\left[\delta_t\vert S_t=s,A_t=a\right]
&amp; = \mathbb{E}\left[r+\gamma v_\pi(s^\prime)-v_\pi(s)\vert s,a\right] \\
&amp; = \mathbb{E}\left[r+\gamma v_\pi(s^\prime)\vert s,a\right] - v_\pi(s) \\
&amp; = q_\pi(s,a)-v_\pi(s) \\
&amp; =: A_\pi(s,a),
\end{align*}
\end{equation}
\]
</div>

where <span class="math-source" markdown="0">\(A_\pi\)</span> is called the advantage function, indicating how much better is the action <span class="math-source" markdown="0">\(a\)</span> chosen in state <span class="math-source" markdown="0">\(s\)</span> than the average performance of the current policy. Thus TD-error is also a single-sample approximation of the advantage function.

Assume the policy uses a Boltzmann distribution:

<div class="math-source" markdown="0">
\[
\begin{equation}
\pi_w(a\vert s)=\frac{\exp(w(s,a))}{\sum_b\exp(w(s,b))},
\end{equation}
\]
</div>

where <span class="math-source" markdown="0">\(w(s,a)\)</span> is a preference parameter for every state-action pair. We can form the policy update as:

<div class="math-source" markdown="0">
\[
\begin{equation}
w(S_t,A_t)\leftarrow w(S_t,A_t)+\alpha\delta_t.
\end{equation}
\]
</div>

If an action produces a better action-value than the current estimated state-value (i.e., <span class="math-source" markdown="0">\(\delta&gt;0\)</span>), the preference for that action will increase; if <span class="math-source" markdown="0">\(\delta&lt;0\)</span>, the preference will then decrease.

At this point, we have derived a model-free incremental algorithm from the precise policy iteration:

<div class="math-source" markdown="0">
\[
\begin{equation}
\begin{cases}
\delta_t = R_{t+1}+\gamma V(S_{t+1})-V(S_t) &amp; \leftarrow\text{TD-error}\\
V(S_t)\leftarrow V(S_t) + \beta \delta_t &amp; \leftarrow\text{policy evaluation} \\
w(S_t,A_t)\leftarrow w(S_t,A_t)+\alpha\delta_t &amp; \leftarrow\text{policy improvement}
\end{cases}
\end{equation}
\]
</div>

So where is the world model? Reinforcement learning gets real experiences from the real world, and the world model is expected to provide experiences close to reality, thus reduce the cost spended on agent-world iteractions.

Formally, we can update the world model <span class="math-source" markdown="0">\(\widehat{\mathcal{W}}\)</span> with real experiences like <span class="math-source" markdown="0">\((s,a,s^\prime,r)\)</span>. Then we can use the model to generate a one-step hypothetical experience

<div class="math-source" markdown="0">
\[
\begin{equation}
(\tilde{S}_{t+1},\tilde{R}_{t+1}) \sim \widehat{\mathcal{W}}_t(\cdot,\cdot|S_t,A_t).
\end{equation}
\]
</div>

Updates generated from real experience correspond to "learning"; updates generated from hypothetical experience correspond to "planning". For each experience with the real world, <span class="math-source" markdown="0">\(k\)</span> hypothetical experiences were generated with the model, representing additional planning. The larger <span class="math-source" markdown="0">\(k\)</span> is, the more real-world interactions is usually saved, but more dependent it becomes on the simulation quality of the world.

Now we've successfully constructed a mechanism in which an agent can internally test actions and obtain corresponding possible consequences before actually taking any action.

Mr. Sutton unveiled two potential problems when Dyna-PI is applied in a changing world. One is named *blocking problem*, referring to the fact that the update of the systems's behavior and the world model is too slow when adding a new barrier blocking the original optimal path; The other is named *shortcut problem*, referring to the fact that the system is unable to take the shortcut when removing a barrier that permitts a shorter path than the original optimal path. *Dyna-Q*, which is based on *Q-learning*, was introduced in the original paper to tackle these problems.

# Where should machines imagine?

In the grid-like maze navigation task, we can simply use sequence numbers to represent states, which indicates the agent's position in the maze. But let's consider the case that a robot doing housework in the kitchen, which is essentially acting in the real world. The robots sees images, hears sounds, feels touch, and experiences motion continuously. So if an agent is facing an actual "real world", what should it consider as a "state"?

Asking an agent to understand every single pixel becomes incredibly difficult, as an image contains a wealth of information: color, lighting, texture, shadows, background, entity positions, and relationships between entities. But what truly determines the agent's next action may only be a small fraction of this information. Thus, a natural idea emerged: instead of making predictions directly in the original world, we can first compress the states of world into latent representations, i.e., encode <span class="math-source" markdown="0">\(s\)</span> into <span class="math-source" markdown="0">\(z\)</span>. Then what the world model really needs to learn is a latent transition

<div class="math-source" markdown="0">
\[
z,a \rightarrow z^\prime
\]
</div>

where the reward <span class="math-source" markdown="0">\(r\)</span> is omitted.

## A Feasible Architecture

Now we introduce the proposed agent model in the *World Models* paper.

{% include widgets/blog_image.html src="WorldModels.png" caption="Picture 2: Flow diagram of the proposed Agent model." %}

### Vision Model: Compressing What We See

At each time step, the agent receives a high-dimensional observation <span class="math-source" markdown="0">\(x_t\)</span>, for example, high-resolution images. The task of *Vision Model (V)* is to compress this observation into a low-dimensional latent representation:

<div class="math-source" markdown="0">
\[
x_t \xrightarrow{\text{V}} z_t.
\]
</div>

In the original paper, V is implemented as a *variational autoencoder*. The encoder maps an image into a latent vector <span class="math-source" markdown="0">\(z_t\)</span>, while the decoder tries to reconstruct the original image from it.

This compression is deliberately lossy. The reconstructed image does not preserve every pixel of the original frame, nor does it need to. What matters is that <span class="math-source" markdown="0">\(z_t\)</span> keeps enough information to represent the visually important structure of the current observation.

In this sense, operating on this kind of internally constructed representation instead of the raw observation itself may answer the question we raised earlier. However, there is still an obvious problem. Suppose we show the agent a single image of a car on a racing track. From <span class="math-source" markdown="0">\(z_t\)</span>, it may know roughly where the car and the road are. But a single image does not tell us whether the car is moving quickly or slowly, whether it is turning left or right, or how its current motion will affect what happens next. In other words, <span class="math-source" markdown="0">\(z_t\)</span> compresses what is currently seen, but a world is not just a collection of static scenes. Therefore, in addition to compressing space, the agent also needs to compress time. Since the system evolves over time, one way to achieve this is to record the evolving history, i.e., memory.

### Memory Model: Compressing What Happens over Time

At every time step, the *Memory Model (M)* receives the current latent observation <span class="math-source" markdown="0">\(z_t\)</span>, the action <span class="math-source" markdown="0">\(a_t\)</span> taken by the agent, and its current internal memory <span class="math-source" markdown="0">\(h_t\)</span>. This module has two output heads, one for updating memory:

<div class="math-source" markdown="0">
\[
h_{t+1}=M(z_t,a_t,h_t),
\]
</div>

where <span class="math-source" markdown="0">\(h_t\)</span> can be roughly understood as a compressed summary of information accumulated from the past; the other for predicting the possible next state in the latent space. Within the internal memory as extra input, the world model learns to output a probability distribution:

<div class="math-source" markdown="0">
\[
P(z_{t+1}\mid z_t,a_t,h_t).
\]
</div>

This gives us a much more interesting notion of a state. We may therefore roughly regard

<div class="math-source" markdown="0">
\[
[z_t,h_t]
\]
</div>

as the agent's internal description of its present situation. One part tells it what the world looks like now. The other tells it how the world has been evolving, which contains predictive temporal information unavailable from a single observation.

There is a subtle point here that I find especially interesting. The agent does not always need to explicitly generate several possible futures, inspect them one by one, and then choose the best action. If the predictive structure of the future has already been compressed into <span class="math-source" markdown="0">\(h_t\)</span>, the controller may simply learn to react to that representation. In other words, prediction can affect action even without explicit rollout.

Recall the distinction between planning and reacting we discussed in the previous section. A skilled driver does not consciously simulate every possible trajectory before turning the steering wheel. Years of experience allow information about future consequences to be embedded in the driver's current perception and reflexes. The paper suggests a computational analogue of this idea: a predictive model of the future can provide useful features for a reactive policy, even when the policy does not explicitly "think several steps ahead".

### Controller: Acting through the Internal World

Compared with V and M, the Controller (C) is surprisingly simple. In the original paper, it's only a linear mapping from the current latent representation and the memory state to an action:

<div class="math-source" markdown="0">
\[
W_c[z_t,h_t]+b_c.
\]
</div>

Most of the complexity of the agent resides in the world model rather than in the policy itself. V learns how observations should be represented; M learns how this representation evolves over time; C only needs to learn how to act based on the representations already produced by V and M. Putting these three components together, interaction with the environment looks roughly like

<div class="math-source" markdown="0">
\[
x_t
\xrightarrow{\mathrm{V}}
z_t,
\qquad
(z_t,h_t)
\xrightarrow{\mathrm{C}}
a_t,
\qquad
(z_t,a_t,h_t)
\xrightarrow{\mathrm{M}}
h_{t+1}.
\]
</div>

The real environment then executes <span class="math-source" markdown="0">\(a_t\)</span> and returns the next observation <span class="math-source" markdown="0">\(x_{t+1}\)</span>, and this loop continues.

At first glance, this may look like little more than an unusual architecture for reinforcement learning. But the predictive distribution produced by M introduces a much more radical possibility. If M can tell us what the next latent state is likely to be, why must we ask the real environment for <span class="math-source" markdown="0">\(x_{t+1}\)</span> at all?

## Learning Inside the Dream

Suppose at time <span class="math-source" markdown="0">\(t\)</span>, instead of performing <span class="math-source" markdown="0">\(a_t\)</span> in the real world, we ask M to predict what would happen:

<div class="math-source" markdown="0">
\[
z_{t+1}
\sim
P(z_{t+1}\mid z_t,a_t,h_t).
\]
</div>

Now treat this sampled <span class="math-source" markdown="0">\(z_{t+1}\)</span> as if it were the next observation. Then the controller chooses another action <span class="math-source" markdown="0">\(a_{t+1}\)</span>, M predicts another latent state:

<div class="math-source" markdown="0">
\[
z_{t+2}
\sim
P(z_{t+2}\mid z_{t+1},a_{t+1},h_{t+1}).
\]
</div>

We can continue to rollout in this manner, thus resulting in a trajectory:

<div class="math-source" markdown="0">
\[
z_t
\xrightarrow{a_t}
z_{t+1}
\xrightarrow{a_{t+1}}
z_{t+2}
\xrightarrow{a_{t+2}}
\cdots.
\]
</div>

We call this kind of latent imagination, a *dream*. If an agent can act inside such a dream, can it also learn inside it?

The *VizDoom* experiment pushes the idea much further. The task is simple: the agent needs to avoid fireballs and survive for as long as possible. To turn M into something that can replace the original game environment, the authors extend it slightly. Besides predicting the next latent observation, M also predicts whether the agent will die:

<div class="math-source" markdown="0">
\[
P(z_{t+1},d_{t+1}\mid z_t,a_t,h_t),
\]
</div>

where <span class="math-source" markdown="0">\(d_{t+1}\)</span> indicates whether the episode terminates. With this addition, the learned model contains enough information to expose an interface similar to the original reinforcement-learning environment. The controller can choose an action, receive a new latent state, and eventually receive a termination signal—all without running the actual game engine.

The training procedure can therefore be separated into two stages:

1. Collect experience from the real world and learn the model;
2. Remove the real world and train the controller inside the learned one;
3. Finally, put the controller back into the actual environment.

And remarkably, this works. The controller trained entirely in the generated VizDoom environment transfers back to the real game and successfully solves the task.

Recall that Dyna used the model to generate <span class="math-source" markdown="0">\(k\)</span> hypothetical experiences in addition to every real experience. This experiment pushes <span class="math-source" markdown="0">\(k\)</span> toward an extreme. Once enough real data has been used to learn the model, the agent can stop interacting with reality altogether and perform its subsequent policy learning inside the model. So the world model is no longer merely an auxiliary component that provides additional samples. It becomes an alternative environment in which learning itself can take place.

Perhaps machines do not need to imagine in the raw sensory world at all. They can learn a compressed latent space, learn its dynamics, and then perform their imagination directly inside that space. The world that matters to the agent does not have to be the world as rendered in pixels. It can be an internal world expressed in a language convenient for prediction and control.

## Hazy Dreams

Remember that the vision model is implemented as a VAE, which learns its representation largely by reconstructing observations. This provides a useful and general latent space, but reconstruction itself does not tell us which aspects of the world actually matter for intelligent behavior. For example, a texture on the wall may require many bits to reconstruct accurately but be irrelevant to the task; the position of a small obstacle may occupy only a few pixels but completely determine whether the next action succeeds or fails.

So we arrive at some deeper questions: What should that latent world contain in the first place? Should a world model try to preserve and predict everything it observes? Or should it deliberately ignore some parts of reality and retain only the structures useful for understanding, prediction, and action?

# What should machines imagine?

Since we've already discussed that learning the world model is a spatial-temporal task, a video prediction scenario would be perfect for explaining the formalized idea. Suppose the system is given two video clips in order, and the goal is to tell what degree the second video clip (denoted as <span class="math-source" markdown="0">\(y\)</span>) is a plausible continuation of the first one (denoted as <span class="math-source" markdown="0">\(x\)</span>). The reason we didn't impose the model to predict <span class="math-source" markdown="0">\(y\)</span> directly from <span class="math-source" markdown="0">\(x\)</span> is because there is an infinite number of plausible continuations of a given clip. But it's tractable for the system to evaluate if a proposed <span class="math-source" markdown="0">\(y\)</span> is compatible with a given <span class="math-source" markdown="0">\(x\)</span>. And a general framework of the model to achieve this is the *Energy-Based Models (EBMs)*.

## Energy-Based Models with Latent Variables

Intuitively, we can formally learn a scalar-valued function <span class="math-source" markdown="0">\(F(x,y)\)</span> that produces low energy values when <span class="math-source" markdown="0">\(x\)</span> and <span class="math-source" markdown="0">\(y\)</span> are compatible and higher values when they are not. That seems promising, but the difficulty lies in the fact that the future is not fully predictable from the past. Consider a car approaching a fork in the road. From the current observation, the car may plausibly turn left or turn right; choosing either option is reasonable.

It is evident that some information influencing the future is not contained in <span class="math-source" markdown="0">\(x\)</span>; therefore, we use a latent variable, denoted as <span class="math-source" markdown="0">\(z\)</span>, to represent this missing information. Thus the predictor can be formed as:

<div class="math-source" markdown="0">
\[
\hat{y}=f_\theta(x,z)\in\mathcal{Y}_\theta(x)=\{f_\theta(x,z)\mid z\in\mathcal{Z}\},
\]
</div>

where <span class="math-source" markdown="0">\(\theta\)</span> is the parameter vector of the neural network that computes the energy function <span class="math-source" markdown="0">\(F_w(x,y)\)</span>, defined as:

<div class="math-source" markdown="0">
\[
\begin{align*}
F_\theta(x,y)
&amp; := \min_{z\in\mathcal{Z}}E_\theta(x,y,z) \\
&amp; = \min_{z\in\mathcal{Z}}\Vert y-f_\theta(x,z)\Vert^2,
\end{align*}
\]
</div>

where <span class="math-source" markdown="0">\(E\)</span> is some qualified energy function. When the future <span class="math-source" markdown="0">\(y\)</span> is known, for example during training, we can find an exist <span class="math-source" markdown="0">\(z\)</span> such that the model explains <span class="math-source" markdown="0">\(y\)</span> well based on <span class="math-source" markdown="0">\(x\)</span>. When the future <span class="math-source" markdown="0">\(y\)</span> has not yet been observed, we can enumerate  different values ​​of <span class="math-source" markdown="0">\(z\in\mathcal{Z}\)</span> or sample <span class="math-source" markdown="0">\(z\)</span> from a probability distribution to generate multiple candidate futures.

So how do we train this EBM?

Given a dataset:

<div class="math-source" markdown="0">
\[
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^N,
\]
</div>

where each pair <span class="math-source" markdown="0">\((x_i,y_i)\)</span> is an observed compatible combination. We hope to learn an parameterized energy function <span class="math-source" markdown="0">\(F_\theta(x,y)\)</span> that satisfies the property mentioned earlier.

To achieve that, we need to devise a loss function <span class="math-source" markdown="0">\(\mathcal{L}\)</span>, such that given a training sample <span class="math-source" markdown="0">\((x,y)\)</span>, minimizing this loss will make the energy <span class="math-source" markdown="0">\(F_\theta(x,y)\)</span> lower than the energies <span class="math-source" markdown="0">\(F_\theta(x,y^\prime)\)</span> of any <span class="math-source" markdown="0">\(y^\prime\)</span> diﬀerent from <span class="math-source" markdown="0">\(y\)</span>. Note that the energy function and the training loss are distinct entities. The energy function evaluates a candidate, whereas the training loss assesses whether that energy function distinguishes sufficiently well between different types of candidates. There're usually two kinds of methods to design the loss function. One is called *contrastive methods*, and the other is called *regularized methods*.

<details class="proof" markdown="1" open>
<summary>Details about contrastive methods.</summary>

The basic contrastive loss functions can be formed as:

<div class="math-source" markdown="0">
\[
\mathcal{L}_\theta(x,y,y^\prime)=H(F_\theta(x,y),F_\theta(x,y^\prime),m(y,y^\prime)),
\]
</div>

where <span class="math-source" markdown="0">\(y\)</span> and <span class="math-source" markdown="0">\(y^\prime\)</span> are respectively called a positive sample and a negative sample; <span class="math-source" markdown="0">\(H\)</span> is an increasing function of <span class="math-source" markdown="0">\(F_\theta(x,y)\)</span> and a decreasing function of <span class="math-source" markdown="0">\(F_\theta(x,y^\prime)\)</span>; <span class="math-source" markdown="0">\(m\)</span> is a positive margin function. For example, the function below

<div class="math-source" markdown="0">
\[
\mathcal{L}_\theta(x,y,y^\prime)=\max\{0,F_\theta(x,y)-F_\theta(x,y^\prime)+\mu\Vert y-y^\prime\Vert^2\}
\]
</div>

is a simple instance. Assume that the value of this function is always positive, we can thus take the derivative of the proposed loss function:

<div class="math-source" markdown="0">
\[
\nabla_\theta\mathcal{L}_\theta(x,y,y^\prime)=\nabla_\theta F_\theta(x,y)-\nabla_\theta F_\theta(x,y^\prime).
\]
</div>

We can use gradient-based methods to update the parameter:

<div class="math-source" markdown="0">
\[
\begin{align*}
\theta
&amp; \leftarrow \theta-\eta\nabla_\theta\mathcal{L}_\theta(x_i,y_i,y_i^\prime) \\
&amp; = \theta-\eta[\underbrace{\nabla_\theta F_\theta(x_i,y_i)}_\text{decrease the energy of the positive sample}-\underbrace{\nabla_\theta F_\theta(x_i,y_i^\prime)}_\text{increase the energy of the negative sample}].
\end{align*}
\]
</div>

Actually the contrastive loss function can take multiple contrastive samples into consideration at the same time:

<div class="math-source" markdown="0">
\[
\mathcal{L}_\theta(x,y,y^\prime_1,\dots,y^\prime_K)=H(F_\theta(x,y),F_\theta(x,y^\prime_1),\dots,F_\theta(x,y^\prime_K)),
\]
</div>

where <span class="math-source" markdown="0">\(H\)</span> must be an increasing function of the first argument, and a decreasing function of all other arguments. An example of such loss is the popular *Information Noise-Contrastive Estimation (InfoNCE)* loss:

<div class="math-source" markdown="0">
\[
\mathcal{L}_\theta(x,y,y^\prime_1,\dots,y^\prime_K)=F_\theta(x,y)+\log\left[\exp(-F_\theta(x,y))+\sum_{k=1}^K\exp(-F_\theta(x,y^\prime_k))\right].
\]
</div>

We can take its derivative:

<div class="math-source" markdown="0">
\[
\begin{align*}
&amp; \nabla_\theta\mathcal{L}_\theta(x,y,y^\prime_1,\dots,y^\prime_K) \\
= &amp; \nabla_\theta F_\theta(x,y)+\frac{\nabla_\theta\left[\exp(-F_\theta(x,y))+\sum\limits_{k=1}^K\exp(-F_\theta(x,y^\prime_k))\right]}{\exp(-F_\theta(x,y))+\sum\limits_{k=1}^K\exp(-F_\theta(x,y^\prime_k))} \\
= &amp; \nabla_\theta F_\theta(x,y)+\frac{\exp(-F_\theta(x,y))\cdot(-\nabla_\theta F_\theta(x,y))+\sum\limits_{k=1}^K\left[\exp(-F_\theta(x,y^\prime_k))\cdot(-\nabla_\theta F_\theta(x,y^\prime_k))\right]}{\exp(-F_\theta(x,y))+\sum\limits_{k=1}^K\exp(-F_\theta(x,y^\prime_k))} .
\end{align*}
\]
</div>

Denote:

<div class="math-source" markdown="0">
\[
\begin{align*}
\pi_0 = \frac{\exp(-F_\theta(x,y))}{\exp(-F_\theta(x,y))+\sum\limits_{k=1}^K\exp(-F_\theta(x,y^\prime_k))}\in(0,1), \\
\pi_k = \frac{\exp(-F_\theta(x,y^\prime_k))}{\exp(-F_\theta(x,y))+\sum\limits_{k=1}^K\exp(-F_\theta(x,y^\prime_k))}\in(0,1).
\end{align*}
\]
</div>

Thus we have:

<div class="math-source" markdown="0">
\[
\nabla_\theta\mathcal{L}_\theta(x,y,y^\prime_1,\dots,y^\prime_K)
= (1-\pi_0)\cdot\nabla_\theta F_\theta(x,y)-\sum_{k=1}^K[\pi_k\cdot\nabla_\theta F_\theta(x,y^\prime_k)].
\]
</div>

Obviously it's similar to the single-negative-sample case mentioned earlier.

A problem of contrastive methods is that when <span class="math-source" markdown="0">\(y\)</span> is in a high-dimensional space, it may require a very large number of contrastive samples to ensure that the energy is higher in all dimensions unoccupied by the local data distribution.

</details>

## JEPA

Even if we somehow knew that the car would turn left, there would still be countless details that are difficult or impossible to predict exactly: the movement of leaves, subtle lighting changes, the precise texture appearing on the road, or the exact configuration of distant objects. However, this is the wrong burden to place on a world model. A world model intended for intelligent behavior may not need to know what every leaf will look like one second later. It may only need to know that there is a tree over there, while allocating much more capacity to something like the vehicle ahead may enter my lane.

In other words, part of intelligence may consist not only in predicting the future, but also in learning which parts of the future are worth predicting. This suggests a different objective. Instead of predicting <span class="math-source" markdown="0">\(y\)</span>, perhaps the system should predict an abstract representation of <span class="math-source" markdown="0">\(y\)</span>, denoted as <span class="math-source" markdown="0">\(s_y\)</span>, in which important aspects of the world remain, while irrelevant and unpredictable details disappear?

This is the basic motivation behind *Joint Embedding Predictive Architecture (JEPA)*.

{% include widgets/blog_image.html src="JEPA.png" caption="Picture 3: A diagram of JEPA." %}

A generic JEPA contains three important pieces.

1. Firstly, the two variables <span class="math-source" markdown="0">\(x\)</span> and <span class="math-source" markdown="0">\(y\)</span> are fed to two distinct encoders, producing two latent presentations <span class="math-source" markdown="0">\(s_x\)</span> and <span class="math-source" markdown="0">\(s_y\)</span>. Since the two encoders are not necessarily identical, <span class="math-source" markdown="0">\(x\)</span> and <span class="math-source" markdown="0">\(y\)</span> may represent different types of information (e.g. video and audio);
2. Then a predictor tries to predict the representation of <span class="math-source" markdown="0">\(y\)</span> from the representation of <span class="math-source" markdown="0">\(x\)</span> and optionaly a latent variable <span class="math-source" markdown="0">\(z\)</span>;
3. Finally, the prediction is evaluated by comparing representations: <span class="math-source" markdown="0">\(E_\theta(x,y,z)=D\left(s_y,\operatorname{Pred}(s_x,z)\right)\)</span>, where <span class="math-source" markdown="0">\(D\)</span> measures the discrepancy between the actual representation and its predicted representation.

Since JEPA performs predictions in reperesentation space, the two encoders are free to discard information that is not useful for prediction, e.g. irrelevant details.

### The Training of JEPA

As we've talked about, we use two separate encoders to get latent representations <span class="math-source" markdown="0">\(s_x\)</span> and <span class="math-source" markdown="0">\(s_y\)</span>, which offers great flexibility, but also gives rise to the *collapse* problem.

# References

[1] Richard S. Sutton. Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming. Machine Learning Proceedings 1990, 216-224 (1990).

[2] Ha, David and Schmidhuber, Jürgen. World Models. Zenodo (2018). https://doi.org/10.5281/zenodo.1207631

[3] Yann LeCun and Courant. A Path Towards Autonomous Machine Intelligence. (2022).
