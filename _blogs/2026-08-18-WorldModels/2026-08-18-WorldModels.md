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

*Dyna* is a class of architectures integrating and permitting tradeoffs among these three approaches, including *Dyna-PI* and *Dyna-Q*. This blog mainly introduces the latter one.

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
& \text{5. If this is a real experience, update world model } \widehat{\mathcal W} \text{ with } (s,a,s^\prime,r)\text{;} \\
& \text{6. Update evaluation function so that } e(s) \text{ is more like } r+\gamma e(s^\prime)\text{;} \\
& \text{7. Update policy - strengthen or weaken the tendency to perform action } a \text{ in state } s \\
& \quad\,\text{according to the error in the evaluation function: } r+\gamma e(s^\prime)-e(s)\text{;} \\
& \text{8. Go to Step 1.}
\end{align*}
$$

Now let's deduce how Dyna-PI originates from policy iteration. To avoid confusion, we will uniformly adopt a finite discount Markov decision process (MDP):

$$
\begin{equation}
\mathcal M=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma),\quad 0\leq \gamma<1.
\end{equation}
$$

Recall that the Bellman expectation operator is defined as:

$$
\begin{equation}
(T_\pi V)(s) =
\sum_{a\in\mathcal A}
\pi(a|s)
\sum_{s^\prime\in\mathcal S} \sum_{r\in\mathcal{R}}
p(s^\prime,r\vert s,a)
\left[ r + \gamma V(s^\prime)\right].
\end{equation}
$$

Like we've talked about, policy evaluation can be seen as repeatedly applying Bellman expectation operator to the estimated state-value until convergence. And policy improvement is simply choosing a greedy action with respect to estimated state-value for each state. *Policy improvement theorem* gaurantees that accurate evaluation and greedy improvement will eventually lead to an optimal policy.

However, this process imposes two strict requirements on online agents: 

- They must sum over all successor states; 
- They must know the complete environment model. 

Dyna-PI can be understood as the result of successively relaxing these three requirements.

Assume the state at current time $t$ is $S_t=s$. We denote the action sampled from the policy as $A_t\sim \pi(\cdot\vert s)$. By performing the action, we can get a feedback from the world as $(S_{t+1},R_{t+1}) \sim p(\cdot, \cdot \vert s,A_t)$. The TD-error is then defined as:

$$
\begin{equation}
\delta_t = R_{t+1}+\gamma V(S_{t+1})-V(S_t).
\end{equation}
$$

We can calculate the expectation of it conditioned on the current state:

$$
\begin{equation}
\begin{align*}
\mathbb{E}_\pi \left[\delta_t \vert S_t=s\right] 
& = \sum_{a}\pi(a\vert s)\sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left[r+\gamma V(s^\prime)-V(s)\right] \\
& = \sum_{a}\pi(a\vert s)\sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left[r+\gamma V(s^\prime)\right] - V(s) \\
& = (T_\pi V)(s) - V(s).
\end{align*}
\end{equation}
$$

Thus the update

$$
\begin{equation}
V(S_t)\leftarrow V(S_t) + \beta \delta_t
\end{equation}
$$

is a single-sample stochastic approximation of 

$$
\begin{equation}
V(S_t)\leftarrow (T_\pi V)(S_t).
\end{equation}
$$

It does not require explicitly enumerating all actions and successor states, and only require one transition sample.



In particular if $V=v_\pi$, the expectation of TD-error conditioned on the current state and a certain action can be written as:

$$
\begin{equation}
\begin{align*}
\mathbb{E}\left[\delta_t\vert S_t=s,A_t=a\right]
& = \mathbb{E}\left[r+\gamma v_\pi(s^\prime)-v_\pi(s)\vert s,a\right] \\
& = \mathbb{E}\left[r+\gamma v_\pi(s^\prime)\vert s,a\right] - v_\pi(s) \\
& = q_\pi(s,a)-v_\pi(s) \\
& =: A_\pi(s,a),
\end{align*}
\end{equation}
$$

where $A_\pi$ is called the advantage function, indicating how much better is the action $a$ chosen in state $s$ than the average performance of the current policy. Thus TD-error is also a single-sample approximation of the advantage function.

Assume the policy uses a Boltzmann distribution:

$$
\begin{equation}
\pi_w(a\vert s)=\frac{\exp(w(s,a))}{\sum_b\exp(w(s,b))},
\end{equation}
$$

where $w(s,a)$ is a preference parameter for every state-action pair. We can form the policy update as:

$$
\begin{equation}
w(S_t,A_t)\leftarrow w(S_t,A_t)+\alpha\delta_t.
\end{equation}
$$

If an action produces a better action-value than the current estimated state-value (i.e., $\delta>0$), the preference for that action will increase; if $\delta<0$, the preference will then decrease.

At this point, we have derived a model-free incremental algorithm from the precise policy iteration:

$$
\begin{equation}
\begin{cases}
\delta_t = R_{t+1}+\gamma V(S_{t+1})-V(S_t) & \leftarrow\text{TD-error}\\
V(S_t)\leftarrow V(S_t) + \beta \delta_t & \leftarrow\text{policy evaluation} \\
w(S_t,A_t)\leftarrow w(S_t,A_t)+\alpha\delta_t & \leftarrow\text{policy improvement}
\end{cases}
\end{equation}
$$

So where is the world model? Reinforcement learning gets real experiences from the real world, and the world model is expected to provide experiences close to reality, thus reduce the cost spended on agent-world iteractions. 

Formally, we can update the world model $\widehat{\mathcal{W}}$ with real experiences like $(s,a,s^\prime,r)$. Then we can use the model to generate a one-step hypothetical experience

$$
\begin{equation}
(\tilde{S}_{t+1},\tilde{R}_{t+1}) \sim \widehat{\mathcal{W}}_t(\cdot,\cdot|S_t,A_t).
\end{equation}
$$

Updates generated from real experience correspond to "learning"; updates generated from hypothetical experience correspond to "planning". For each experience with the real world, $k$ hypothetical experiences were generated with the model, representing additional planning. The larger $k$ is, the more real-world interactions is usually saved, but more dependent it becomes on the simulation quality of the world.

Now we've successfully constructed a mechanism in which an agent can internally test actions and obtain corresponding possible consequences before actually taking any action. 

Mr. Sutton unveiled two potential problems when Dyna-PI is applied in a changing world. One is named *blocking problem*, referring to the fact that the update of the systems's behavior and the world model is too slow when adding a new barrier blocking the original optimal path; The other is named *shortcut problem*, referring to the fact that the system is unable to take the shortcut when removing a barrier that permitts a shorter path than the original optimal path. *Dyna-Q*, which is based on *Q-learning*, was introduced in the original paper to tackle these problems.

# Where should machines imagine?

In the grid-like maze navigation task, we can simply use sequence numbers to represent states, which indicates the agent's position in the maze. But let's consider the case that a robot doing housework in the kitchen, which is essentially acting in the real world. The robots sees images, hears sounds, feels touch, and experiences motion continuously. So if an agent is facing an actual "real world", what should it consider as a "state"?

Asking an agent to understand every single pixel becomes incredibly difficult, as an image contains a wealth of information: color, lighting, texture, shadows, background, entity positions, and relationships between entities. But what truly determines the agent's next action may only be a small fraction of this information. Thus, a natural idea emerged: instead of making predictions directly in the original world, we can first compress the states of world into latent representations, i.e., encode $s$ into $z$. Then what the world model really needs to learn is a latent transition

$$
z,a \rightarrow z^\prime
$$

where the reward $r$ is omitted. 

## A Feasible Architecture

Now we introduce the proposed agent model in the *World Models* paper.

{% include widgets/blog_image.html src="WorldModels.png" caption="Picture 2: Flow diagram of the proposed Agent model." %}

### Vision Model: Compressing What We See

At each time step, the agent receives a high-dimensional observation $x_t$, for example, high-resolution images. The task of *Vision Model (V)* is to compress this observation into a low-dimensional latent representation:

$$
x_t \xrightarrow{\text{V}} z_t.
$$

In the original paper, V is implemented as a *variational autoencoder*. The encoder maps an image into a latent vector $z_t$, while the decoder tries to reconstruct the original image from it.

This compression is deliberately lossy. The reconstructed image does not preserve every pixel of the original frame, nor does it need to. What matters is that $z_t$ keeps enough information to represent the visually important structure of the current observation.

In this sense, operating on this kind of internally constructed representation instead of the raw observation itself may answer the question we raised earlier. However, there is still an obvious problem. Suppose we show the agent a single image of a car on a racing track. From $z_t$, it may know roughly where the car and the road are. But a single image does not tell us whether the car is moving quickly or slowly, whether it is turning left or right, or how its current motion will affect what happens next. In other words, $z_t$ compresses what is currently seen, but a world is not just a collection of static scenes. Therefore, in addition to compressing space, the agent also needs to compress time. Since the system evolves over time, one way to achieve this is to record the evolving history, i.e., memory.

### Memory Model: Compressing What Happens over Time

At every time step, the *Memory Model (M)* receives the current latent observation $z_t$, the action $a_t$ taken by the agent, and its current internal memory $h_t$. This module has two output heads, one for updating memory:

$$
h_{t+1}=M(z_t,a_t,h_t),
$$

where $h_t$ can be roughly understood as a compressed summary of information accumulated from the past; the other for predicting the possible next state in the latent space. Within the internal memory as extra input, the world model learns to output a probability distribution:

$$
P(z_{t+1}\mid z_t,a_t,h_t).
$$

This gives us a much more interesting notion of a state. We may therefore roughly regard

$$
[z_t,h_t]
$$

as the agent's internal description of its present situation. One part tells it what the world looks like now. The other tells it how the world has been evolving, which contains predictive temporal information unavailable from a single observation.

There is a subtle point here that I find especially interesting. The agent does not always need to explicitly generate several possible futures, inspect them one by one, and then choose the best action. If the predictive structure of the future has already been compressed into $h_t$, the controller may simply learn to react to that representation. In other words, prediction can affect action even without explicit rollout.

Recall the distinction between planning and reacting we discussed in the previous section. A skilled driver does not consciously simulate every possible trajectory before turning the steering wheel. Years of experience allow information about future consequences to be embedded in the driver's current perception and reflexes. The paper suggests a computational analogue of this idea: a predictive model of the future can provide useful features for a reactive policy, even when the policy does not explicitly "think several steps ahead".

### Controller: Acting through the Internal World

Compared with V and M, the Controller (C) is surprisingly simple. In the original paper, it's only a linear mapping from the current latent representation and the memory state to an action:

$$
W_c[z_t,h_t]+b_c.
$$

Most of the complexity of the agent resides in the world model rather than in the policy itself. V learns how observations should be represented; M learns how this representation evolves over time; C only needs to learn how to act based on the representations already produced by V and M. Putting these three components together, interaction with the environment looks roughly like

$$
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
$$

The real environment then executes $a_t$ and returns the next observation $x_{t+1}$, and this loop continues.

At first glance, this may look like little more than an unusual architecture for reinforcement learning. But the predictive distribution produced by M introduces a much more radical possibility. If M can tell us what the next latent state is likely to be, why must we ask the real environment for $x_{t+1}$ at all?

## Learning Inside the Dream

Suppose at time $t$, instead of performing $a_t$ in the real world, we ask M to predict what would happen:

$$
z_{t+1}
\sim
P(z_{t+1}\mid z_t,a_t,h_t).
$$

Now treat this sampled $z_{t+1}$ as if it were the next observation. Then the controller chooses another action $a_{t+1}$, M predicts another latent state:

$$
z_{t+2}
\sim
P(z_{t+2}\mid z_{t+1},a_{t+1},h_{t+1}).
$$

We can continue to rollout in this manner, thus resulting in a trajectory:

$$
z_t
\xrightarrow{a_t}
z_{t+1}
\xrightarrow{a_{t+1}}
z_{t+2}
\xrightarrow{a_{t+2}}
\cdots.
$$

We call this kind of latent imagination, a *dream*. If an agent can act inside such a dream, can it also learn inside it?

The *VizDoom* experiment pushes the idea much further. The task is simple: the agent needs to avoid fireballs and survive for as long as possible. To turn M into something that can replace the original game environment, the authors extend it slightly. Besides predicting the next latent observation, M also predicts whether the agent will die:

$$
P(z_{t+1},d_{t+1}\mid z_t,a_t,h_t),
$$

where $d_{t+1}$ indicates whether the episode terminates. With this addition, the learned model contains enough information to expose an interface similar to the original reinforcement-learning environment. The controller can choose an action, receive a new latent state, and eventually receive a termination signal—all without running the actual game engine. 

The training procedure can therefore be separated into two stages: 

1. Collect experience from the real world and learn the model;
2. Remove the real world and train the controller inside the learned one;
3. Finally, put the controller back into the actual environment.

And remarkably, this works. The controller trained entirely in the generated VizDoom environment transfers back to the real game and successfully solves the task. 

Recall that Dyna used the model to generate $k$ hypothetical experiences in addition to every real experience. This experiment pushes $k$ toward an extreme. Once enough real data has been used to learn the model, the agent can stop interacting with reality altogether and perform its subsequent policy learning inside the model. So the world model is no longer merely an auxiliary component that provides additional samples. It becomes an alternative environment in which learning itself can take place. 

Perhaps machines do not need to imagine in the raw sensory world at all. They can learn a compressed latent space, learn its dynamics, and then perform their imagination directly inside that space. The world that matters to the agent does not have to be the world as rendered in pixels. It can be an internal world expressed in a language convenient for prediction and control. 

## Hazy Dreams

Remember that the vision model is implemented as a VAE, which learns its representation largely by reconstructing observations. This provides a useful and general latent space, but reconstruction itself does not tell us which aspects of the world actually matter for intelligent behavior. For example, a texture on the wall may require many bits to reconstruct accurately but be irrelevant to the task; the position of a small obstacle may occupy only a few pixels but completely determine whether the next action succeeds or fails.

So we arrive at some deeper questions: What should that latent world contain in the first place? Should a world model try to preserve and predict everything it observes? Or should it deliberately ignore some parts of reality and retain only the structures useful for understanding, prediction, and action?

# What should machines imagine?

Since we've already discussed that learning the world model is a spatial-temporal task, a video prediction scenario would be perfect for explaining the formalized idea. Suppose the system is given two video clips in order, and the goal is to tell what degree the second video clip (denoted as $y$) is a plausible continuation of the first one (denoted as $x$). The reason we didn't impose the model to predict $y$ directly from $x$ is because there is an infinite number of plausible continuations of a given clip. But it's tractable for the system to evaluate if a proposed $y$ is compatible with a given $x$. And a general framework of the model to achieve this is the *Energy-Based Models (EBMs)*.

## Energy-Based Models with Latent Variables

Intuitively, we can formally learn a scalar-valued function $F(x,y)$ that produces low energy values when $x$ and $y$ are compatible and higher values when they are not. That seems promising, but the difficulty lies in the fact that the future is not fully predictable from the past. Consider a car approaching a fork in the road. From the current observation, the car may plausibly turn left or turn right; choosing either option is reasonable. 

It is evident that some information influencing the future is not contained in $x$; therefore, we use a latent variable, denoted as $z$, to represent this missing information. Thus the predictor can be formed as:

$$
\hat{y}=f_\theta(x,z)\in\mathcal{Y}_\theta(x)=\{f_\theta(x,z)\mid z\in\mathcal{Z}\},
$$

where $\theta$ is the parameter vector of the neural network that computes the energy function $F_w(x,y)$, defined as:

$$
\begin{align*}
F_\theta(x,y)
& := \min_{z\in\mathcal{Z}}E_\theta(x,y,z) \\
& = \min_{z\in\mathcal{Z}}\Vert y-f_\theta(x,z)\Vert^2,
\end{align*}
$$

where $E$ is some qualified energy function. When the future $y$ is known, for example during training, we can find an exist $z$ such that the model explains $y$ well based on $x$. When the future $y$ has not yet been observed, we can enumerate  different values ​​of $z\in\mathcal{Z}$ or sample $z$ from a probability distribution to generate multiple candidate futures.

So how do we train this EBM?

Given a dataset:

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^N,
$$

where each pair $(x_i,y_i)$ is an observed compatible combination. We hope to learn an parameterized energy function $F_\theta(x,y)$ that satisfies the property mentioned earlier.

To achieve that, we need to devise a loss function $\mathcal{L}$, such that given a training sample $(x,y)$, minimizing this loss will make the energy $F_\theta(x,y)$ lower than the energies $F_\theta(x,y^\prime)$ of any $y^\prime$ diﬀerent from $y$. Note that the energy function and the training loss are distinct entities. The energy function evaluates a candidate, whereas the training loss assesses whether that energy function distinguishes sufficiently well between different types of candidates. There're usually two kinds of methods to design the loss function. One is called *contrastive methods*, and the other is called *regularized methods*.

<details class="proof" markdown="1" open>
<summary>Details about contrastive methods.</summary>

The basic contrastive loss functions can be formed as:

$$
\mathcal{L}_\theta(x,y,y^\prime)=H(F_\theta(x,y),F_\theta(x,y^\prime),m(y,y^\prime)),
$$

where $y$ and $y^\prime$ are respectively called a positive sample and a negative sample; $H$ is an increasing function of $F_\theta(x,y)$ and a decreasing function of $F_\theta(x,y^\prime)$; $m$ is a positive margin function. For example, the function below

$$
\mathcal{L}_\theta(x,y,y^\prime)=\max\{0,F_\theta(x,y)-F_\theta(x,y^\prime)+\mu\Vert y-y^\prime\Vert^2\}
$$

is a simple instance. Assume that the value of this function is always positive, we can thus take the derivative of the proposed loss function:

$$
\nabla_\theta\mathcal{L}_\theta(x,y,y^\prime)=\nabla_\theta F_\theta(x,y)-\nabla_\theta F_\theta(x,y^\prime).
$$

We can use gradient-based methods to update the parameter:

$$
\begin{align*}
\theta
& \leftarrow \theta-\eta\nabla_\theta\mathcal{L}_\theta(x_i,y_i,y_i^\prime) \\
& = \theta-\eta[\underbrace{\nabla_\theta F_\theta(x_i,y_i)}_\text{decrease the energy of the positive sample}-\underbrace{\nabla_\theta F_\theta(x_i,y_i^\prime)}_\text{increase the energy of the negative sample}].
\end{align*}
$$

Actually the contrastive loss function can take multiple contrastive samples into consideration at the same time:

$$
\mathcal{L}_\theta(x,y,y^\prime_1,\dots,y^\prime_K)=H(F_\theta(x,y),F_\theta(x,y^\prime_1),\dots,F_\theta(x,y^\prime_K)),
$$

where $H$ must be an increasing function of the first argument, and a decreasing function of all other arguments. An example of such loss is the popular *Information Noise-Contrastive Estimation (InfoNCE)* loss:

$$
\mathcal{L}_\theta(x,y,y^\prime_1,\dots,y^\prime_K)=F_\theta(x,y)+\log\left[\exp(-F_\theta(x,y))+\sum_{k=1}^K\exp(-F_\theta(x,y^\prime_k))\right].
$$

We can take its derivative:

$$
\begin{align*}
& \nabla_\theta\mathcal{L}_\theta(x,y,y^\prime_1,\dots,y^\prime_K) \\
= & \nabla_\theta F_\theta(x,y)+\frac{\nabla_\theta\left[\exp(-F_\theta(x,y))+\sum\limits_{k=1}^K\exp(-F_\theta(x,y^\prime_k))\right]}{\exp(-F_\theta(x,y))+\sum\limits_{k=1}^K\exp(-F_\theta(x,y^\prime_k))} \\
= & \nabla_\theta F_\theta(x,y)+\frac{\exp(-F_\theta(x,y))\cdot(-\nabla_\theta F_\theta(x,y))+\sum\limits_{k=1}^K\left[\exp(-F_\theta(x,y^\prime_k))\cdot(-\nabla_\theta F_\theta(x,y^\prime_k))\right]}{\exp(-F_\theta(x,y))+\sum\limits_{k=1}^K\exp(-F_\theta(x,y^\prime_k))} .
\end{align*}
$$

Denote:

$$
\begin{align*}
\pi_0 = \frac{\exp(-F_\theta(x,y))}{\exp(-F_\theta(x,y))+\sum\limits_{k=1}^K\exp(-F_\theta(x,y^\prime_k))}\in(0,1), \\
\pi_k = \frac{\exp(-F_\theta(x,y^\prime_k))}{\exp(-F_\theta(x,y))+\sum\limits_{k=1}^K\exp(-F_\theta(x,y^\prime_k))}\in(0,1).
\end{align*}
$$

Thus we have:

$$
\nabla_\theta\mathcal{L}_\theta(x,y,y^\prime_1,\dots,y^\prime_K)
= (1-\pi_0)\cdot\nabla_\theta F_\theta(x,y)-\sum_{k=1}^K[\pi_k\cdot\nabla_\theta F_\theta(x,y^\prime_k)].
$$

Obviously it's similar to the single-negative-sample case mentioned earlier.

A problem of contrastive methods is that Second, when $y$ is in a high-dimensional space, it may require a very large number of contrastive samples to ensure that the energy is higher in all dimensions unoccupied by the local data distribution.

</details>

## JEPA

Even if we somehow knew that the car would turn left, there would still be countless details that are difficult or impossible to predict exactly: the movement of leaves, subtle lighting changes, the precise texture appearing on the road, or the exact configuration of distant objects. However, this is the wrong burden to place on a world model. A world model intended for intelligent behavior may not need to know what every leaf will look like one second later. It may only need to know that there is a tree over there, while allocating much more capacity to something like the vehicle ahead may enter my lane.

In other words, part of intelligence may consist not only in predicting the future, but also in learning which parts of the future are worth predicting. This suggests a different objective. Instead of predicting $y$, perhaps the system should predict an abstract representation of $y$, denoted as $s_y$.

The question then becomes:

> Can we learn a representation in which important aspects of the world remain, while irrelevant and unpredictable details disappear?

This is the basic motivation behind *Joint Embedding Predictive Architecture (JEPA)*, the architecture proposed in the paper *A Path Towards Autonomous Machine Intelligence*.

{% include widgets/blog_image.html src="JEPA.png" caption="Picture 3: A diagram of JEPA." %}

A generic JEPA contains three important pieces. First, instead of operating directly on $x$ and $y$, it encodes both of them:

$$
s_x=\operatorname{Enc}_x(x),\qquad s_y=\operatorname{Enc}_y(y).
$$

The two encoders do not even have to be identical. In principle, $x$ and $y$ may represent different types of information. Then a predictor tries to predict the representation of $y$ from the representation of $x$:

$$
\operatorname{Pred}(s_x,z),
$$

where $z$ is an optional latent variable that we will discuss shortly. Finally, the prediction is evaluated not by comparing generated observations, but by comparing representations:

$$
D
\left(
s_y,
\operatorname{Pred}(s_x,z)
\right).
$$

Here $D$ measures the discrepancy between the actual representation $s_y$ and its predicted representation $\tilde{s}_y$. In the language of the paper, this discrepancy can be interpreted as an energy: compatible pairs of $x$ and $y$ should have low energy, while incompatible pairs should have high energy. 

The architectural difference may initially appear small.

A generative model performs something like

$$
x\rightarrow y.
$$

JEPA performs

$$
\operatorname{Enc}(x)
\rightarrow
\operatorname{Enc}(y).
$$

But the consequence is significant.

Because the target $s_y$ is itself learned, the encoder is free to discard information about $y$ that is not useful for prediction.

Suppose two possible futures differ only in the precise movement of leaves on a tree. A generative model must somehow account for the difference between the two images. A JEPA may simply learn

$$
\operatorname{Enc}(y_1)
\approx
\operatorname{Enc}(y_2),
$$

if that difference is irrelevant to the abstract structure being represented. The uncertainty has disappeared from representation space. Not because the model has successfully predicted the movement of every leaf, but because it has learned that those movements do not need to be represented in the first place.

## What Should Be Ignored, and What Should Remain Uncertain?

Of course, not every unpredictable event can simply be discarded. Return to the car approaching a fork. Whether it turns left or right may be uncertain from the current observation, but it clearly matters. If our encoder mapped both futures to exactly the same representation,

$$
s_{\mathrm{right}},
$$

the representation would be predictable, but useless for navigation.

JEPA therefore distinguishes, at least conceptually, between two kinds of uncertainty. One kind can be removed by invariance. If differences between two possible observations are irrelevant, their encoder representations can become similar. The precise leaf configuration changes, but:

$$
\operatorname{Enc}
(\text{tree with leaves in configuration A})
\approx
\operatorname{Enc}
(\text{tree with leaves in configuration B}).
$$

The other kind of uncertainty must remain represented. If the future genuinely contains several distinct possibilities that matter, the predictor may use the latent variable (z):

$$
\operatorname{Pred}(s_x,z).
$$

Different values of (z) can correspond to different compatible futures. For example,

$$
z=z_{\mathrm{left}}
$$

may produce a representation corresponding to a left turn, while

$$
z=z_{\mathrm{right}}
$$

produces a representation corresponding to a right turn.

The paper therefore gives JEPA two ways of handling a world with many possible futures: ignore variations that should not matter through representation invariance; represent meaningful uncertainty through the latent variable (z). This is subtly different from asking a generative model to reproduce every possible version of the future. The goal is not to explain all uncertainty. It is to separate uncertainty that matters from uncertainty that does not. 

## Predictable, but Not Empty

At this point, however, JEPA appears to have an embarrassingly simple solution.

If the goal is to make $s_y$ easy to predict from $s_x$, why not let both encoders output exactly the same constant vector for every input? For example,

$$
\operatorname{Enc}_x(x)=0,
$$

and

$$
\operatorname{Enc}_y(y)=0.
$$

Then the predictor simply outputs

$$
\tilde{s}_y=0,
$$

and achieves perfect prediction. The world has become completely predictable. Unfortunately, the representation contains absolutely no information about the world. This is known as representational collapse. It reveals the real difficulty of learning abstractions. We do not simply want predictable representations. We want representations that are simultaneously:

$$
\boxed{\text{informative}}
$$

and

$$
\boxed{\text{predictable}}.
$$

These two objectives pull the model in opposite directions. If we preserve every detail of an observation, the representation is highly informative but much of it becomes difficult to predict. If we remove everything, prediction becomes trivial but the representation becomes useless.

JEPA therefore needs to find a point somewhere between these two extremes. The paper describes this using four broad training criteria:

- $s_x$ should retain substantial information about $x$;
- $s_y$ should retain substantial information about $y$;
- $s_y$ should be predictable from $s_x$;
- the latent variable $z$ should contain as little information as necessary.

The first two criteria prevent the encoders from collapsing into uninformative constant representations. The third encourages the model to discover predictable structure. The fourth prevents another trivial solution: simply storing the entire target $y$ inside $z$, allowing the predictor to reconstruct $s_y$ without learning anything from $x$. So the representation is placed under two opposing pressures:

$$
\text{preserve information}
\quad\leftrightarrow\quad
\text{discard unpredictability}.
$$

The result, ideally, is an abstraction containing as much information as possible about the world while excluding details that prevent useful prediction. This gives us a more precise answer to our original question. A machine should not imagine everything. It should imagine the largest part of the world that can be represented meaningfully and predicted reliably.

## From Compression to Abstraction

This gives us an interesting point of comparison with World Models.

The Vision Model in World Models also maps observations into latent representations:

$$
x\rightarrow z.
$$

JEPA similarly maps observations into representations:

$$
x\rightarrow s_x.
$$

So superficially, both approaches seem to say the same thing: Do not model the world directly in pixel space. Compress it first.

But there is an important conceptual difference in what shapes the representation. In World Models, the VAE is trained to make (z) useful for reconstructing (x). The basic pressure is:

$$
z
\quad\text{should preserve enough information to recover }x.
$$

JEPA instead asks that the representation become useful for predicting another representation. Its pressure is closer to:

$$
s_x
\quad\text{should preserve information that helps predict }s_y.
$$

This turns compression into abstraction. Compression asks:

> How can I describe this observation using fewer numbers?

Abstraction asks:

> Which distinctions in this observation should matter at all?

The difference is fundamental. A compressed representation of a photograph may still preserve texture, color and lighting because those features help reconstruct the image. An abstract predictive representation is allowed to decide that many of those distinctions are irrelevant. The internal world is therefore not merely a lower-resolution version of reality. It can have a different ontology. Two observations that look very different in pixel space may correspond to essentially the same state in the agent's internal world. And two observations differing by only a few pixels may correspond to very different states if those few pixels signal something important for future behavior. This is perhaps the deepest answer JEPA provides to the question:

> What should machines imagine?

Not an accurate copy of everything they see. Rather, an abstract world in which the distinctions preserved are those needed to make the future predictable.

## Different Futures Require Different Levels of Abstraction

There is one more problem. Even if we know that the world should be represented abstractly, there may not be a single correct level of abstraction. Imagine planning how to travel from home to another country. At a short time scale, very detailed information matters: move the steering wheel slightly to the left. At an intermediate scale: drive to the airport. At a longer time scale: fly to Singapore. When planning the entire journey, it would be absurd to predict the exact angle of the steering wheel several hours into the future. Those details matter locally, but they are meaningless for long-term prediction.

This suggests that abstraction should increase with prediction horizon. At low levels, representations can preserve relatively detailed information and support short-term prediction. At higher levels, they should discard more details and represent slower, more abstract changes. The paper therefore proposes extending JEPA into a Hierarchical JEPA, or H-JEPA.

Conceptually,

$$
\text{JEPA}_1
$$

operates on relatively detailed representations and predicts over short time scales. Its representations are then fed into

$$
\text{JEPA}_2,
$$

which produces a more abstract representation and predicts farther into the future. More levels can, in principle, continue this process:

$$
\text{detailed state}
\rightarrow
\text{abstract state}
\rightarrow
\text{more abstract state}
\rightarrow
\cdots
$$

As the level increases, details that are difficult to predict over long horizons can progressively disappear. The paper summarizes the intuition clearly: low-level representations may contain enough detail for short-term prediction, while higher-level representations sacrifice detail in exchange for predictions over longer time scales. This creates an internal world with not one, but multiple temporal resolutions.

A machine may imagine:

$$
\text{where my hand will be in 0.2 seconds}
$$

using a detailed representation, while imagining:

$$
\text{whether I will have completed cooking dinner in 20 minutes}
$$

using a much more abstract one. Both predictions refer to the same world. They simply describe it at different levels.

## From Hierarchical Prediction to Hierarchical Planning

This hierarchy also provides a possible answer to a problem that appeared at the very beginning of this blog. Planning over long action sequences is extremely difficult. Suppose we want a robot to prepare a cup of coffee. At the highest level, a useful plan might look like:

$$
\text{get cup}
\rightarrow
\text{make coffee}
\rightarrow
\text{serve coffee}.
$$

But each of these operations must eventually be decomposed. For example,

$$
\text{get cup}
$$

may become

$$
\text{walk to cabinet}
\rightarrow
\text{open cabinet}
\rightarrow
\text{grasp cup}.
$$

And grasping itself eventually becomes a sequence of low-level motor commands. Trying to optimize all of these motor commands simultaneously over the full time horizon would create an enormous search problem.

With a hierarchical world model, the agent could instead first plan in a highly abstract space. The actor proposes a sequence of abstract actions. The world model predicts their abstract consequences. A cost module evaluates whether the predicted future is desirable.

Once a high-level plan is selected, each abstract action can be passed downward and decomposed into increasingly concrete subgoals, until the lowest level produces executable actions.

This is one of the broader ambitions of *A Path Towards Autonomous Machine Intelligence*: not merely to learn representations, but to use hierarchical predictive world models as the basis of planning across multiple time scales. 

The architecture therefore closes a loop:

$$
\text{Perception}
\rightarrow
\text{Abstract State}
\rightarrow
\text{Predict Future States}
\rightarrow
\text{Evaluate Futures}
\rightarrow
\text{Choose Actions}.
$$

This brings us back remarkably close to Dyna.
