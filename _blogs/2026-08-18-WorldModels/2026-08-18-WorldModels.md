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

{% include widgets/blog_image.html src="Dyna-PI.png" caption="Picture 1: Overview of the proposed Dyna-PI architecture from the paper. The module in the lower left corner is like a single pole, double throw switch. With the "WORLD" in place as shown we have reinforcement learning; with the "WORLD MODEL" in place as shown we have planning." %}

The detailed algorithm of Dyna-PI:

$$
\begin{align*}
& \text{1. Decide if this will be a real experience or a hypothetical one;} \\
& \text{2. Pick a state } x \text{. If this is a real experience, use the current state;} \\
& \text{3. Choose an action: } a\leftarrow\operatorname{Policy}(x)\text{;} \\
& \text{4. Do action } a \text{; obtain next state } y \text{ and reward } r \text{ from world or world model;} \\
& \text{5. If this is a real experience, update world model from } x \text{, } a \text{, } y \text{ and } r\text{;} \\
& \text{6. Update evaluation function so that } e(x) \text{ is more like } r+\gamma e(y)\text{;} \\
& \text{7. Update policy - strengthen or weaken the tendency to perform action } a \text{ in state } x \\
& \quad\,\text{according to the error in the evaluation function: } r+\gamma e(y)-e(x)\text{;} \\
& \text{8. Go to Step 1.}
\end{align*}
$$

Now let's deduce how Dyna-PI originates from policy iteration.




## Dyna-Q



# Where should machines imagine?

# What should machines imagine?


