# Exploration: Why we need that?

在前面的几讲中，我们已经给出了很多常见的RL算法。实验上也发现，这些算法在许多简单的任务和环境上面表现都已经十分优秀。但是，当我们把这些算法应用到一些复杂的任务上时，我们会发现它们的表现并不尽如人意。

比如考虑一个简单的例子：在一个游戏中，开始你和一群小兵战斗，但当打死这些小兵之后，你会遇到一个很强大的boss。而如果你被小兵击败，你就会重新复活，回到游戏起点。对于我们的模型来说，其最好的策略可能和我们想象的不太一样——如果打败了小兵，遇到boss，模型会发现很难获胜，因此reward很小；但是如果总是被小兵打死，那么总是可以复活，从而可以不断积累这个比较小的reward。这样的话，模型可能会选择总是被小兵打死，而不去挑战boss。换句话说，我们的模型总是喜欢“摆烂”，不思进取。而为什么我们自己在玩这种游戏的时候不会轻易“摆烂”呢？其实是因为我们清楚这个游戏中各种规则的实际含义（“what these sprites means”）

那么，如何避免我们的模型出现这种情况呢？我们需要让模型去探索环境。比如，模型也许通过某种巧合探索到了击败boss之后会有巨大的reward，它就会学会和boss战斗的技巧。这样，前面的问题就在理论上可以被解决。

但exploration和exploitation（“剥削”，也就是“不思进取”，只在已有的知识上最大化reward）的平衡是困难的。可以像像，如果过分地explore，很可能陷入混乱；而只exploit更容易陷入局部最优。在理论上，在某些简化的情况下，可以推导出最优的exploration策略；但实际上对于我们这些复杂的任务，这些了理论方法往往太复杂以至于无法实现。这就是为何我们要研究exploration。

你可能还记得之前我们在介绍Q learning中提到的exploration 策略（比如 $\epsilon$ -greedy和Boltzmann）。但那些只是最简单的exploration策略。我们这一讲就来介绍一些理论背景更强硬的，更复杂的exploration策略。

# Toy Model: Multi-arm Bandit

“多手臂土匪”模型是一个相对简单的，用来在理论上分析exploration策略的模型。在这一模型中，土匪可以采用 $n$ 中不同的action：

$$
\mathcal{A}= \{a_1,a_2,\cdots,a_n\}
$$

给定其中的一个action，土匪会立刻获得一个reward（因此这一模型被称为stateless的）。这一reward是随机的，服从某种未知分布

$$
r_{i}\sim p(\cdot |a_i)
$$

我们进一步认为，土匪的 $n$ 种操作具有某种共性，因此我们可以找到 $n$ 个参数 $\theta_1,\cdots,\theta_n$ ，使得

$$
p(r|a_i)=p_{\theta_i}(r)
$$

其中 $p_{\theta_i}$ 是一个probability density model。

> 一个简单的例子： $p(r|a_i)=\text{Bernulli}(\theta_i)$ 。当然，接下来的结果对于更一般的情形也是成立的。

这一表述也可以理解为一种**POMDP**(Partially Observable Markov Decision Process)：我们的observation是这里的reward，而其背后的state是 $\theta_i$ 。一个概念叫做**belief state**，它指的是根据我们目前的所有observation情况下推测的概率分布 $\hat{p}(\theta_1,\cdots,\theta_n)$ （注意 $\hat{p}$ 和 $p$ 没有任何关系）。

我们的目标是，最大化

$$
R=\sum_{t=1}^T r(a_t)
$$

注意这个过程包含了我们试错的过程。等价地，我们也可以把这一目标写为**regret**的形式：

$$
\text{Reg}=T\cdot \mathbb{E}_{r\sim p(r|a_n)}[r]-\sum_{t=1}^T r(a_t)
$$

它代表了我们的模型和最优模型之间的差距。接下来，我们介绍三种方法，来最小化这一regret。

## Method 1: Optimistic Exploration/UCB(Upper Confidence Bound)

这一方法有点像“model free”，它不对reward产生的分布做任何假设。它的思想是，对于纯粹exploit的策略，选择action的方法是

$$
a=\arg\max_{a_i}\hat{\mu}_{a_i}
$$

其中 $\hat{\mu}_{a_i}$ 代表我们目前使用 $a_i$ 得到的所有reward的平均值。为了把explore加入，我们注意到 $\hat{\mu}_{a_i}$ 存在误差。其实际值估计应该是

$$
\hat{\mu}_{a_i}\pm C\hat{\sigma}_{a_i}
$$

这种方法的思想是，我们最optimistic，因此取

$$
a=\arg\max_{a_i}[\hat{\mu}_{a_i}+B(a_i)]
$$

其中 $B(a_i)$ 代表给action $a_i$ 加的bonus。理论上可以证明，如果取

$$
B(a_i)=\sqrt{\frac{2\log T}{N(a_i)}}
$$

其中 $N(a_i)$ 代表我们选择了 $a_i$ 的次数，那么我们的regret会以 $\log T$ 的速度增长，这是理论最优的。

## Method 2: Thompson Sampling/Posterior Sampling

Thompson sampling方法稍微加入了一些model based的成分：我们来根据现在的知识预测可能的 $\theta_i$ 值的分布 $\hat{p}(\theta_1,\cdots,\theta_n)$ （就是之前所说的belief state）。接下来，我们从中随机采样一组 $(\theta_1,\cdots,\theta_n)$ ，然后基于这一组参数作出最优的决策。利用这一决策带来的数据，我们可以重新update我们的belief model，从而不断重复。

显然，这一方法在理论上难以分析。但实际上，它跑的很好。

## Method 3: Information Gain

这一方法的思想是：我们要平衡exploitation和exploration。对于exploitation我们已经可以给出一个定量的刻画——reward；但我们并不好说怎么的方法是一个explore。它想到，explore的程度可以用**获得信息的量**来刻画。根据信息理论，定义information gain（也叫做**互信息**）

$$
\text{IG}(a)=\mathcal{H}(\hat{p}({\theta}|h))-\mathbb{E}_{r\sim p(\cdot|a)}\left[\mathcal{H}(\hat{p}(\theta|h,a,r))\right]
$$

其中 $\mathcal{H}$ 代表分布的熵，而 $h$ 代表所有的历史， $\hat{p}(\theta|h)$ 和 $\hat{p}(\theta|h,a,r)$ 代表根据历史（后者比前者多一组数据）训练出的belief state分布。这一表达式的代表了选择action $a$ 之后增加的信息量（减少的不确定性）。最后，我们选取action的方式是，

$$
a=\arg\min_a \frac{\left(r(a^{\star})-\mathbb{E}_{r\sim p(\cdot|a)}[r]\right)^2}{\text{IG}(a)}
$$

这一表达式中，分子代表不应该选取太差的策略（代表适度的exploitation）；而分母代表我们应该选取能够获得最多信息的策略（代表适度的exploration）。

接下来，让我们把这些方法从multi-arm bandit的模型推广到更一般的RL问题中。

# Exploration in Deep RL

## UCB in Deep RL

一个很直接的方法是，我们直接把bonus加在reward function上面：

$$
r^{+}(s,a)=r(s,a)+B(s)=r(s,a)+C\sqrt{\frac{\log T}{N(s)}}
$$

然后，我们把这一reward代入任何之前的算法，然后tune一下这个超参数 $C$ 就可以了。也有其他的bonus的形式（上面这种叫做UCB），比如MBIE-EB：

$$
B_{\text{MBIE-EB}}(s)=\sqrt{\frac{1}{N(s)}}
$$

和BEB：

$$
B_{\text{BEB}}(s)=\frac{1}{N(s)}
$$

但是这里有一个小细节： $N(s)$ 怎么算？我们不能直接数，否则（比如说）对于连续的state，count一定是1。不仅如此，即使对于离散的state，直接数数意义也不大（比如说，假设游戏的图形界面上显示了游戏时间，那么同样的情况可能对应的state就大不相同了）。因此，可以想到，我们需要从 $s$ 中拿出某种特征，然后根据这些特征来计数。接下来就介绍若干计数的方法。

### Counting Method 1: Fitting Generative Model

这一方法的思想是，我们用一个generative model学习 $p_\phi(s)$ ，代表**按照我们见过的所有数据的数据集训练**，state $s$ 出现的概率。

假设有了这样一个model，那么理想情况下，对于一个特定的state $s$ ，有

$$
p_\phi(s)\approx \frac{N(s)}{N}
$$

（其中 $N$ 代表我们见过的所有state的数量），而如果在此基础上再见一次 $s$ ，那么模型更新之后就会近似有

$$
p_{\phi'}(s)\approx \frac{N(s)+1}{N+1}
$$

这样，我们可以近似地给出

$$
N(s)\approx p_{\phi}(s)\cdot \frac{1-p_{\phi'}(s)}{p_{\phi'}(s)-p_\phi(s)}
$$

这一方法还有一个细节：我们应该怎样选取这个"generative model" $p_{\phi}(s)$ 呢？注意这个模型的目的和普通的generative model不一样，它不是为了生成，而是为了计算概率。因此，GAN之类生成效果最吊打但没有概率的的模型并不适用。实际上采用的是一种奇怪的模型，类似于pixel CNN，叫做"CTS"。具体细节可以参考[这篇论文](https://arxiv.org/abs/1606.01868)，它也是Count-Based Exploration方法的开山之作。

### Counting Method 2: Counting with Hash

这一方法的思想是，我们创造一个hash，使得semantic相近的state被映射到接近的hash值。这样，我们就可以用hash值的计数来代替state的计数。

hash函数应该如何选取？一个自然的选取是一个autoencoder，使用它的latent space作为hash。而autoencoder也要随着数据的积累逐渐训练。这样的方法也有很好的效果。

### Counting Method 3: Counting with Exemplar Models

这一方法另辟蹊径，我们可以不采用generative model，而是discriminative model。具体地，我们训练一个classifier，判断某个state是否在历史的数据集中出现过。它的训练集是：

$$
\mathcal{D}_{s}^{(+)}=\{ \text{all historical states} \},\quad \mathcal{D}_{s}^{(-)}=\{s\}
$$

其中 $s$ 代表现在我们要计数的state。然后，我们用这个classifier $D_s$ 来输出 $D_s(s)$ 。

乍一看，这没道理——这应该总是输出label “ $-$ ” 啊! 但是仔细想并不是这样：如果和 $s$ 类似的数据在 $\mathcal{D}_{s}^{(+)}$ 中出现过 $N(s)$ 次，那么大概会有

$$
\Pr(D_s(s)=+)\approx \frac{N(s)}{N(s)+1}
$$

因此我们就可以近似地估计

$$
N(s)\approx \frac{\Pr(D_s(s)=+)}{1-\Pr(D_s(s)=+)}
$$

直观上， $s$ 这个state越是“稀有”，那么被判断为“ $-$ ”的概率越大。因此，这一方法也可以给出一个比较好的计数。但问题在于，为了计算每一个state的N，我们都需要重新训练一个模型，这是非常耗时的。

解决办法类似于VAE中的Amortize方法。我们不是对于每一个 $s$ 训练一个网络 $D_s$ ，而是训练一个 $D(\cdot,s)$ ，而计算的从 $D_s(s)$ 变成 $D(s,s)$ 。这一方法在实验上也取得了很好的结果。

### Counting Method 4: Heuristic estimation using errors

还记得在DL中，我们说模型的generalization 问题：测试的数据和原始的数据集分布差距越大，模型的误差就越差。而我们现在刚好就是想判断一个state是否和历史上的数据相似。因此，我们可以用模型的误差来估计这一点。

具体地，我们随便找一个比较feature的函数（也就是说它不能太简单） $f$ ，然后在见过的数据上面拟合一个模型 $f_\theta$ 。这样，我们只需要计算 $f_\theta$ 和 $f$ 的误差（比如mse loss），就可以估计这个state是否在历史上出现过。误差越大，奖励越高。这一方法也被称为**RND(Random Network Distillation)**。

当然，这个函数有时候并不是随机选取的。比如，可以刚好选取 $s'=f(s,a)$ ，这样看起来更靠谱。不过，具体细节肯定还是要在实验上探讨。

## Thompson Sampling in Deep RL

我们回顾，thompson sampling的方法是说，我们根据现在已有的知识通过模型预测出环境的一些隐藏参数满足的概率分布，然后从中随机采样一组参数，作出决策，再用新的数据重新训练模型。

在multi-arm bandit的模型中，我们预测的就是一组参数 $\theta_1,\cdots,\theta_n$ ，因为我们的目标（也就是reward）完全由它们决定；而怎样将它拓展到一般的情况呢？我们想到，我们应该预测一个Q function。

这样，我们就有一个算法：

> Thompson Sampling

重复：

1. 从当前的模型 $p_\phi(Q)$ 采样一个 $Q$ ；
2. 利用 $Q$ 进行最优决策，跑一个rollout；
3. 用这个rollout的数据更新 $p_\phi(Q)$ 。

我们如何构造一个函数 $Q$ 的分布呢？最自然的方法就是还是采用ensemble（或者叫作**Bootstrapped**）方法，我们训练 $N$ 个网络 $Q_1,\cdots,Q_N$ ，然后随机从中采样一个。更进一步，我们可以保持前面的提取特征的网络不变，只是加上 $N$ 个 projection head。这样，我们就可以在不增加太多参数的情况下实现ensemble。

你可能会感到奇怪，这个方法看起来就是在Q learning的基础上增加了一个ensemble。但这实际上是关键的：还记得原先的 $\epsilon$ -greedy策略，它的exploration通常是乏力的，因为每一步的explore都相当于是随机游走。但现在我们的explore相当于是更加强大的。

> 举个例子：比如一个游戏里，需要连击以获得巨大的奖励。如果每一步随机按照某个方式explore，那么很难找到这个连击的方法。但很有可能，我们的ensemble中的某一个网络刚好学会了这个连击的方法，按照它的Q function来决策，我们就有可能发现这种方式。

## Information Gain in Deep RL

和前一种方法一样，我们也需要想清楚，原先的multi-arm bandit中预测的 $\hat{p}(\theta)$ 现在应该变成什么。（也就是说，计算什么的information gain）。我们有几个选择：

- 计算reward $r(s,a)$ 的information gain，也就是构造一个模型学习 $r_\theta(s,a)$ ，然后计算对 $\theta$ 的熵；
- 计算state density $p(s)$ 的information gain
- 计算dynamic $p(s'|s,a)$ 的information gain

具体的实现中可以采用不同的目标，因为我们的目标函数并不重要，重要的是它反映获得这个state $s$ 之后我们增加了多少信息（这一点体现在 $\theta$ 上）。

我们接下来不管选取哪种目标，而是考虑如何改写之前multi-arm bandit那里的information gain。我们有

$$
\text{IG}(a)=\mathbb{E}_{p((s,a,s')|a)}\left[\mathcal{H}({p}({\theta}|h))-\mathcal{H}({p}(\theta|h,s,a,s'))\right]
$$

可以证明（见下一讲），

$$
\text{IG}(a)=\mathbb{E}_{p((s,a,s')|a)}\left[\text{KL}(p(\theta|h,s,a,s')||p(\theta|h))\right]
$$

而为了方便，我们可以使用单采样进行近似：

$$
\text{IG}(a)\approx \text{KL}(p(\theta|h,s,a,s')||p(\theta|h))
$$

这里的 $h$ 代表所有的历史数据。但是这里的 $\theta$ 可能很复杂，甚至可能是神经网络的参数。因此，我们需要再次使用Bayesian方法，训练的不是一个确定的网络，而是一个分布。而这个分布就是把每一个参数变成从一个高斯分布取样，均值和方差都是另外的可训练参数，记为 $\phi$ 。换句话说，贝叶斯网络的目标是

$$
p(\theta|h)\approx q(\theta|\phi)
$$

> 注意品味这个表达式的意思：左边代表，把历史作为训练集，训练出来可能的网络参数 $\theta$ 的分布；而右边代表，我们拿历史作为训练集，用一种特殊的方法训练出一个“贝叶斯神经网络”，这个网络的参数 $\phi$ 用来预测原先的网络 $\theta$ ，因此给出了一个人为的、近似的分布。

一般来说，贝叶斯网络采取独立假设，即认为每个参数的分布是独立高斯分布

$$
q(\theta|\phi)=\prod_i \mathcal{N}(\theta_i|\mu_{\phi,i},\sigma_{\phi,i})
$$

因此，只有在左边真实的分布也是这样的形式时，才能做的比较好。不过实际上，贝叶斯网络的效果还是很好的。

为了训练这样的一个贝叶斯网络，我们最小化

$$
\text{KL}(q(\theta|\phi)||p(\theta|h))=\text{KL}\left(q(\theta|\phi)\Bigg|\Bigg|p(h|\theta)\frac{p(\theta)}{p(h)}\right)
$$

一般认为 $p(\theta)$ 取自isotropic Gaussian，而 $p(h)$ 作为和 $\theta$ 无关的常数，在forward KL的计算中可以当作全局常数而被去除。

有了这样的工具，我们就可以成功地给出一个exploration bonus的不错的近似：

$$
\text{IG}(a)\approx \text{KL}(q(\theta|\phi')||q(\theta|\phi))
$$

其中 $\phi'$ 代表在加入 $(s,a,s')$ 这一组数据之后的新的参数。使用这个exploration bonus，再使用前面multi-arm bandit中的选择 $a$ 的策略，我们就可以成功地实现一个比较好的exploration。

上面的算法也被称为VIME（Variational Information Maximization Exploration）。它在数学上十分强大，但需要的算力也很巨大，因为每一步都需要训练一个贝叶斯网络。

## Summary

除了上面的三种方法之外，还有一些其他的方法。出于篇幅的考虑，我们就不一一介绍了。

可以看到，无论是前面的哪一种方法，为了计算用于exploration 的bonus，都需要在**每一步**训练一个新的模型。因此，我们也可以看到，为了解决exploration这个困难的问题，我们必定是要付出很大的代价的。

# Reference Papers

1. [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)（RND）
2. [A Possibility for Implementing Curiosity and Boredom in Model-Building Neural Controllers](https://ieeexplore.ieee.org/document/6294131)
3. [Incentivizing Exploration in Reinforcement Learning with Deep Predictive Models](https://arxiv.org/abs/1507.00814)
4. [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)（Bootstrapped Method）
5. [VIME: Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674)（VIME）
6. [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868)（CTS）
7. [\#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](https://arxiv.org/abs/1611.04717)
8. [EX2: Exploration with Exemplar Models for Deep Reinforcement Learning](https://arxiv.org/abs/1703.01260)（Counting with Exemplar Models）