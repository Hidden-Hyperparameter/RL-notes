# Value Function Methods

作为引入，我们考虑能否不给模型指定一个明确的policy，而是直接选取当前最优的action。这样的方法称为value function methods。

具体地，假设我们知道了使用策略$\pi$时，当前state $s_t$ 下面各个action $a_t$ 对应的value $Q^\pi(s_t,a_t)$，那么我们可以直接选取最优的action：

$$
a_t^\star=\arg\max_{a_t}Q^\pi(s_t,a_t)
$$

这样，只要我们可以得到 Q-function 的值就可以了。应该如何实现这一点呢？我们可以使用前面第4讲中给出的递推关系

$$
V^{\pi}(s_t)=\mathbb{E}_{a_t\sim\pi(a_t|s_t)}\left[Q^{\pi}(s_t,a_t)\right]=\mathbb{E}_{a_t\sim\pi(a_t|s_t)}\left[r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}(s_{t+1})]\right]
$$

和

$$
Q^{\pi}(s_t,a_t)=r(s_t,a_t)+\gamma\mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}\left[V^{\pi}(s_{t+1})\right]
$$

对于有限大小的state space，我们就可以用不断迭代的方法计算它们。应用在V上，我们就有以下的**policy iteration**算法：

> **policy iteration**

重复:
1. Policy Evaluation: 计算 

$$
V^\pi(s_t)\leftarrow r(s_t,\pi(s_t))+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,\pi(s_t))}[V^{\pi}(s_{t+1})]
$$

2. 根据新的value计算最新策略：
$$
\pi(s_t)=\arg\max_{a_t}r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}(s_{t+1})]
$$

这就是value-based方法的基本思路。

## Value Iteration

首先，从上面的表达开始，我们可以用Q function简化记号，把两步拆为三步：
1. $V^\pi(s_t)\leftarrow Q^\pi(s_t,\pi(s_t))$
2. $Q^\pi(s_t,a_t)\leftarrow r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}(s_{t+1})]$
3. $\pi(s_t)\leftarrow \arg\max_{a_t} Q^\pi(s_t,a_t)$

然后，把$\pi$消去，就得到了**value iteration**算法：

$$
V^\pi(s_t)\leftarrow \max_{a_t}\left\{r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}(s_{t+1})]\right\}
$$

### Fitting Methods

前面的方法基于dynamic programming，也就是必须列出每一个位置处的Q或者V。但是对于很高的维度，我们完全不能这样做，此时需要训练一个网络来拟合Q或者V，称为fitting methods。

最简单的推广是**Fitted Value Iteration**：我们从前面value iteration算法的表达式出发，但是把V换成一个网络。

> **Fitted Value Iteration**

重复:
1. 计算新的value：
$$
[V^\pi(s_t)]^\star=\max_{a_t}\left\{r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}_\phi(s_{t+1})]\right\}
$$

2. 最小化拟合误差：
$$
\phi\leftarrow \arg\min_\phi \sum_{s_t}\left(V^\pi_\phi(s_t)-[V^\pi(s_t)]^\star\right)^2
$$

注意，$[V^\pi(s_t)]^\star$中的$\star$代表这个数是网络的训练目标。注意计算的时候右边 $r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}_\phi(s_{t+1})]$也和$\phi$相关，因此需要**stop-gradient**。

同时，也应该注意，第二步并不是一个gradient step，而是很多步，因为这一步代表着让$V^\pi_\phi$接近于之前“动态规划”那样方法的准确值，神经网络的学习必须跟上value function update的进度。而1-2作为一整轮，才相当于之前“动态规划”方法中update $V$的方式。

## Fitting Q Iteration

我们很容易发现前面算法的重要问题：第一步必须涉及对不同的$a_t$获得很多reward和$s_{t+1}$，并计算最大值。这样的操作导致其必须和环境进行反复交互。能否解决这个问题？

可以发现，如果反过来拟合 Q function，就不存在这样的问题了！因为，回顾我们之前的三步

1. $V^\pi(s_t)\leftarrow Q^\pi(s_t,\pi(s_t))$
2. $Q^\pi(s_t,a_t)\leftarrow r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}(s_{t+1})]$
3. $\pi(s_t)\leftarrow \arg\max_{a_t} Q^\pi(s_t,a_t)$

此时如果不是消去Q而是消去V，那么就会得到

$$
Q^\pi(s_t,a_t)\leftarrow r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}\left[\max_{a_{t+1}}Q^{\pi}(s_{t+1},a_{t+1})\right]
$$

这样的更新关系。注意到现在最大值的位置跑到了内部！这样我们不需要和环境多次交互，只需要多跑几次神经网络就可以了。

这个方法就是著名的**Q Iteration Algorithm**。当然，代价也是明显的：我们的网络需要输入$s_t,a_t$两个参数，因此拟合的难度也比较高。但考虑到DL的巨大发展已经为人们扫平了大部分的障碍，这个方法完全瑕不掩瑜。

最后，我们也很容易把它推广到Fitted Q Iteration，在此作一总结：

> **Fitted Q Iteration Algorithm**

重复：
1. 收集数据集，包含一系列的$\{s_t,a_t,s_{t+1},r(s_t,a_t)\}$（有时候，为了方便，也会记为 $\{s,a,s',r\}$，需要搞清楚它们的对应关系）；
2. 重复$K$次：

2.1. 计算新的$Q$：

$$
[Q^\pi(s_t,a_t)]^\star=r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}\left[\max_{a_{t+1}}Q^{\pi}_\phi(s_{t+1},a_{t+1})\right]
$$

2.2. 最小化拟合误差：（注意，这一步本身也可能包含很多次gradient step，总数记为 $S$ ）

$$
\phi \leftarrow \arg\min_\phi \sum_{s_t,a_t}\left(Q^\pi_\phi(s_t,a_t)-[Q^\pi(s_t,a_t)]^\star\right)^2
$$

注意到这个算法的另外一个性质是，我们无论怎样更新$\pi$，我们原先用来update $Q$的数据仍然可以**重复利用**。这样的性质允许了上面“重复$K$轮”的操作，也使得这个算法非常适合于off-policy的任务。当然，也必须提及另外一个极端——如果我们让$K\to \infty$，也就是一直只在很少的样本上面重复训练，就会产生问题：我们并不了解没有采样到的地方是否会有更好的策略。因此，和这个算法匹配的必定是一些能够实现**exploration**的算法。

当然，实验给出的最优方式就是像上面的算法描述的那样：对同样的一组数据训练$K$轮，然后作exploration，再重新根据某种policy $\pi$来收集数据，再训练$K$轮，如此循环。

## Q learning

Q learning是Q iteration的online版本。具体地，我们每一次和环境交互一次，立刻用上面Q iteration的方法处理得到的数据，并根据一个策略继续交互。

> **Q Learning Algorithm**

重复：
1. 从环境中根据某种policy采样一个$(s,a,s',r)$；
2. 计算**一个** $[Q^\pi(s,a)]^\star=r(s,a)+\gamma\max_{a'}Q^{\pi}_\phi(s',a')$
3. 对 $L=([Q^\pi(s,a)]^\star-Q^{\pi}_\phi(s,a))^2$作**一步**梯度下降。

这些步骤保证了它是一个online的算法。当然，就如之前所说，这里用来第一步采样的“某种policy”并不一定是当前$Q$值分布下的最佳policy；但从另一个角度来看，这个policy多少应该贴近现在的最优policy，这样有利于模型“加强训练”。除此之外，必须注意我们也需要有一个exploration的机制，否则我们可能会陷入一个很差的解。

这一切都使得**Q learning中的exploration**成为一个非常重要的问题。我们将在之后的某讲讨论这个问题。但在现在，可以根据intuition给出一些介绍：

- **$\epsilon$-greedy**：以$1-\epsilon$的概率选择最优的action，以$\epsilon$的概率随机选择其他的一个action；
- **Boltzmann**：$\pi(a|s)=\frac{1}{Z}e^{Q(s,a)/\tau}$。注意这个方法相比于$\epsilon$-greedy的合理性：
    - 如果有两个Q很接近的action，那么它们被选中的概率应该接近；
    - 如果有两个都不是最优，但从Q上能明确分清主次的action，那么它们被选中的概率应该也有一定差距。

# Theoretical Analysis

## Optimal Policy

我们把Q iteration的loss写出来：

$$
L=\mathbb{E}_{(s,a)}\left[\left(Q^\pi(s_t,a_t)-r(s_t,a_t)-\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}\left[\max_{a_{t+1}}Q^{\pi}(s_{t+1},a_{t+1})\right]\right)^2\right]
$$

（这里先忽略了fitting的loss）我们就可以注意到理论最优的策略$\pi_\star$必然满足$L=0$。这给了我们很重要的希望：理论上，我们可以通过训练一个Q function来找到最优的policy。

## Convergence

但是实际上必须考虑的问题就是convergence。我们可以引入Bellman operator $\mathcal{B}$ 来研究收敛性。

我们还是以Q iteration为例，我们可以把Q iteration的更新写成一个operator的形式：

$$
Q\leftarrow \mathcal{B}Q
$$

其中

$$
\mathcal{B}Q=r+\gamma T\max_{a}Q
$$

其中$T$是第四讲提到的transition matrix，把$Q$的$s_{t+1}$维度transform到$(s_t,a_t)$两个维度。关键在于，$\mathcal{B}$这个operator具有一定的性质：

$$
|\mathcal{B}(Q_1-Q_2)|_\infty=\gamma |T\max_{a}(Q_1-Q_2)|_\infty\leq \gamma|Q_1-Q_2|_\infty
$$

这里用到了$T$的归一性质，其中$|\cdot|_\infty$代表最大分量（这里只是一个proof sketch，具体的证明细节没有显示）。这一点的作用在于，我们可以得知$\mathcal{B}$的不动点是唯一的，并且任何初始的$Q$都会收敛到这个不动点。

但是事情并非如此简单——我们实际应用中需要进行fitting。考虑到我们的神经网络都只具有有限的representation power，即使不断梯度下降，最终也只能留下有限的误差。因此，用模型拟合这一步相当于做一个向模型的值域空间的**投影**：

$$
Q\leftarrow \Pi Q
$$

这样，整个的过程就成为了 $\Pi \mathcal{B}\Pi \mathcal{B}\cdots Q$。直观上，这个投影很可能会干扰收敛。实际上也确实如此——理论上已经证明，Q learning是不保证收敛的。

同样，用这样的思路，还可以证明fitting value iteration，以至于第六讲介绍的actor-critic中critic的训练都不能收敛。这无疑是很糟糕的消息，但好在人们并不特别在意——反正这些算法跑起来都不错。

### Residual Gradient

一个很奇怪的思路是，考虑我们的loss

$$
L(s,a)=\left(Q^\pi_\phi(s_t,a_t)-r(s_t,a_t)-\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}\left[\max_{a_{t+1}}\text{SG}\left[Q^{\pi}_\phi(s_{t+1},a_{t+1})\right]\right]\right)^2
$$

我们会发现它基本上就是一个linear regression的形式。为什么不收敛呢？一定是因为stop gradient把原先的同时优化变为了交替优化。因此，residual gradient考虑去除stop gradient，试着优化这个新的loss。

理论上听起来很吸引人，因为根据linear regression的理论，它一定能收敛；但实际上，它甚至跑得不如不收敛的Q learning。唉，可能你也发现了：理论上的东西，骗骗哥们得了，别把自己也给骗了。