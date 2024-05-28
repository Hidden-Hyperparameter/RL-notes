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

我们可以用Q function简化记号，两步可以分别写为
1. $V^\pi(s_t)\leftarrow Q(s_t,\pi(s_t))$
2. $\pi(s_t)=\arg\max_{a_t}$


直接将这里的$\pi$带入到第一步，也就是

$$
V^\pi(s_t)\leftarrow r(s_t,\pi(s_t))+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,\pi(s_t))}[V^{\pi}(s_{t+1})]
$$




进而给出**value iteration**算法：

> **value iteration**

重复:
1. 计算新的value：
$$
V^\pi(s_t)=\max_{a_t}Q^\pi(s_t,a_t)
$$

2. 计算新的Q-function：
$$
Q^\pi(s_t,a_t)=r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}(s_{t+1})]
$$