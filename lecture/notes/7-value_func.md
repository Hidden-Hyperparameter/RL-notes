# Value Function Methods

作为引入，我们考虑能否不给模型指定一个明确的policy，而是直接选取当前最优的action。这样的方法称为value function methods。

具体地，假设我们知道了使用策略 $\pi$ 时，当前state $s_t$ 下面各个action $a_t$ 对应的value $Q^\pi(s_t,a_t)$ ，那么我们可以直接选取最优的action：

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

然后，把 $\pi$ 消去，就得到了**value iteration**算法：

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

注意， $[V^\pi(s_t)]^\star$ 中的 $\star$ 代表这个数是网络的训练目标。注意计算的时候右边 $r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}_\phi(s_{t+1})]$ 也和 $\phi$ 相关，因此需要**stop-gradient**。

同时，也应该注意，第二步并不是一个gradient step，而是很多步，因为这一步代表着让 $V^\pi_\phi$ 接近于之前“动态规划”那样方法的准确值，神经网络的学习必须跟上value function update的进度。而1-2作为一整轮，才相当于之前“动态规划”方法中update $V$ 的方式。

## Fitting Q Iteration

我们很容易发现前面算法的重要问题：第一步必须涉及对不同的 $a_t$ 获得很多reward和 $s_{t+1}$ ，并计算最大值。这样的操作导致其必须和环境进行反复交互。能否解决这个问题？

可以发现，如果反过来拟合 Q function，就不存在这样的问题了！因为，回顾我们之前的三步

1. $V^\pi(s_t)\leftarrow Q^\pi(s_t,\pi(s_t))$
2. $Q^\pi(s_t,a_t)\leftarrow r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}(s_{t+1})]$
3. $\pi(s_t)\leftarrow \arg\max_{a_t} Q^\pi(s_t,a_t)$

此时如果不是消去Q而是消去V，那么就会得到

$$
Q^\pi(s_t,a_t)\leftarrow r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}\left[\max_{a_{t+1}}Q^{\pi}(s_{t+1},a_{t+1})\right]
$$

这样的更新关系。注意到现在最大值的位置跑到了**内部**！这样我们不需要和环境多次交互，只需要多跑几次神经网络就可以了。换句话说，**相比于前面的V方法，我们用 $Q(s_t,a_t)$ 能够generalize到未见过的 $a_t$ 上面作为赌注，换得了采更少样本的机会**。

这个方法就是著名的**Q Iteration Algorithm**。当然，代价也是明显的：我们的网络需要输入 $s_t,a_t$ 两个参数，因此拟合的难度也比较高。但考虑到DL的巨大发展已经为人们扫平了大部分的障碍，这个方法完全瑕不掩瑜。

最后，我们也很容易把它推广到Fitted Q Iteration，在此作一总结：

> **Fitted Q Iteration Algorithm**

重复：
1. 收集数据集，包含一系列的 $\{s_t,a_t,s_{t+1},r(s_t,a_t)\}$ （有时候，为了方便，也会记为 $\{s,a,s',r\}$ ，需要搞清楚它们的对应关系）；
2. 重复 $K$ 次：

2.1. 计算新的 $Q$ ：

$$
[Q^\pi(s_t,a_t)]^\star=r(s_t,a_t)+\gamma \max_{a_{t+1}}Q^{\pi}_\phi(s_{t+1},a_{t+1})
$$

2.2. 最小化拟合误差：（注意，这一步本身也可能包含很多次gradient step，总数记为 $S$ ）

$$
\phi \leftarrow \arg\min_\phi \sum_{s_t,a_t}\left(Q^\pi_\phi(s_t,a_t)-[Q^\pi(s_t,a_t)]^\star\right)^2
$$

和前面的fitted value iteration一样，应该注意 $K$ 和 $S$ 的区别： $K$ 的迭代作为外循环，代表着之前“动态规划”方法中update $Q$ 的方式，而 $S$ 的迭代作为内循环，是保证神经网络能够跟上Q function update的进度。

注意到这个算法的另外一个性质是，我们无论怎样更新 $\pi$ ，我们原先用来update $Q$ 的数据仍然可以**重复利用**。这样的性质允许了上面“重复 $K$ 轮”的操作，也使得这个算法非常适合于off-policy的任务。当然，也必须提及另外一个极端——如果我们让 $K\to \infty$ ，也就是一直只在很少的样本上面重复训练，就会产生问题：我们并不了解没有采样到的地方是否会有更好的策略。因此，和这个算法匹配的必定是一些能够实现**exploration**的算法。

当然，实验给出的最优方式就是像上面的算法描述的那样：对同样的一组数据训练 $K$ 轮，然后作exploration，再重新根据某种policy $\pi$ 来收集数据，再训练 $K$ 轮，如此循环。

## Q learning: Introduction

Q learning是Q iteration的online版本。具体地，我们每一次和环境交互一次，立刻用上面Q iteration的方法处理得到的数据，并根据一个策略继续交互。

> **Vanilla Q Learning Algorithm**

重复：
1. 从环境中根据某种policy采样一个 $(s,a,s',r)$ ；
2. 计算**一个** $[Q^\pi(s,a)]^\star=r(s,a)+\gamma\max_{a'}Q^{\pi}_\phi(s',a')$
3. 对 $L=([Q^\pi(s,a)]^\star-Q^{\pi}_\phi(s,a))^2$ 作**一步**梯度下降。

其中第二步的“一个”是为了保证它是一个online的算法。而第三步之所以只作一个gradient step，是因为我们的训练数据只有一个（也就是刚采样的那个数据）。

另外值得一提的是第一步采样的policy。就如之前所说，因为off-policy的性质，这里采样的“某种policy”并不一定是当前 $Q$ 值分布下的最佳policy；但从另一个角度来看，直觉上这个policy多少应该贴近现在的最优policy，这样有利于模型“加强训练”。但除此之外，我们必须注意policy需要有一个exploration的机制，否则我们可能会陷入一个很差的解。

这一切都使得**Q learning中的exploration**成为一个非常重要的问题。我们将在之后的某讲讨论这个问题。但在现在，可以根据intuition给出一些介绍：

- **$\epsilon$ -greedy**：以 $1-\epsilon$ 的概率选择最优的action，以 $\epsilon$ 的概率随机选择其他的一个action；
- **Boltzmann**： $\pi(a|s)=\frac{1}{Z}e^{Q(s,a)/\tau}$ 。注意这个方法相比于 $\epsilon$ -greedy的合理性：
    - 如果有两个Q很接近的action，那么它们被选中的概率应该接近；
    - 如果有两个都不是最优，但从Q上能明确分清主次的action，那么它们被选中的概率应该也有一定差距。

# Further on Q learning

*Side Note.* 这里的内容实际上是从第八讲搬迁过来的，使得内容更连贯。

## Issues

如果你仔细观察上面vanilla Q learning的算法，那么很大概率你会觉得这个算法有着巨大的不合理之处。

首先，Q learning是每一次训练的数据都是和环境交互的最新结果（换句话说，在同一条trajectory上面）。这就造成问题了：如果考虑Q learning的连续几步gradient step，那么就会发现，它的训练数据是**相关**的。这会导致Q learning的收敛性变得很难保证。

不仅如此，每一步的gradient step都伴随着一个value的update，这就使得它的训练目标是“移动的”，这也会导致收敛性的问题。

举一个不恰当的例子：比如我们训练MNIST的分类模型，上来给你一张图片label是1；接下来，只让你做一步gradient step，然后立刻把这张图片（比如说）加一点噪声或者扭转一下（强相关性），然后把label改成2。这样要是能收敛，那就是奇迹了。

因此，这个简单的版本理论上存在很大的缺陷。我们需要想办法让它work。

## Avoid Correlation

回顾一下第六讲的Online actor-critic algorithm，我们也看到了类似的问题：当时是我们每一次采集一组数据，然后update value function approximater 和 policy function。

> 回顾：Online actor-critic algorithm
> 1. 用当前的策略 $\pi_\theta$ 走一步，记为 $\{s_t,a_t,s_{t+1},r=r(s_t,a_t)\}$ ；
> 2. 用一步的数据 $\{V_{\phi}^{\pi_\theta}(s_t),V_{\phi}^{\pi_\theta}(s_{t+1}),r\}$ 训练 $V^{\pi_\theta}_{\phi}$ ；
> 3. 计算 $A^{\pi_\theta}(s_t,a_t)=\gamma V_\phi^{\pi_\theta}(s_{t+1})-V_\phi^{\pi_\theta}(s_{t})+r(s_t,a_t)$
> 4. 计算 $\nabla_\theta J(\theta)=\nabla_{\theta}\log \pi_\theta(a_t|s_t)A^{\pi_\theta}(s_t,a_t)$
> 5. 用这个梯度更新 $\theta$

我们当时就提出了两种方法：第一种是进行并行，用多个线程进行交互，然后把数据收集到一起一起update；第二种是用一个replay buffer，把数据存起来，然后每次从buffer里面sample一些数据进行update。

对于并行的版本，和之前一样，主要的难点来自于并行的同步问题，这里不再详细介绍。我们接下来主要考虑replay buffer的方法。注意现在的情况比之前policy gradient好很多，因为现在的算法自动是off-policy的。因此，我们无需做任何修改。

简单地概括加入replay buffer后的算法如下：

> **Q learning with replay buffer**

重复：

1. 从环境中根据某种policy采样一个 $\{s,a,s',r\}$ ，并加入buffer $B$ ；
2. 重复 $K$ 次：
    1. 从buffer $B$ 中sample**一个batch**的数据 $\{s,a,s',r\}$ ；
    2. 在这个batch上计算目标Q value，并梯度下降**一次**。

这里注意我们还是只对这个batch下降一次，就像普通的DL任务里，每组batch也只下降一次一样。这样可以避免模型在一个batch上面过拟合。同时，不仅如此，我们还可以避免模型把精力浪费在我们不关心的Q values上面（比如说，若干Q value分别是1,2,3，但有一个是-10000000。那么我们并不希望模型费劲地学会那个-10000000，因为就算它有点误差，变成了-5000000，也不会对我们的任务产生太大影响）。

这样，我们就解决了correlation的问题。

## Fixing our target

我们刚才还提到第二个问题：拟合过程

$$
\arg\min_\phi \left(r(s,a)+\gamma\max_{a'}Q^{\pi}_\phi(s',a')-Q^{\pi}_\phi(s,a)\right)^2
$$

实际上是一个“打移动的靶子”的问题：你每一次update之后，目标里面因为带有 $\phi$ ，所以也会变化。

这个问题实际上很好解决，因为我们原来的fitted Q iteration是没有这个问题的，还记得我们当时提到了

> “应该注意 $K$ 和 $S$ 的区别： $K$ 的迭代作为外循环，代表着之前“动态规划”方法中update $Q$ 的方式，而 $S$ 的迭代作为内循环，是保证神经网络能够跟上Q function update的进度。”

也就是说，原先的gradient step是为了让神经网络能够跟上Q function的update进度，因此目标是确定的（根据上一轮的参数决定的，而在梯度下降的过程中保持不变）。现在我们只需要把这个思路搬到Q learning上面就可以了。

这一思想也被称为**Target Network**。具体地，我们直接修改目标为

$$
\arg\min_\phi \left(r(s,a)+\gamma\max_{a'}Q^{\pi}_{\phi_0}(s',a')-Q^{\pi}_\phi(s,a)\right)^2
$$

其中 $\phi_0$ 代表着旧的网络参数。这样，梯度下降的过程也就成为了线性回归的梯度下降，我们必然可以保证它收敛。当收敛了（或者经过了一定的步数），我们再把 $\phi_0$ 更新为 $\phi$ ，然后继续训练。

## Summary: General Q learning Algorithm

我们可以把上面的讨论总结为一个通用的Q learning算法；不仅如此，这一算法如此的general，以至于还可以包含前面介绍的fitted Q iteration算法。

- **process 1**: 从环境中根据某种policy采样一个 $\{s,a,s',r\}$ ，并加入buffer $B$ 。记为`InteractWithEnv()`；
- **process 2**: 根据replay buffer中的数据和网络 $Q_{\phi_0}$ 来计算若干Q value的target。其中 $\phi_0$ 代表某时刻的网络参数，不一定是最新的。记为`GetTargetFromBuffer()`；
- **process 3**: 根据target进行regression，更新 $\phi$ 。记为`GradStep()`；
- **process 4**: 更新process 2使用的参数： $\phi_0\leftarrow \phi$ 。直接记为`phi_0=phi`

我们其实可以发现，这三个过程实际上是基本独立的，而包含了前面各种Q-function based method的核心部分。比如说，fitted Q iteration可以叙述为：

> **Fitted Q Iteration In new language**

- Buffer size 比较小，每一次采样都会完全更新buffer；
- 伪代码如下：

```python
while True:
    InteractWithEnv()
    for _ in range(K):
        GetTargetFromBuffer()
        for _ in range(S):
            GradStep()
        phi_0=phi
```

我们前面最简单的Q learning算法也可以叙述为：
- Buffer size 为1；
- 伪代码如下：

```python
while True:
    InteractWithEnv()
    GetTargetFromBuffer()
    GradStep()
    phi_0=phi
```


最后，我们可以把前面的种种改进结合起来，得到著名的**DQN**算法，用这样的语言描述：

> **DQN**

- buffer size可以自由设置；
- 伪代码如下：

```python
while True:
    for _ in range(N):
        InteractWithEnv()
        for _ in range(K):
            GetTargetFromBuffer()
            GradStep()
    phi_0=phi
```

我们可以直观地理解这个算法。用DL的语言来叙述，就是：每一个 $\phi_0$ 对应着**一个“regression problem”**；对于每一个这样的problem，我们训练 $N$ 个epoch；对于每一个epoch，我们从buffer里面取出 $K$ 个batch，就像从trainloader取出数据那样；最后，interact with env相当于我们的数据集在不断地更新。

```python
while True:
    problem = NewRegressionProblem() # a regression problem based on phi_0 and the replay buffer
    for epoch in range(N):
        for x,y in problem.trainloader: # trainloader has length K
            optimizer.zero_grad()
            loss = calc_loss()
            loss.backward()
            optimizer.step()
        problem.add_data_from_env()
```

容易看出， $K$ 越大这个回归的过程越稳定。但实际上，我们一般取 $K=1$ ，这样使得算法更加online一些。

另外，容易看到， $K=1$ 的时候这个算法比起前面最开始提出的Q learning with replay buffer来的优势就是 $\phi_0$ 的update慢了 $N$ 倍。这样的做法才保证了我们的网络能够跟上Q function的update进度。

### Modification: Alternative Target network

在前面的general algorithm的基础上，还有一个可能的修改需要被介绍。我们发现，大部分时候 $\phi_0$ 都是老的数值，但某一次突然变成了新的数值。换句话说，target长期保持不变，但偶然突然变化一下。这样的突变可能影响训练的效果。

为了解决这一问题，提出了一种方法，可以使得参数逐渐的变化，同时依然避免moving target。类似于Polyak averaging，我们可以用一个滑动平均来代替 $\phi_0$ ：

$$
\phi_0 \leftarrow \alpha \phi_0 + (1-\alpha)\phi
$$

其中 $\alpha\approx 0.999$ 。这样的操作也基本避免了moving target：target在移动，但速度十分的缓慢。相比之下，原来的target是长时间不变，但是突然变化。

这个修改在实验中可能起到一定的作用，但理论上Q-learning的最关键的提升还是前面的DQN算法。

# Theoretical Analysis

## Optimal Policy

我们把Q iteration的loss写出来：

$$
L=\mathbb{E}_{(s,a)}\left[\left(Q^\pi(s_t,a_t)-r(s_t,a_t)-\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}\left[\max_{a_{t+1}}Q^{\pi}(s_{t+1},a_{t+1})\right]\right)^2\right]
$$

（这里先忽略了fitting的loss）我们就可以注意到理论最优的策略 $\pi_\star$ 必然满足 $L=0$ 。这给了我们很重要的希望：理论上，我们可以通过训练一个Q function来找到最优的policy。

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

其中 $T$ 是第四讲提到的transition matrix，把 $Q$ 的 $s_{t+1}$ 维度transform到 $(s_t,a_t)$ 两个维度。关键在于， $\mathcal{B}$ 这个operator具有一定的性质：

$$
|\mathcal{B}(Q_1-Q_2)|_\infty=\gamma |T\max_{a}(Q_1-Q_2)|_\infty\leq \gamma|Q_1-Q_2|_\infty
$$

这里用到了 $T$ 的归一性质，其中 $|\cdot|_\infty$ 代表最大分量（这里只是一个proof sketch，具体的证明细节没有显示）。这一点的作用在于，我们可以得知 $\mathcal{B}$ 的不动点是唯一的，并且任何初始的 $Q$ 都会收敛到这个不动点。

但是事情并非如此简单——我们实际应用中需要进行fitting。考虑到我们的神经网络都只具有有限的representation power，即使不断梯度下降，最终也只能留下有限的误差。因此，用模型拟合这一步相当于做一个向模型的值域空间的**投影**：

$$
Q\leftarrow \Pi Q
$$

这样，整个的过程就成为了 $\Pi \mathcal{B}\Pi \mathcal{B}\cdots Q$ 。直观上，这个投影很可能会干扰收敛。实际上也确实如此——理论上已经证明，Q learning是不保证收敛的。

同样，用这样的思路，还可以证明fitting value iteration，以至于第六讲介绍的actor-critic中critic的训练都不能收敛。这无疑是很糟糕的消息，但好在人们并不特别在意——反正这些算法跑起来都不错。

### Residual Gradient

一个很奇怪的思路是，考虑我们的loss

$$
L(s,a)=\left(Q^\pi_\phi(s_t,a_t)-r(s_t,a_t)-\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}\left[\max_{a_{t+1}}\text{SG}\left[Q^{\pi}_\phi(s_{t+1},a_{t+1})\right]\right]\right)^2
$$

我们会发现它基本上就是一个linear regression的形式。为什么不收敛呢？一定是因为stop gradient把原先的同时优化变为了交替优化。因此，residual gradient考虑**去除stop gradient**，直接试着优化这个新的loss。

理论上听起来很吸引人，因为根据linear regression的理论，它一定能收敛；但实际上，它甚至跑得不如不收敛的Q learning。唉，可能你也发现了：理论上的东西，骗骗哥们得了，别把自己也给骗了。

# Reference Papers

1. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)（十分著名的对Q learning或者actor critic的改进，强烈建议了解）
2. [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)（DQN for games）
3. [Deep reinforcement learning with double Q-learning](https://arxiv.org/abs/1509.06461)（Double Q-learning）