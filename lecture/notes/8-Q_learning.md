# Notice
为了前后的连贯性，本讲的一部分内容（包括Q-learning的改进等）被搬迁到了前面一讲（第七讲）。

# Advanced Q-Learning

前面介绍的DQN看起来已经解决了大部分算法中的不合理之处或可能的问题。实际上，DQN也有极好的应用效果。但是它还有一定的改进空间。

## Double Q-learning

### Over-confidenence of Q values

大量的实验结果表明，我们的模型学习出的Q值往往在多次训练后高于定义值

$$
Q(s_t,a_t)=r(s_t,a_t)+\mathbb{E}_{s_{t+1},a_{t+1},\cdots}\left[\sum_{t'>t}\gamma^{t'-t} r(s_{t'},a_{t'})\right]
$$

这看似很奇怪，但因为它是系统误差而不是随机误差，所以我们必须要给出一个解释。实际上，也相对直接。我们从模型迭代的表达式出发

$$
Q_\phi(s_t,a_t)\leftarrow r(s_t,a_t)+\gamma\max_{a'}Q_\phi(s_{t+1},a')
$$

首先注意到以下的事实：

$$
\mathbb{E}[\max\{x_1,x_2,\cdots,x_n\}]\ge \max\{\mathbb{E}[x_1],\mathbb{E}[x_2],\cdots,\mathbb{E}[x_n]\}
$$

所以，我们可以做如下的想象：假设开始的时候$Q_\phi$值接近于真值，但具有一个随机误差$\epsilon(s,a)$，满足其对$(s,a)$的期望为零：

$$
Q_\phi(s,a)=Q^\star (s,a)+\epsilon(s,a)
$$

那么，我们就会有

$$
\max_{a}Q_\phi(s,a)=\max_{a}Q^\star(s,a)+\bar{\epsilon}(s)
$$

并且 $\mathbb{E}[\bar{\epsilon}(s)]\ge 0$。这样，在下一次迭代的时候，位于等式左边的$Q_\phi(s_t,a_t)$就会有一个期待值非负的误差。这样，反复的迭代，我们的$Q$就会偏离地越来越大。

这一问题实际上是上一次介绍的DQN在实践中的**主要问题**。为了解决它，引入了**Double Q-learning**。

### Double Q-learning

如何解决这个问题呢？一个比较关键的步骤是想清楚为什么 $
\mathbb{E}[\max\{x_1,x_2,\cdots,x_n\}]\ge \max\{\mathbb{E}[x_1],\mathbb{E}[x_2],\cdots,\mathbb{E}[x_n]\}$。我们会发现，实际上是因为左边每一个点取出argmax再evaluate，这就会导致有一个偏大的bias。同样地，应用到我们的场景里，就是

$$
\max_{a}Q(s,a)=Q(s,\arg\max_a Q(s,a))
$$

可以看到第二个表达式出现了两个$Q$；二者实现了类似“正反馈”的关系，导致误差总是偏大。一个自然的想法就是我们引入**两个网络**：

$$
Q^{A}_\phi(s_t,a_t)\leftarrow r(s_t,a_t)+\gamma Q^B_\phi(s_{t+1},\arg\max_a Q^A_\phi(s_{t+1},a))
$$

$$
Q^{B}_\phi(s_t,a_t)\leftarrow r(s_t,a_t)+\gamma Q^A_\phi(s_{t+1},\arg\max_a Q^B_\phi(s_{t+1},a))
$$

这样就避免了“正反馈”的问题。更进一步地，我们发现这个方法甚至可以实现“负反馈”，也就是“自己给自己纠错”：如果对于某一个$a_t$，$Q^A$的计算偏大而$Q^B$的计算相对准确，那么$Q^A$在更新的时候就会被$Q^B$纠正（注意$Q^A$本身的数值在第一个update中并不重要，因为只是求argmax）；另一方面，对于$Q^B$ 的更新而言，其并不会完全受到$Q^A$的影响，因为在第二个update中，取哪个action是$Q^B$决定的，从而可以大概率避开$Q^A$的错误点。这样，我们就避免了过度估计的问题。

当然，在实践上，我们并不会多去训练一个网络（训练两个网络干一件事，听起来就很逆天）。相反，我们刚好利用两组参数$\phi_0$（老的参数）和$\phi$，分别对应$Q^A$和$Q^B$。当然，为了保证稳定性，我们就不对$\phi_0$更新了，从而只做

$$
Q_\phi(s_t,a_t)\leftarrow r(s_t,a_t)+\gamma Q_{\phi_0}(s_{t+1},\arg \max Q_\phi(s_{t+1},a))
$$

（虽然理论上，这个方法不是完整的；但实验上跑的还不错，所以就不管了！）这样，再结合fitting的部分，我们就可以写出
$$
[Q(s_t,a_t)]^{\star}=r(s_t,a_t)+\gamma Q_{\phi_0}(s_{t+1},\arg \max Q_{\textcolor{red}{\phi}}(s_{t+1},a))
$$

$$
\phi\leftarrow \arg\min_{\phi}\left([Q(s_t,a_t)]^{\star}-Q_\phi(s_t,a_t)\right)^2
$$

对比原来的算法，可以看到这里的唯一改动就是，计算argmax并不使用$\phi_0$，而是使用$\phi$。只需要在原来的代码里稍微修改一点点，就立刻会有巨大的提升。Double Q-learning的力量就在这里。

## Multi-step returns

类似policy gradient，我们这里也遇到了采样的variance和引入模型的bias的tradeoff。因此同样，我们可以利用多步的力量，减少我们的模型带来的bias。

$$
[Q(s_t,a_t)]^{\star}=\mathbb{E}_{s_{t+1},a_{t+1},\cdots}\left[\sum_{t'=t}^{t+N-1}\gamma^{t'-t}r(s_{t'},a_{t'})\right]+\gamma^N \max_a Q_{\phi}(s_{t+N},a)
$$

但必须注意，在$N>1$的时候，这个表达式就变成**on-policy**的了！（因为$a_{t+1}$从$s_{t+1}$中采样的方式是由$\pi$决定的）但如果要on-policy地训练，又和之前的replay buffer的优化冲突了。这时也没有什么办法能解决，只能再次摆烂——强行把这个on-policy的算法当off-policy来跑了。实验上，效果也还不错。

## Q learning in continuous action space

和之前的policy gradient不同，Q-learning算法延伸到continous action space的难度略高。这是因为update过程有一个max操作。在连续的空间如何实现这个max呢？有三种方法。

1. optimization：在每一次需要取max的时候，我们用一个optimizer（比如GD或者不使用梯度的stochastic optimization方法等）来选取$a$使得最大化$Q(s,a)$。问题也很明显——这会导致计算太慢了。

2. **NAF(Normalized Advantage Functions)**: 我们选取$Q$函数必须为指定的形式：$Q(s,a)=-\frac{1}{2}(a-\mu_\phi(s))M_\phi(s)(a-\mu_\phi(s))+V_\phi(s)$，其中包含了若干神经网络作为参数。这样计算确实可以很快，但是这个方法的问题也比较明显：$Q$只能是$a$的二次函数！这样的限制太大了。

3. Learn an approximate maximizer: 我们在训练$Q$的同时还训练另外一个网络$\pi_\theta$，使得$\pi_\theta(s)\approx \arg\max_a Q(s,a)$。训练方法也很直接：对$\theta$最大化$Q(s,\pi_\theta(a))$即可。可以发现这个方法实际上是最好的。同时，我们还能看到，我们似乎又再一次引入了policy $\pi$！这一方法也被称为“deterministic actor critic method”。

使用上面的方法3,人们得到了著名的**DDPG Algorithm**：

> **DDPG Algorithm**

重复：
1. 从环境中根据某种policy采样一个$(s,a,s',r)$，加入replay buffer $B$；
2. 从replay buffer取出一个batch(相当于$K=1$)，计算目标 $[Q(s,a)]^\star=r(s,a)+\gamma Q_{\phi_0}(s',\pi_{\theta_0}(s'))$；
3. 对 $L(\phi)=\sum_{(s,a)}([Q(s,a)]^\star-Q_\phi(s,a))^2$作一步梯度下降；
4. 对 $L(\theta)=-\sum_s Q_\phi(s,\pi_\theta(s))$做一步梯度下降；
5. 更新$\phi_0,\theta_0$，可以使用隔$N$次更新一次的方法，也可以使用Polyak平均的方法。

而类似地，我们还可以学习一个不是deterministic的maximizer。这就是 **SAC(Soft Actor Critic)** 算法。这个算法的核心思想是，我们不再学习一个确定的$\pi$，而是学习一个概率分布$p(a|s)$，使得$Q(s,a)$的期望最大化。我们会发现，此时的问题完全变成了之前的policy gradient问题！只不过这次我们的reward是$Q$值而不是advantage。这样，我们就可以使用policy gradient的updage公式来解决这个问题。这里出于简略，不再详细介绍。可以参考[这篇论文](https://arxiv.org/abs/1801.01290)。

# Implementing Q learning

我们已经知道，理论上Q-learning是不收敛的。因此，参数的调整必须十分地小心。有以下的tip：
- Large replay buffer make training stable：这相当于训练的数据集很大，可以更好地收敛；
- Gradually reduce exploration：开始的时候应该多explore，以防止陷入一个局部最优解；但最后摸清了环境的套路后，应该专注在把最好的$Q$值做的更精确上。
- **Huber loss** for training Q network：`huber_loss`是$\frac{x^2}{2}$的“截断”版本，也就是当$x$很大的时候loss是线性而非二次的。
    - 为什么这样做有效？还是上一讲的那个例子：如果一个batch里面有一些Q为1,2,3的数据，还有一个Q为-1000000的数据，那么参数肯定会被后者的巨大梯度拽到使得那个很差的Q最精确的地方。但我们实际上并不关心那个很差的Q到底是-1000000还是-500000，因此这个截断版本的loss很有效。
- **Always** use Double Q learning: it has no downsides
- Sometimes use multi-step returns (but it has a theoretical error)