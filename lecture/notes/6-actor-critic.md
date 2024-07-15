# Improve Policy Gradient

## Introduce Q,V into Policy Gradient

我们重新考虑之前的考虑causality的 policy gradient的表达式：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim p_{\bar{\theta}}(\tau)}\left[\sum_{t=1}^T\left(\nabla_{\theta}\log \pi_\theta(a_t|s_t)\sum_{t'=t}^Tr(s_{t'},a_{t'})\right)\right]=\mathbb{E}_{\tau \sim p_{\bar{\theta}}(\tau)}\left[\sum_{t=1}^T\nabla_{\theta}\log \pi_\theta(a_t|s_t)\hat{Q}^{\pi_\theta}_{n,t}
\right]
$$

但这个表达式具有比较大的variance。仔细一想，是因为对于每一次 $\hat{Q}^{\pi_\theta}_{n,t}$ 我们只计算一条轨迹，这会导致variance很大。请记住本讲的takeover：

> **重要思想.** 对于很多问题，有两种方案：
> 1. 一个单采样的估计，虽然unbiased但方差大；
> 2. 一个模型的拟合，虽然对不准确的model有一定bias，但方差小。
> 
> 我们需要在这两者之间做一个tradeoff。

本讲很多地方都要采用这种思想。比如这里，根据这个**重要思想**，我们希望把 $\hat{Q}^{\pi_\theta}_{n,t}$ 换成很多轨迹的期待值。

那这个期待值是什么？我们发现，其实就是

$$
Q^{\pi_\theta}(s_t,a_t)=\mathbb{E}_{\tau_{\ge t} \sim p_{\bar{\theta}}(\tau_{\ge t}|s_t,a_t)}\left[\sum_{t'=t}^Tr(s_{t'},a_{t'})\right]=\mathbb{E}_{\tau_{\ge t} \sim p_{\bar{\theta}}(\tau_{\ge t}|s_t,a_t)}\left[\hat{Q}^{\pi_\theta}_{n,t}\right]
$$

所以一个更好的（variance更小）的表达式是

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=1}^T\nabla_{\theta}\log \pi_\theta(a_t|s_t)Q^{\pi_\theta}(s_t,a_t)\right]
$$

### With Baseline

为了进一步减小variance，我们可以引入一个**baseline**。它的选取应该是 $Q^{\pi_\theta}(s_t,a_t)$ 的某种平均，我们会发现相比于之前最普通的直接平均，我们可以有

$$
V^{\pi_\theta}(s_t)=\mathbb{E}_{a_t\sim \pi_\theta(a_t|s_t)}\left[Q^{\pi_\theta}(s_t,a_t)\right]
$$

这样，就有了一个新的表达式

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=1}^T\nabla_{\theta}\log \pi_\theta(a_t|s_t)\left(Q^{\pi_\theta}(s_t,a_t)-V^{\pi_\theta}(s_t)\right)\right]
$$

这里的 $Q^{\pi_\theta}(s_t,a_t)-V^{\pi_\theta}(s_t)$ 也被称作Advantage Function（记作 $A^{\pi_\theta}(s_t,a_t)$ ）：它代表做action $a_t$ 相比于平均情况的优势。

## Fitting Q and V

显然，既然引入了Q和V，我们就需要一个方法来估计 $Q$ 和 $V$ 。假设我们的算力只支持拟合一个网络，我们的神经网络拟合哪个比较好呢？

我们注意到

$$
Q^{\pi_\theta}(s_t,a_t)=r(s_t,a_t)+\mathbb{E}_{\tau_{> t} \sim p_{\bar{\theta}}(\tau_{> t}|s_t,a_t)}\left[\sum_{t'=t+1}^Tr(s_{t'},a_{t'})\right]
=r(s_t,a_t)+\mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}\left[V^{\pi_\theta}(s_{t+1})\right]
$$

根据之前提到的**重要思想**：期待值虽然带来小的variance，但是需要多付出一个model（用来拟合 $Q$ ）的代价。因此，这里我们不如随机选取一个 $s_{t+1}$ ，然后直接用

$$
Q^{\pi_\theta}(s_t,a_t)\approx r(s_t,a_t)+V^{\pi_\theta}(s_{t+1})
$$

显然这样的估计虽然unbiased但是variance很大。但这样我们完全可以不用再拟合 $Q$ ，所以作了很好的tradeoff。接下来，只需要拟合 $V$ 了。

### Policy Evaluation

policy evaluation就是指拟合 $V$ 。问题的关键是如何收集用来拟合 $V$ 的数据。最简单的思路就是Monte Carlo：我们就利用每一条轨迹的reward来拟合 $V$ 。具体地，每次采样 $N$ 条轨迹，计算

$$
y_{n,t}=\sum_{t'=t}^Tr(s_{n,t'},a_{n,t'})
$$

然后训练

$$
L(\phi)=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T}\left(y_{n,t}-V^{\pi_\theta}_{\phi}(s_{n,t})\right)^2
$$

注意这里的 $V^{\pi_\theta}_{\phi}$ 是一个只依赖于参数 $\phi$ ，而不依赖于 $\theta$ 的网络。

当然，容易看到这里 $y_{n,t}$ 又是一个单采样，所以variance很大。因此，根据前面的**重要思想**，我们可以利用模型的结果：

$$
{y}_{n,t}=r(s_{n,t},a_{n,t})+\sum_{t'=t+1}^Tr(s_{n,t'},a_{n,t'})\approx r(s_{n,t},a_{n,t})+V^{\pi_\theta}_{\phi}(s_{n,t+1}):=\hat{y}_{n,t}
$$

这样，我们的目标变成了

$$
\hat{L}(\phi)=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T}\left(\text{SG}[\hat{y}_{n,t}]-V^{\pi_\theta}_{\phi}(s_{n,t})\right)^2=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T}\left(r(s_{n,t},a_{n,t})+\text{SG}[V^{\pi_\theta}_{\phi}(s_{n,t+1})]-V^{\pi_\theta}_{\phi}(s_{n,t})\right)^2
$$

（注意这里stop gradient，以稳定训练）这个最终的表达式就是我们的policy evaluation的目标。

## Summary

简单总结一下，我们究竟作了哪些事情：

- 引入Q,V来代替policy gradient中的reward求和，减小variance；
- 通过近似的方式，避免了对Q的拟合（虽然这步会增大variance）；
- 通过巧妙设计拟合V的目标，再次减小variance。

# Actor-Critic Algorithm

把前面的方法总结一下，我们就得到了Actor-Critic算法：

1. 利用现在的policy $\pi_\theta$ 取 $N$ 个trajectory；
2. 用这些数据训练 $V^{\pi_\theta}_{\phi}$ ；
3. 计算 $A^{\pi_\theta}(s_t,a_t)=V_\phi^{\pi_\theta}(s_{t+1})-V_\phi^{\pi_\theta}(s_{t})+r(s_t,a_t)$
4. 计算 $\nabla_\theta J(\theta)=\frac{1}{N}\sum_{n}\left[\sum_{t=1}^T\nabla_{\theta}\log \pi_\theta(a_t|s_t)A^{\pi_\theta}(s_t,a_t)\right]$
5. 用这个梯度更新 $\theta$

第二步就是我们前面讨论的policy evaluation，可以采用前面的两种方法中任何一种。

## Discount Factor

如果简单按照前面的方法训练，我们理想情况的 $V^{\pi_\theta}_\phi$ 应该就是

$$
V^{\pi_\theta}_\phi(s_t)=\mathbb{E}_{\tau_{\ge t} \sim p_{\bar{\theta}}(\tau_{\ge t}|s_t)}\left[\sum_{t'=t}^Tr(s_{t'},a_{t'})\right]
$$

但这个求和对于infinite horizon的情况是发散的！因此我们引入一个discount factor $\gamma\in [0,1)$ ，重新定义我们的目标

$$
V^{\pi_\theta}_\phi(s_t):=\mathbb{E}_{\tau_{\ge t} \sim p_{\bar{\theta}}(\tau_{\ge t}|s_t)}\left[\sum_{t'=t}^T\gamma^{t'-t}r(s_{t'},a_{t'})\right]
$$

这会带来很多地方的改变。

> **小贴士**
> 
> 本讲有一些地方完全改变了原来的算法。为了保证理解，建议每一次这种地方都自己先思考一下引入的新修改会导致原来算法哪些部分的改变。比如这里：

- $V^{\pi_\theta}_\phi$ 的训练要改变
- $A^{\pi_\theta}(s_t,a_t)$ 的计算要改变

### $V^{\pi_\theta}_\phi$ 的训练

具体地，我们

$$
{y}_{n,t}=r(s_{n,t},a_{n,t})+\sum_{t'=t+1}^T\gamma^{t'-t}r(s_{n,t'},a_{n,t'})\approx r(s_{n,t},a_{n,t})+\gamma V^{\pi_\theta}_{\phi}(s_{n,t+1}):=\hat{y}_{n,t}
$$

因此要修改训练方式：

$$
L(\phi)=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T}\left(\hat{y}_{n,t}-V^{\pi_\theta}_{\phi}(s_{n,t})\right)^2=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T}\left(r(s_{n,t},a_{n,t})+\gamma V^{\pi_\theta}_{\phi}(s_{n,t+1})-V^{\pi_\theta}_{\phi}(s_{n,t})\right)^2
$$

### $A^{\pi_\theta}(s_t,a_t)$ 的计算

$$
Q^{\pi_\theta}(s_t,a_t)=r(s_t,a_t)+\gamma\mathbb{E}_{\tau_{> t} \sim p_{\bar{\theta}}(\tau_{> t}|s_t,a_t)}\left[\sum_{t'=t+1}^Tr(s_{t'},a_{t'})\right]
=r(s_t,a_t)+\gamma\mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}\left[V^{\pi_\theta}_\phi(s_{t+1})\right]
$$

这样

$$
A^{\pi_\theta}(s_t,a_t)=Q^{\pi_\theta}(s_t,a_t)-V^{\pi_\theta}_\phi(s_t)=r(s_t,a_t)+\gamma V^{\pi_\theta}_\phi(s_{t+1})-V^{\pi_\theta}_\phi(s_t)
$$

## Two Kinds of Discount Factor (Optional)

**Warning.** 如果已经感觉有点晕了，千万别看这一节。

我们回顾第一次引入discount factor的表达式

$$
V^{\pi_\theta}_\phi(s_t):=\mathbb{E}_{\tau_{\ge t} \sim p_{\bar{\theta}}(\tau_{\ge t}|s_t)}\left[\sum_{t'=t}^T\gamma^{t'-t}r(s_{t'},a_{t'})\right]
$$

稍作变形，可以发现这对应着普通policy gradient中，我们这样加入 $\gamma$ ：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=1}^T\nabla_{\theta}\log \pi_\theta(a_t|s_t)\left(\sum_{t'=t}^T\gamma^{t'-t}r(s_{t'},a_{t'})\right)\right]
$$

那当然也会有另外一种可能的加入方式：

$$
\nabla_\theta J'(\theta)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=1}^T\nabla_{\theta}\log \pi_\theta(a_t|s_t)\left(\sum_{t'=t}^T\gamma^{t'-1}r(s_{t'},a_{t'})\right)\right]
$$

这种加撇号的方式（称为**方式2**）也有其道理：我们相当于直接把reward记作

$$
R=\sum_{t=1}^T\gamma^{t-1}r(s_t,a_t)
$$

而原先的**方式1**甚至写不出一个像这样的，通用的reward形式！这很多时候会造成误解，认为方式2是正确的。但实际上完全不是：

**在大部分问题中，discount factor的正确加入方式是方式1。**

我们可以来简单论述一下为什么。这需要理解两种方式表达的意义。

方式2的形式实际上更加直观：这可以直接理解为MDP加入一个**dead state**：每一次有 $1-\gamma$ 的概率死掉，死掉之后只能保持在dead state，然后reward全部为0。这样，我们就可以理解 $R$ 的形式；同时，我们可以写出对应的value function：

$$
V^{\pi_\theta}_\phi(s_t)=\mathbb{E}_{\tau_{\ge t} \sim p_{\bar{\theta}}(\tau_{\ge t}|s_t)}\left[\sum_{t'=t}^T\gamma^{t'-1}r(s_{t'},a_{t'})\right]=\gamma^{t-1}\mathbb{E}_{\tau_{\ge t} \sim p_{\bar{\theta}}(\tau_{\ge t}|s_t)}\left[\sum_{t'=t}^T\gamma^{t'-t}r(s_{t'},a_{t'})\right]
$$

可以看出，即使后面的步骤（ $t$ 很大）作出了比较好的决策，由于 $\gamma$ 的存在，这个value function也会变得很小。在这种设定下，也是合理的，因为死亡的威胁逼迫模型更早地作出好的决策。

而方式1则代表着，**discount factor并不真正存在于MDP中**，而只是value function定义的一部分。还句话说，它**只是一个数学工具，没有太多实际上的意义**；甚至可以理解为我们为了避免发散乘上一个 $\gamma$ ，但理论上最后应该取 $\gamma\to 1$ 。

在这样的设定下，每一步的地位都是等同的；但当你站在 $t$ 步估计之后的value的时候，我们略微减少更靠后的步骤的权重。

当然，两种方式究竟哪种正确，必须具体问题具体分析。但大部分问题中并没有所谓的dead state，并且应该具有时间平移不变性。因此，在这些情况下，方式1都是正确的。

# Actor-Critic in Practice

加入discount factor之后，我们就得到了最基本的batch actor-critic算法。

> **Actor-Critic Algorithm**
1. 利用现在的policy $\pi_\theta$ 取 $N$ 个trajectory；
2. 用这些数据训练 $V^{\pi_\theta}_{\phi}$ ；
3. 计算 $A^{\pi_\theta}(s_t,a_t)=\gamma V_\phi^{\pi_\theta}(s_{t+1})-V_\phi^{\pi_\theta}(s_{t})+r(s_t,a_t)$ ；
4. 计算 $\nabla_\theta J(\theta)=\frac{1}{N}\sum_{n}\left[\sum_{t=1}^T\nabla_{\theta}\log \pi_\theta(a_t|s_t)A^{\pi_\theta}(s_t,a_t)\right]$ ；
5. 用这个梯度更新 $\theta$ 。

接下来我们介绍一些改进。

## Online Actor-Critic

我们之前的算法是batch的，即每次都要收集 $N$ 条轨迹。能否把这个算法变成一个online的算法呢？注意的，普通的policy gradient是绝对不能变成online的，因为它必须计算 $t'\ge t$ 的reward求和。但是现在我们的算法是基于Q,V的，所以我们可以尝试把算法变为

1. 用当前的策略 $\pi_\theta$ 走一步，记为 $\{s_t,a_t,s_{t+1},r=r(s_t,a_t)\}$ ；
2. 用一步的数据 $\{V_{\phi}^{\pi_\theta}(s_t),V_{\phi}^{\pi_\theta}(s_{t+1}),r\}$ 训练 $V^{\pi_\theta}_{\phi}$ ；
3. 计算 $A^{\pi_\theta}(s_t,a_t)=\gamma V_\phi^{\pi_\theta}(s_{t+1})-V_\phi^{\pi_\theta}(s_{t})+r(s_t,a_t)$
4. 计算 $\nabla_\theta J(\theta)=\nabla_{\theta}\log \pi_\theta(a_t|s_t)A^{\pi_\theta}(s_t,a_t)$
5. 用这个梯度更新 $\theta$

但我们很快发现这多了很多的单采样，variance会很大。解决方法是并行地进行很多个trajectory，然后用这些trajectory的平均来更新。具体实现方法不再介绍。

如果不能并行，那么我们还有另外一种方法，就是下面的**Off-policy Actor-Critic**。

## Off-policy Actor-Critic

采用**Replay Buffer**的方法也有助于解决前面variance较大的问题。具体地，我们虽然每一次只和环境交互一次，但这并不代表我们只能用这一次的数据来训练。我们可以把这些数据存储在一个Replay Buffer里面，然后每次从这个Buffer里面随机取一些数据来训练。

因为buffer里面的数据并非全部来自当前policy，所以这个方法也是off-policy的。我们称这种方法为**Off-policy Actor-Critic**。但是引入replay buffer肯定要涉及算法的修改。

> 现在，请你想一想，有哪些地方需要修改？

### 第2步的训练

我们本来打算

$$
L(\phi)=\sum_{\text{batch}}\left(r(s_{t},a_{t})+\gamma V^{\pi_\theta}_{\phi}(s_{t+1})-V^{\pi_\theta}_{\phi}(s_{t})\right)^2
$$

但是注意到现在的 $a_t$ 不再是从 $\pi_\theta$ 中采样得到的（而是从replay buffer里面随机找出来的）。这就导致

$$
\mathbb{E}_{a_t\sim \textcolor{red}{\pi_{\text{old}}},s_{t+1}\sim p(s_{t+1}|s_t,a_t)}\left[r(s_t,a_t)+\gamma \hat{V}^{\pi_\theta}(s_{t+1})\right]\ne \hat{V} ^{\pi_\theta}(s_t)
$$

（其中 $\hat{V}$ 代表真实的value function），也就是上面的训练目标即使在理想情况也不应该是0。

为了解决这一问题，有一个巧妙的方法：我们**不用 $V$ ，而是用 $Q$ 来训练**！类似地，我们可以写出一个 $Q$ 的目标，并且这时候是对的：

$$
\mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t),a_{t+1}\sim \pi_\theta}\left[r(s_t,a_t)+\gamma \hat{Q}^{\pi_\theta}(s_{t+1},a_{t+1})\right]= \hat{Q} ^{\pi_\theta}(s_t,a_t)
$$

因此，第2步的训练方法可以修改为：

> 用这些数据训练 $Q^{\pi_\theta}_{\phi}$ ，具体地，最小化

$$
L_{\text{new}}(\phi)=\sum_{\text{batch}}\left(r(s_{t},a_{t})+\gamma Q^{\pi_\theta}_{\phi}(s_{t+1},a_{t+1})-Q^{\pi_\theta}_{\phi}(s_{t},a_t)\right)^2
$$

> 其中， $a_{t+1}$ 从 $\pi_\theta$ 中采样。

### 第3步的计算

首先，我们还需要计算advantage function。理论上，因为现在有的是 $Q$ 而非 $V$ ，我们需要计算

$$
A^{\pi_\theta}(s_t,a_t)=Q^{\pi_\theta}_\phi(s_t,a_t)-V^{\pi_\theta}(s_t)=Q^{\pi_\theta}_\phi(s_t,a_t)-\mathbb{E}_{a_t\sim \pi_\theta}\left[Q^{\pi_\theta}_\phi(s_t,a_t)\right]
$$

但估计这个平均值稍微有些昂贵：假设我们选取的样本不够多，很可能对于每一个 $(s_t,a_t)$ 得到的baseline不一致，这会造成很大的bias；但如果选取太多的样本，每一个过一次网络的时间就会很长。

因此，一个摆烂的方法是用干脆不要 $V$ 了，我们直接选

$$
A^{\pi_\theta}(s_t,a_t)=Q^{\pi_\theta}_\phi(s_t,a_t)
$$

这样虽然没有baseline减小的variance，但是我们可以“质量换数量”，用更多的样本来减小variance。

### 第4步的训练

我们原来的表达式是

$$
\nabla_\theta J(\theta)=\sum_{\text{batch}}\nabla_{\theta}\log \pi_\theta(a_t|s_t)A^{\pi_\theta}(s_t,a_t)
$$

乍一看，这个表达式依然成立。但是实际上完全不是：我们需要回到原先policy gradient的推导，就会发现，我们必须保证 $(s_t,a_t)$ 对是从 $\pi_\theta$ 的trajectory中采样才是对的。因此，现在有两个问题：

- $s_t$ 的分布（相当于很多不同 $\pi_{\text{old}}$ 在第 $t$ 步的分布的叠加）和 $\pi_\theta$ 在第 $t$ 步的的分布不一致；
- $a_t|s_t$ 这一分布并不是 $\pi_\theta$

后者很明显很好解决：我们只需要从 $\pi_\theta$ 里面重新采样一个 $\tilde{a_t}$ 就可以。但是前者就比较麻烦。我们这里选取再次摆烂的方法：我们直接**不管**这个问题。

> 抛开一些细节不谈，这个还是合理的：我们相当于训练我们的策略要在不管如何的初始分布 $p_{\pi}(s_t)$ 下都能做的很好，这相当于一个更难的要求，但是是sufficient的。我们相信我们的模型能完成这个更难的任务😊

### Summary

最终，我们给出了off-policy的actor-critic实现方式。

> **Off-Policy Actor-Critic Algorithm**

1. 利用现在的policy $\pi_\theta$ 走一步，得到 $\{s_t,a_t,s_{t+1},r(s_t,a_t)\}$ ，存入Replay Buffer；
2. 从Replay Buffer中随机取一个batch，用这些数据训练 $Q^{\pi_\theta}_{\phi}$ （见前面的目标函数）。这个过程中，对每一组数据我们需要采样一个新的action $a_{t+1}$ ；
3. 取 $A^{\pi_\theta}(s_t,a_t)=Q^{\pi_\theta}_\phi(s_t,a_t)$ ；
4. 计算 $\nabla_\theta J(\theta)=\sum_{\text{batch}}\nabla_{\theta}\log \pi_\theta(\tilde{a_t}|s_t)A^{\pi_\theta}(s_t,\tilde{a_t})$ ，其中 $\tilde{a_t}$ 是从 $\pi_\theta$ 中采样得到的；
5. 用这个梯度更新 $\theta$ 。

## Implementation Details

无论是前面的batch actor-critic还是off-policy actor-critic，都和实践中采用的版本相差甚远。更高级的处理思路会在后面介绍（或者不会被介绍），这里我们只介绍一个有趣的小细节——Net Architecture。

可以看到，之前的算法中，我们需要训练一个 $V$ （或 $Q$ ）网络和一个 $\pi$ 网络。这两个网络都输入 $s_t$ 。自然地，我们会想到，能不能把这两个网络合并成一个网络呢？这显然是好的，因为比如 $s_t$ 是一张图片，那么提取信息的卷积核就可以被共享。最后，只需要加入两个不同的projection head就可以了。这称为**Shared Network Design**。

当然，容易看出，这两个网络的公共部分会吃来自两个不同来源的梯度。这会使得训练不稳定，也使得hyperparameter的选择变得更加困难。实际情况中，我们需要具体问题具体分析。

# State-dependent Baselines

回顾本讲的开头，我们还记得policy gradient的loss表达式为

$$
J_{\text{PG}}(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T\nabla_{\theta}\log \pi_\theta(a_t|s_t)\left(\sum_{t'=t}^T \gamma^{t'-t} r(s_{t'},a_{t'})\right)\right]
$$

我们当时作出的第一步是减少variance，因此用模型预测的期待值 $Q^{\pi_\theta}_\phi(s_t,a_t)$ 取代单采样的 $\sum_{t'=t}^T \gamma^{t'-t} r(s_{t'},a_{t'})$ 。但是回顾我们很早之前提到的**重要思想**，这二者之间存在tradeoff：这样引入了模型不准确造成的bias。

一个折中的方法是，我们依然使用模型，但这次模型**只是作为baseline**出现：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)\left[\left(\sum_{t'=t}^T \gamma^{t'-t} r(s_{t'},a_{t'})\right)-V^{\pi_\theta}_\phi(s_t)\right]\right]
$$

相比于第五讲介绍policy gradient的时候简单的平均baseline，这里模型产生的的baseline显然强大多了；同时，因为模型作为baseline（本身理论上不会影响期望，只是减少方差）出现，其对模型的bias要求也小的多。这个巧妙的设计被称为**State-dependent Baselines**。

## Control Variates

除了依赖state之外，我们还可以进一步有Control Variates：它指的是**同时依赖action和state**的baseline。完全类似地，我们可以有

$$
\nabla_\theta J_1(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)\left[\left(\sum_{t'=t}^T \gamma^{t'-t} r(s_{t'},a_{t'})\right)-Q^{\pi_\theta}_\phi(s_t,a_t)\right]\right]
$$

但这里略有一个小问题：根据之前policy gradient的推导，我们只能增加一个常数，而不能增加一个和 $a_t$ 相关的函数。因此，真正的loss应该是

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)\left[\left(\sum_{t'=t}^T \gamma^{t'-t} r(s_{t'},a_{t'})\right)-Q^{\pi_\theta}_\phi(s_t,a_t)\right]\right]+\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)Q^{\pi_\theta}_\phi(s_t,a_t)\right]
$$

引入之前 $\hat{Q}^{\pi_\theta}_{t}$ 的简记，我们可以写出

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)\left(\hat{Q}^{\pi_\theta}_{t}-Q^{\pi_\theta}_\phi(s_t,a_t)\right)\right]+\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)Q^{\pi_\theta}_\phi(s_t,a_t)\right]
$$

直到这里，你也许认为这只是没有意义的恒等变形——第一项减去一项，第二项又加上一项。但关键之处在于，第一项包含期待值是0的 $\hat{Q}^{\pi_\theta}_{t}-Q^{\pi_\theta}_\phi(s_t,a_t)$ ，因此应该相比第二项会小很多；但关键是第二项不再和环境有关系——这样，我们可以想取多少sample选取多少sample！

因此，我们可以修改 $\nabla_\theta J(\theta)$ 的计算：
- **取少量的sample计算出第一项**，虽然方差较大但是因为这一项本身相对第二项很小，影响不大；
- **取大量的sample计算第二项**，因为这一项的计算相比于 $\hat{Q}^{\pi_\theta}_{t}$ 而言只需要 $(s_t,a_t)$ ，不需要后面的轨迹，所以可以取很多sample同时保持计算的高效性。这就是Control Variates的核心思想。

# Hybrid Methods

在这最后的一部分，我们考虑把policy gradient和actor-critic结合起来。

## N-step returns

我们对比一下policy gradient和actor-critic的loss：

$$
\nabla_\theta J_{\text{PG}}(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)\left(\sum_{t'=t}^T \gamma^{t'-t} r(s_{t'},a_{t'})\right)\right]
$$

$$
\nabla_\theta J_{\text{AC}}(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)\left(r(s_t,a_t)+\gamma V_\phi^{\pi_\theta}(s_{t+1})-V^{\pi_\theta}_\phi(s_t)\right)\right]
$$

那么，很自然的一个想法就是取

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)\left[\left(\sum_{t'=t}^{t+n-1} \gamma^{t'-t} r(s_{t'},a_{t'})\right)+\gamma^n V_\phi^{\pi_\theta}(s_{t+n})-V^{\pi_\theta}_\phi(s_t)\right]\right]
$$

这个方法被称为N-step returns。它为什么合理呢？我们看到，对于较近的决策（ $t'\approx t$ ），即使是单次采样方差也不会太大，并且避免了bias；而对于远处的决策，我们使用模型的预测结果以避免采样。

> Q: 直观上，为什么较近的决策方差小？
>
> A: 不妨带入我们的人生。如果考虑你明天会做什么，多少可以通过你现在的计划或者他人（环境）的事情安排决定下来；但倘若问到二十年后你会做什么，那又有谁真正能确定呢？

## Generalized Advantage Estimation

Generalized Advantage Estimation方法在前面的N-steps return的基础上再进一步。如果之前我们给出的advantage function是

$$
A^{\pi_\theta}_n(s_t,a_t)=\left(\sum_{t'=t}^{t+n-1} \gamma^{t'-t} r(s_{t'},a_{t'})\right)+\gamma^n V_\phi^{\pi_\theta}(s_{t+n})-V^{\pi_\theta}_\phi(s_t)
$$

那么GAE给出的advantage function是

$$
A_{\text{GAE}}^{\pi_\theta}(s_t,a_t)=\sum_{n=1}^\infty \lambda^{n-1} A^{\pi_\theta}_n(s_t,a_t)
$$

这样的exponential decay的方法可以看作是对N-step return的一个平滑。也可以展开，会发现

$$
A_{\text{GAE}}^{\pi_\theta}(s_t,a_t)=\sum_{t'\ge t}(\gamma\lambda)^{t'-t}\left[r(s_{t'},a_{t'})+\gamma V^{\pi_\theta}_\phi(s_{t'+1})-V^{\pi_\theta}_\phi(s_{t'})\right]
$$

因此，我们也可以把 $\gamma\lambda$ 整体看作参数：每一项 $r(s_{t'},a_{t'})+\gamma V^{\pi_\theta}_\phi(s_{t'+1})-V^{\pi_\theta}_\phi(s_{t'})$ 代表着这一步是否优秀；而 $\gamma\lambda$ 代表着我们对每一步作出重要性如何随着时间而衰减。

最后，值得一提：GAE是一个比较general的算法，之前的两种方法都可以被视为GAE的特例。具体地，当 $\lambda=1$ 时，可以消去中间项发现GAE advantage成为了[state-independent baseline](#state-dependent-baselines)的形式；而当 $\lambda=0$ 时，GAE advantage成为了[vanilla actor-critic](#actor-critic-in-practice)的形式。

# Reference Papers

1. [Bias in Natural Actor-Critic Algorithms](https://proceedings.mlr.press/v32/thomas14.html)（讨论discount的问题）
2. [Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](https://arxiv.org/abs/1611.02247)（一个control variates的例子）
3. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)（GAE算法）
4. [Asynchronous methods for deep reinforcement learning](https://arxiv.org/abs/1602.01783)（**A3C**算法，一种并行的actor-critic算法）