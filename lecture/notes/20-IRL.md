# Inverse Reinforcement Learning

在这一讲，我们讨论一个有趣的问题。在之前的问题里，我们一般都是认为reward是已知的，来求解最优的策略；而现在，我们反过来：我们被提供一个**比较优的expert的策略**，希望反过来能够推断出reward函数。这一问题就是**Inverse Reinforcement Learning**(IRL)。

为什么要学习reward函数而不是简单地对专家做imitation learning？这是因为，有些问题中，简单的模仿是不够的，我们需要让model理解expert的**意图**。就像老师让小学生罚站不是为了让他们多站一会而是为了让他们反思一样，我们希望model能理解专家的行为背后的原因，甚至可能在这一意图下作出更好的决策。

> 当然，像让小学生理解罚站的目的这样的比喻可能对于模型而言太困难了。一个比较简单的例子是，generalize to unknown dynamics：专家展示的策略里面，既包含了(已知的)dynamics，也包含了(未知的)reward。然后我们希望model能学会reward。之后，如果到达不同的环境，当agent通过交互感受到dynamics的变化后，就可以很好地generalize，在新的环境作出和原来专家演示类似的行为。相反，如果只是imitation地学会一个policy，就在新的环境里可能会表现得很差。
>
> 一个实验是，用专家演示一个ant环境里转身的操作。用IRL学习reward function之后，在新的环境里训练agent，在这个环境里agent的两个前肢被缩短。结果发现，IRL学习的agent能够很好地适应这个变化，继续作出类似转身的举动。

此外，必须强调，**Learn Reward Function**是一个非常**ill-defined**的问题。主要的原因在于，很多reward会给出同样的最优策略，因此我们无法从最优策略中推断reward。比如，如果仅仅是让专家策略最优，全零的reward就可以满足要求。因此，任何的IRL方法都需要先陈述一个**additional assumption**，才能确保解有意义。

## Feature Matching

我们来介绍我们遇到的第一个IRL方法——Feature Matching。这是最早期的IRL方法之一，因此，人们决定采用线性的方式来近似reward：

$$
r_{\psi}(s,a)=\psi^T f(s,a)
$$

其中，$\psi$是我们要训练的参数，而$f$可以当作某种可以很好地提取特征的函数。

接下来，我们给出一个目标。一个直观的想法是，令

$$
\mathbb{E}_{\pi_{\psi}}[f(s,a)]=\mathbb{E}_{\pi^{\star}}[f(s,a)]
$$

其中，$\pi^\star$代表专家数据给出的策略，而$\pi_{\psi}$代表在reward $r_{\psi}$下的最优策略。这就叫做feature matching——可以看出，这是保证专家reward最大的一个充分条件；另一方面，这也合理，因为如果$f$真的很好地提取了特征，那么两边应该相等。

> 比如，在游戏里，可以认为特征比如有击败敌人，领取金币等等。也许它们对应着不同的reward（即$\psi$给出的系数），但我们希望最优策略和专家策略在**每一个**特征上的表现都是一样的，而非只是总的加权总和一致。

即便作出了这么强的假设，可以论证，现在的这一命题依然是ill-defined的——我们仍然无法确定下$\psi$。为了完全将其确定，我们需要用类似于SVM的方法：最大化margin。也就是说，

$$
\psi = \arg \max_{\psi: |\psi|^2\le 1}\left[\psi^T \mathbb{E}_{\pi^\star}[f(s,a)]-\psi^T \max_{\pi}\mathbb{E}_{\pi}[f(s,a)]\right]
$$

（我们要加一个约束限制住$\psi$的模长）这一目标大概是说，我们希望最大化专家策略，最小化其他策略。

但是，这一方法并不太好——对于任意的策略，他都同等地打压。而专家策略附近做一些小扰动，得到的一定不好吗？因此，应该选择性地打压——让和专家策略差的很远的策略受到惩罚，而和专家策略差不多的策略几乎不受惩罚。

如何刻画这一点呢？一个比较直观的方法是，我们先用SVM duality来重写上面的问题：

$$
\psi: \min |\psi|^2 \text{ such that }  \psi^T \mathbb{E}_{\pi^\star}[f(s,a)]\ge \psi^T \max_{\pi}\mathbb{E}_{\pi}[f(s,a)]+ 1
$$

然后，我们就可以很轻易地加入刻画$\pi$和$\pi^{\star}$接近程度的项：

$$
\psi: \min |\psi|^2 \text{ such that }  \psi^T \mathbb{E}_{\pi^\star}[f(s,a)]\ge \psi^T \max_{\pi}\left(\mathbb{E}_{\pi}[f(s,a)] + D(\pi,\pi^\star)\right)
$$

其中$D$可以选取为任何一种metric。这就是feature matching的基本思想。

当然，feature matching作为一种基本上过时的方法，有着比较严重的缺点：

- maximize margin的方法使得他有点过于“相信”专家了。这有点像generative models的contrastive learning（比如EBM），但问题是，之前generative model我们就是为了学习数据集的分布；但现在我们并不是为了找到一个奇怪的reward来克隆专家的分布，而是希望找到专家的意图。
- 基于同样的原因，我们无法处理专家策略不是最优的情形，或者专家策略有一定随机性的情形（因为专家可能的偶然的非最优行为被contrastive learning强行打上了最优的标签）。当然，你可能说这是因为我们选择了hard-margin的SVM，如果加入一些slack variable，就可以解决这一问题。但是实验上发现，这样的方法并不work。
- 最后，这一目标从实践上也很难被优化。这就比较细节，这里不再讨论。

## Max Entropy IRL

Maximum Entropy IRL是近些年来很有名的IRL方法。在介绍为什么是maximum entropy之前，我们先从上一讲的graphical model出发。回顾一下，我们计算了forward和backward message，并给出在given optimality之后的policy。

一个很好的想法是，我们能不能选取合适的reward 函数（还记得graphical model模型中的reward是如何[定义](./19-soft-optimality.md#graphical-model-of-dicision-making)的吗？），使得这个policy就是我们的专家策略呢？

有了这一目标，接下来的问题还是，我们该如何训练？好处在于，grahical model直接给出了trajectory的概率分布，因此我们可以直接用maximum likelihood来训练我们的reward模型：

$$
\psi = \arg\max \mathbb{E}_{\tau\sim \pi^\star}[\log p(\tau|O_{1..T};\psi)]
$$

其中，$\tau\sim \pi^\star$代表按照$\pi^\star$采集一个trajectory；$O$是我们之前提到的optimality variable，而$\psi$表示这个$p$和$\psi$有关。在上一讲我们便知道，这个概率可以直接计算：

$$
p(\tau|O_{1..T};\psi)\propto p(\tau)\exp \sum_{t}r_\psi(s_t,a_t)
$$

其中

$$
p(\tau)=p(s_1)p(s_2|s_1,a_1)\cdots
$$

是只与环境有关的转移概率。进一步地，我们写

$$
p(\tau|O_{1..T};\psi)=\frac{1}{Z(\psi)}p(\tau)\exp \sum_{t}r_\psi(s_t,a_t)
$$

这样，我们就可以化简MLE objective：

$$
J=\mathbb{E}_{\tau\sim \pi^\star}[\log p(\tau|O_{1..T};\psi)]=\mathbb{E}_{\tau\sim \pi^\star}\left[\sum_{t}r_\psi(s_t,a_t)\right]-\log Z(\psi)
$$

我们发现，这有点像一个[EBM](https://github.com/szjzc2018/dl/blob/master/note2.md#211-energy-based-model)，因此我们知道一个trick是梯度可以写为contrastive learning的形式：

$$
\nabla_{\psi}J=\mathbb{E}_{\tau\sim \pi^\star}\left[\sum_{t}\nabla r_\psi(s_t,a_t)\right]-\frac{1}{Z(\psi)}\int p(\tau)\exp \sum_{t}r_\psi(s_t,a_t)\sum_{t}\nabla r_\psi(s_t,a_t)d\tau
$$

$$
=\mathbb{E}_{\tau\sim \pi^\star}\left[\sum_{t}\nabla r_\psi(s_t,a_t)\right]-\mathbb{E}_{\tau\sim p(\tau|O_{1..T};\psi)}\left[\sum_{t}\nabla r_\psi(s_t,a_t)\right]
$$

那么，如何从$p(\tau|O_{1..T};\psi)$里面采样呢？和EBM不一样的是，这里可以直接采样，因为环境的部分是确定的，我们只需要给出$p(a_t|s_t,O_{1..T};\psi)$即可，而这就是policy $\pi_{\psi}$：

$$
\pi_\psi: \pi_\psi(a_t|s_t,O_{1..T})=\frac{\beta(s_t,a_t)}{\beta(s_t)}
$$

其中$\beta$就是前一讲我们提到的backward message。这样，上面的公式也可以表达为

$$
\nabla_{\psi}J=\mathbb{E}_{\tau\sim \pi^\star}\left[\sum_{t}\nabla r_\psi(s_t,a_t)\right]-\mathbb{E}_{\tau\sim \pi_\psi}\left[\sum_{t}\nabla r_\psi(s_t,a_t)\right]
$$

这个表达式给出了一种选择——我们按照前面的policy表达给出一系列trajectory，然后做contrastive learning。但当然，我们也可以直接地计算梯度：

$$
\nabla_{\psi}J=\mathbb{E}_{\tau\sim \pi^\star}\left[\sum_{t}\nabla r_\psi(s_t,a_t)\right]-\mathbb{E}_{\tau\sim p(\tau|O_{1..T};\psi)}\left[\sum_{t}\nabla r_\psi(s_t,a_t)\right]
$$

$$
=\mathbb{E}_{\tau\sim \pi^\star}\left[\sum_{t}\nabla r_\psi(s_t,a_t)\right]-\sum_t \mathbb{E}_{s_t\sim p(s_t|O_{1..T};\psi),a_t\sim \pi_\psi}\left[\nabla r_\psi(s_t,a_t)\right]
$$

我们发现，上一讲刚好计算过

$$
p(s_t|O_{1..T};\psi) = \alpha(s_t)\beta(s_t)
$$

因此

$$
\nabla_{\psi}J=\mathbb{E}_{\tau\sim \pi^\star}\left[\sum_{t}\nabla r_\psi(s_t,a_t)\right]-\sum_t \sum_{s_t,a_t}\beta(s_t,a_t)\alpha(s_t)\nabla_\psi r_\psi(s_t,a_t)
$$

这也就给出了另外一种选择——我们手动计算这个求和，然后直接计算梯度。

> **Max Entropy IRL** (enumerate version)

1. 在当前的$\psi$下，使用上一讲的方法计算$\beta(s_t,a_t)\alpha(s_t)$；
2. 用上面的公式计算梯度，其中$\pi^\star$的期望就是数据集的平均；
3. 用梯度更新$\psi$。

当然，这个求和的工作量巨大，是$|S|\times|A|\times T$。而且，每一个gradient step，都要重新算一次！如果没钱买不起A100，还有救吗？别急——还记得，刚才我们还给出了一个基于采样和contrastive learning的方法。我们来看一看，这一方法能否减少计算量。

### Using policy to reduce computation

我们需要的实际上只是一个policy $\pi_\psi$就够了。前面其实已经给出了，$\pi_\psi(a_t|s_t)=\frac{\beta(s_t,a_t)}{\beta(s_t)}$。但我们要做就做的极致一些，递归地计算$\beta$还是工作量太大！我们于是就想到，上一讲给出了这个递推的近似解，也就是通过**soft optimality**的方法给出的答案。比如说，

$$
\hat{Q}(s_t,a_t)=r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}}[\hat{V}(s_{t+1})]
$$

$$
\hat{V}(s_{t+1})=\alpha\cdot \mathbb{E}_{a_{t+1}\sim \text{Uniform}}\left[\exp \left(\frac{\hat{Q}(s_{t+1},a_{t+1})}{\alpha}\right)\right]
$$

$$
\pi(a_t|s_t)\propto \frac{\exp \left(\frac{\hat{Q}(s_t,a_t)}{\alpha }\right)}{\exp \left(\frac{\hat{V}(s_t)}{\alpha }\right)}
$$

这样，我们不需要求解递推，只需要跑一下这个类似于Q learning的方法，就可以得到一个近似的policy $\pi_\psi$的近似解。然后，我们就可以用这个policy来采样，然后做contrastive learning。

> 你有可能觉得不对劲——按照前一讲中的说法，上面的公式给出的$\pi$实际上理论值并不是$\frac{\beta(s_t,a_t)}{\beta(s_t)}$？因为原来$\frac{\beta(s_t,a_t)}{\beta(s_t)}$的那个策略被证明有overconfidence的问题，因此我们做了修正，才得到现在这个表达式。但现在，我们似乎就是要$\frac{\beta(s_t,a_t)}{\beta(s_t)}$，并不需要修正，所以用这个表达式并不合适？
>
> 但实际上， ~~我也不知道为什么，咱先别管这个~~

![](./assets/not_implement.png)

但即使是如此，我们还是嫌工作量太大了。不是我们太苛刻——注意我们讨论的全部是每一个gradient step都要做的内容！换句话说，我们训练$\psi$的每一步，都需要做一次完整的SAC的训练！所以工作量确实很大。

如何进一步减轻负担呢？在EBM里，我们其实也遇到了类似的情况。当时，我们提出了contrastive learning之后，也遇到了同样的问题。那时候的问题是，为了给出负例，我们需要初始化，然后做langevin sampling来到达energy model的分布。如果要完全到达energy model的分布，那么需要消耗一个很长的mixing time；但我们在训练的时候实际上只取几步或者几十步sample即可。为什么呢？当时给出的理由是，langevin sample的步数少一些有点类似一个更"fuzzy"的分布，因此在这个分布上选取负例，有助于减少“误伤”（更细节的讨论请参考DL的PPT）。

因此，虽然话说了这么多，但最后做的改动其实很简单——我们每次不要训policy训练的太狠，稍微训练几步其实就可以了。这就给出了下面的方法：

> **Max Entropy IRL** (sampling version)

初始化$\pi_\psi$为随机策略；

1. 在当前的$\psi$下，使用soft optimality的目标训练$\pi_\psi$几步；
2. 用上面contrastive learning的公式，通过从$\pi_\psi$中采样来计算梯度；
3. 用梯度更新$\psi$。

这样，它很好地避免了计算量，同时在实践中也有着很不错的效果。

> Q: 等等，我看你这个流程，好像是$\pi_\psi$先在第一个$\psi$下面训练几步，又在第二个$\psi$下面训练几步，不断这样？不应该每一次都把$\pi_\psi$完全初始化吗？
>
> A: 不，每一次都初始化固然可以，但这个方法相当于是一个优化。在EBM里面，其实也有这种优化，但当时的negative sample有很多张图片，所以它的优化方式是基于replay buffer：langevin sample之后的图片放到replay buffer里面，然后初始化一部分是随机的，一部分从replay buffer里面采集。
>
> 现在，我们这个其实是一个道理——只不过，我们没有能力训练许多policy，因此只能拿一个policy，然后放入replay buffer，立刻再拿出来，这样不断训练。这就回到了上面的方法。

最后，再稍微提一下一个可能的改进：因为我们的采样并非严格遵守$\pi_\psi$，因此期待值的估计略有偏差。为此，我们可以采用importance sampling。

我们有importance weight

$$
w(\tau)=\frac{\frac{1}{p(O_{1..T})}\cdot p(\tau)\exp \sum_t r(s_t,a_t)}{\pi(\tau)}=\frac{1}{p(O_{1..T})}\frac{\exp \sum_t r(s_t,a_t)}{\pi(a_1|s_1)\cdots\pi(a_T|s_T)}
$$

其中，$\pi$是我们前面算法里的那个policy。这里，虽然不知道归一化系数，但有一个比较神奇的方法。我们首先写出

$$
\mathbb{E}_{\tau\sim p(\tau|O_{1..T})}[\nabla r_\psi(s_t,a_t)]=\mathbb{E}_{\tau\sim \pi}[w(\tau)\nabla r_\psi(s_t,a_t)]
$$

然后，我们写

$$
\mathbb{E}_{\tau\sim \pi}[w(\tau)\nabla r_\psi(s_t,a_t)]\approx \frac{1}{N}\sum_{i=1}^N w(\tau_i)\nabla r_\psi(s_t,a_t)
$$

但是$w$不会算啊！别急——我们注意到

$$
\mathbb{E}_{\tau\sim \pi}[w(\tau)]=\int \pi(\tau)\cdot \frac{p(\tau|O_{1..T})}{\pi(\tau)}d\tau=1
$$

所以，对于很大的$N$，其实可以期待

$$
N\approx \sum_{i=1}^N w(\tau_i)
$$

我们据此写

$$
\mathbb{E}_{\tau\sim p(\tau|O_{1..T})}[\nabla r_\psi(s_t,a_t)]\approx \frac{1}{\sum_{i=1}^N w(\tau_i)}\sum_{i=1}^N w(\tau_i)\nabla r_\psi(s_t,a_t)
$$

这样，不同的$w$之间做比值，我们不会算的归一化系数$p(O_{1..T})$就消失了。这样，我们成功地使用了importance sampling。虽然这感觉有点空手套白狼，但可能确实是可行的。

### What about Max Entropy?

等等，你说了这么多，这为什么叫做maximum entropy IRL呢？这里的entropy在哪里呢？

这确实是很诡异的一部分。PPT上论证了，对于$r_\psi(s_t,a_t)=\psi^T f(s_t,a_t)$的情况，上面的contrastive learning objective等价于

$$
\max_\psi \mathcal{H}(\pi_\psi)\text{ such that } \mathbb{E}_{\pi_\psi}[f]=\mathbb{E}_{\pi^\star}[f]
$$

但是我并不会证明这一点……求助，如果愿意捞一下我可以提交一个PR。

![](./assets/not_implement.png)

### Benefits of Max Entropy IRL

最后，我们来看一下Max Entropy IRL的优点。为何他可以解决之前feature matching的问题？

- 首先，即是expert policy不是最优的，这一方法完全没有问题。因为它基于的graphical model本身就是处理sub-optimality的。同样，即使expert policy有一定的随机性，这一方法也可以处理。
- 其次，我们之前提到feature matching的maximize marginal的做法意义不明确；但不这么做会导致reward不唯一（还记得我们一直强调的，IRL是一个ill-defined的问题）。这里，我们引入了何种假设来解决ill-define的问题呢？它又为何有意义呢？
    - 当然，我们是用MLE来训练的，这当然完全是唯一的；
    - 但是另外一方面，MLE的这个假设具有意义。这一意义是前一节的等效表述所赋予的——它说明contrastive learning等价于在保证feature matching的情况下，最大化entropy。
    - 这是什么意思？实际上，就是说：能推断的（feature）尽可能推断；不能推断的不要有偏见（因为有了偏见就没有完全正确理解expert的“意图”），而是让它尽可能随机。这就是Max Entropy IRL的重要意义。

## IRL as a GAN

之前我们说Maximum Entropy IRL有些像EBM；实际上，它在思想上也有一点像GAN（本质上，它们都属于contrastive learning）。我们前面的算法中，不断训练$\pi_\psi$，使得其近似为$\psi$的soft optimal policy；而$\psi$的训练目标又是尽可能区分开$\pi_\psi$和$\pi^\star$。这就有点像GAN的generator和discriminator；刚好，在我们上面的表述中，也是一人训练一个epoch。

一个自然的尝试是，我们就使用GAN的目标，看看这样给出的IRL方法是否比较好。我们建立discriminator，输入trajectory $\tau$，输出其真假性；而训练的正例是$\pi^\star$的trajectory，负例是我们poicy $\pi$的trajectory。这样，我们就可以得到一个IRL的GAN：

$$
J = \min_{\pi}\max_D \mathbb{E}_{\tau\sim \pi^\star}[\log D(\tau)]+\mathbb{E}_{\tau\sim \pi}[\log (1-D(\tau))]
$$

但是，如何构造一个discriminator能从trajectory映射到概率呢？一个可能的方向是sequence modeling的方法，但这里有更好的选择——我们可以数学上算出optimal discriminator：

$$
\frac{p_{\pi^\star}(\tau)}{D(\tau)}-\frac{p_\pi(\tau)}{1-D(\tau)}=0
$$

这给出

$$
D(\tau)=\frac{p_{\pi^\star}(\tau)}{p_{\pi^\star}(\tau)+p_\pi(\tau)} = \frac{\frac{1}{p(O_{1..T})}\exp \sum_t r(s_t,a_t)}{\frac{1}{p(O_{1..T})}\exp \sum_t r(s_t,a_t) + \prod_t \pi(a_t|s_t)}
$$

虽然我们现在不知道$p_{\pi^\star}(\tau)$，也不知道$p(O_{1..T})$。这可以启示我们，**把discriminator的形式设置为这样**。也就是，

$$
D_\psi(\tau;\pi)=\frac{\frac{1}{Z_\psi}\exp \sum_t r_\psi(s_t,a_t)}{\frac{1}{Z_\psi}\exp \sum_t r_\psi(s_t,a_t) + \prod_t \pi(a_t|s_t)}
$$

其中，$r_\psi$是网络，$Z_\psi$是参数。注意这个discriminator某种程度上“依赖于”generator $\pi$。此时，我们就可以写出完整的objective：

$$
J = \min_{\pi}\max_\psi \mathbb{E}_{\tau\sim \pi^\star}[\log D_\psi(\tau;\pi)]+\mathbb{E}_{\tau\sim \pi}[\log (1-D_\psi(\tau;\pi))]
$$

这一目标看起来相比于普通的GAN还要更加混乱。但实践得出的结果是，只要调参做的不错，这一方法就能表现得非常好。

可以说，这个方法和前面的Max Entropy IRL是有一定的联系的。这一方法虽然作为一种GAN，有着更高的训练难度；但这样学习出来的reward也会更加robust，因此质量更高。这就有点像EBM虽然更好训练，但是生成的图片远不如GAN。

### What if we just use the vanilla GAN?

也许有人说，你的方法确实可以work，但我就用一个sequence model作用在$\tau$上面，然后直接训练GAN，不就好了吗？理论上，完全没有问题，并且这个训练肯定会相比于上面的东西更加稳定；但是问题在于，这样我们完全无法recover出reward function（因为我们知道最优的discriminator最后会变成random guess）。因此，这样的方法只能算是imitation learning，而不是IRL。

# Reference Papers

1. [Apprenticeship Learning via Inverse Reinforcement Learning](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)（介绍了IRL的基本思想）
2. [Maximum margin planning](https://martin.zinkevich.org/publications/maximummarginplanning.pdf)
3. [Maximum Entropy Inverse Reinforcement Learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)
4. [A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models](https://arxiv.org/abs/1611.03852)
5. [Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248)