## Exploration

Regret: $\text{Reg}(T)=TE(r(a^*))-\sum_{t=1}^T r(a_t)$, where $a^*$ is the optimal action

### Optimistic Exploration

use upper confidence bound (UCB) to choose actions

$$
a_t=\arg\max_a\left(Q(s_t,a)+c\sqrt{\frac{\log t}{N(s_t,a)}}\right)
$$

where $N(s_t,a)$ is the number of times action $a$ has been selected in state $s_t$. The intuition is that we should try actions that we are uncertain about or have high $Q$-values.

Trouble with counts is in high-dimensional continuous action spaces, we never visit the same state-action pair twice.

Idea: fit density model $p_\theta(s)$, when observing a new state $s_i$, fit new model $p_{\theta'}(s)$ using $\mathcal{D}\cup\{s_i\}$, then we get pseudo-counts

$$
p_{\theta}(s_i)=\frac{\hat{N}(s_i)}{\hat{n}}\quad p_{\theta'}(s_i)=\frac{\hat{N}(s_i)+1}{\hat{n}+1}\\

\hat{N}(s_i)=\frac{1-p_{\theta'}(s_i)}{p_\theta'(s_i)-p_\theta(s_i)}p_\theta(s_i)
$$

another way: counting with hashes: compress $s$ into a $k$-bit code via $\phi(s)$, use $N(\phi(s))$ to count the number of times we have visited state $s$. Improve by learning a compression

### Posterior Sampling

assume the reward is drawn from a distribution like

$$
f(x;\alpha,\beta)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}
$$

update $\alpha$ and $\beta$ using the reward, $\alpha_k\leftarrow\alpha_k+r_t,\beta_k\leftarrow\beta_k+1-r_t$ if action $k$ is selected

1. sample $Q$ from $p(Q)$
2. act according to $Q$ for one episode
3. update $p(Q)$

how to represent a distribution over functions? Sample from dataset, train ensemble of models, then choose randomly. To avoid train $N$ models, we can share the model except the last layer

### Information Gain ?

define $\mathcal{H}(\hat{p}(z))$ be the current entropy of $z$ estimate, $\mathcal{H}(\hat{p}(z|y))$ be the entropy of $z$ after observing $y$, then the information gain is

$$
IG(z,y|a)=\mathbb{E}_y[\mathcal{H}(\hat{p}(z))-\mathcal{H}(\hat{p}(z|y))|a]
$$

which defines how much we learn about $z$ from taking action $a$ after observing $y$. Also, define expected suboptimality of $a$ as $\delta(a)=\mathbb{E}[r(a^*)-r(a)]$, choose $a$ according to

$$
\arg\min_a\frac{\delta^2(a)}{IG(z,y|a)}
$$

how to pick $y$ ? We can use variational inference:

IG can be written as $D_{KL}(p(z|y)||p(z))$, we want to learn about transitions $p_\theta(s_{t+1}|s_t,a_t)$, so pick $z=\theta,y=h,a=(s_t,a_t,s_{t+1})$, where $h$ is history of all prior transitions

$$
IG=D_{KL}(p(\theta|h,s_t,a_t,s_{t+1})||p(\theta|h))
$$

Intuition: a transition is informative if it changes our beliefs about the transition model. Use variational inference to approximate the posterior distribution $q(\theta|\phi)\approx p(\theta|h)$, e.g. a product of independent Gaussian parameter distributions

$$
\min_{\phi} D_{KL}(q(\theta|\phi)||p(h|\theta)p(\theta))
$$

give new transition $(s_t,a_t,s_{t+1})$, update $\phi$ to get $\phi'$, then

$$
IG=D_{KL}(q(\theta|\phi')||q(\theta|\phi))
$$

### Learn without Rewards

Define mutual information

$$
\mathcal{I}(x;y)=D_{KL}(p(x,y)||p(x)p(y))=\mathcal{H}(p(y))-\mathcal{H}(p(y|x))
$$

define $z$ as vectors in the latent space, $x$ as the current state, we can learn without any rewards

1. Propose goal: $z_g\sim p(z),x_g\sim p_\theta(x_g|z_g)$
2. Attemept to reach goal using $\pi(a|x,x_g)$, reach $\hat{x}$
3. use data to update $\pi$, $p_\theta(x_g|z_g),q_\phi(z_g|x_g)$

To get diverse goals, update $p_\theta,q_\phi$ using

$$
\theta,\phi=\arg\max_{\theta,\phi}\mathbb{E}(w(\hat{x})\log p(\hat{x}))
$$

where $w(\hat{x})=p_\theta(\hat{x})^\alpha$, the key result is for any $\alpha\in[-1,0)$, $\mathcal{H}(p_\theta(x))$ increases

Update $\pi$: we want to train $\pi(a|S,G)$ to reach goal $G$. As $\pi$ gets better, final state $S$ gets close to $G$, which means $p(G|S)$ becomes more deterministic, $\mathcal{H}(p(G|S))$ decreases. To have a good exploration, we need to increase $\mathcal{H}(p(G))$, the objective is

$$
\max\mathcal{H}(p(G))-\mathcal{H}(p(G|S))=\max\mathcal{I}(S;G)
$$

### Random Network Distillation

Three possible factors cause next-state prediction error:

1. Prediction error is high where the predictor fails to generalize from previously seen examples. Novel experience then corresponds to high prediction error.
2. Prediction target is stochastic.
3. Necessary for the prediction is missing, or the model class of predictors is too limited to fit the complexity of the target function.

To explore, we want to emphasize on the first factor. Use a random network $f_\theta^*(s')$ to predict the next state, then trains another network $\hat{f}_\phi(s')$ to predict the next state

$$
\phi^*=\arg\min_\phi\mathbb{E}_{s,a,s'\sim\mathcal{D}}||\hat{f}_\phi(s)-f_\theta^*(s)||
$$

for states we already seen, $\hat{f}_\phi(s)$ is close to $f_\theta^*(s)$, for states we haven't seen, $||\hat{f}_\phi(s)-f_\theta^*(s)||$ is expected to be high

Add a bonus to the reward function $\tilde{r}(s)=r(s)+\lambda||\hat{f}_\phi(s)-f_\theta^*(s)||^2$, then train $\pi$ to maximize $E_{\pi}(\tilde{r}(s))$ using $Q$-learning

### Others

Intrinsic motivation: add a reward bonus to the reward function $\tilde{r}(s,a)=r(s,a)-\log p_\pi(s)$.

Repeating update $\pi(a|s)$ to maximize $E_{\pi}(\tilde{r}(s,a))$, then update $p_\pi(s)$ to fit the new state distribution

State Marginal Matching: Suppose we want to learn $\pi(a|s)$ so as to minimize $D_{KL}(p_\pi(s)||p^*(s))$. In iteration $k$

1. Learn $\pi^k(a|s)$ to maximize $\mathbb{E}_{\pi^k}(\tilde{r}(s,a))$, where $\tilde{r}(s,a)=\log p^*(s)-\log p_\pi(s)$
2. update $p_{\pi^k}(s)$ to fit all states seen so far

Finally, return $\pi^*(a|s)=\sum_k\pi^k(a|s)$, which is a Nash equilibrium of game between $\pi$ and $p_\pi(s)$

Learning diversity skills: different skills should visit different state-space regions, we want to learn $\pi(a|s,z)$ where $z$ is task.

$$
\pi(a|s,z)=\arg\max_\pi\sum_z\mathbb{E}_{s\sim\pi(s|z)}(r(s,z))
$$

where $r(s,z)=\log p_\phi(z|s)-\log p(z)$, $p_\phi$ is a discriminator that tries to predict the task $z$ from state $s$. Iteratively update $\pi$ and $p_\phi$. It's equivalent to learn a policy that maximizes the mutual information $\mathcal{I}(s;z)$
