## Inverse Reinforcement Learning

Given state $s\in\mathcal{S}$, action $a\in\mathcal{A}$, transition model $p(s'|s,a)$ (optional), trajectory $\{\tau_i\}$ sampled from expert policy, learn reward function $r_\psi(s,a)$, then use it to learn $\pi^*(a|s)$

Linear reward function $r_\psi(s,a)=\sum_i\psi_i f_i(s,a)=\psi^T f(s,a)$, where $f_i(s,a)$ is feature function

If feature is important, we want to match expectation: $\mathbb{E}_{\pi^{r_\psi}}[f(s,a)]=\mathbb{E}_{\pi^*}[f(s,a)]$

however this is ambiguous, since different reward function can lead to same policy. One way is use maximum margin

$$
\max_{\psi,m} m\quad\text{ s.t. }\psi^T\mathbb{E}_{\pi^{r_\psi}}[f(s,a)]\geq\psi^T\mathbb{E}_{\pi^*}[f(s,a)]+m
$$

using SVM trick

$$
\min_{\psi,m} \frac{1}{2}||\psi||^2\quad\text{ s.t. }\psi^T\mathbb{E}_{\pi^{r_\psi}}[f(s,a)]\geq\max_{\pi\in\Pi}\psi^T\mathbb{E}_{\pi^*}[f(s,a)]+D(\pi,\pi^*)
$$

for distribution with much difference, we want to use the formula to increase the margin

Issue: no clear model of expert suboptimality, messy constrainted optimization

Learning the optimality variable: use $O_t$ to denote optimality, then $p(\tau|O_{1:T},\psi)\varpropto p(\tau)\exp(\sum_{t=1}^T r_\psi(s_t,a_t))$

$$
\max_{\psi}\frac{1}{N}\sum_{i=1}^N \log p(\tau_i|O_{1:T},\psi)=\max_\psi\frac{1}{N}\sum_{i=1}^N r_\psi(\tau_i)-\log Z
$$

however $Z=\int p(\tau)\exp(r_\psi(\tau))d\tau$ is intractable, but we can use Monte Carlo to estimate it

$$
\nabla_\psi\mathcal{L}=\frac{1}{N}\nabla_\psi r_\psi(\tau_i)-\frac{1}{Z}\int p(\tau)\exp(r_\psi(\tau))\nabla_\psi r_\psi(\tau)d\tau\\

=\mathbb{E}_{\tau\sim\pi^*(\tau)}[\nabla_\psi r_\psi(\tau)]-\mathbb{E}_{\tau\sim p(\tau|O_{1:T},\psi)}[\nabla_\psi r_\psi(\tau)]
$$

The first term estimate with expert samples, the second term is soft optimal policy under current reward

simplify the second term

$$
\mathbb{E}_{\tau\sim p(\tau|O_{1:T},\psi)}[\nabla_\psi r_\psi(\tau)]=\sum_{t=1}^T\mathbb{E}_{(s_t,a_t)\sim p(s_t,a_t|O_{1:T},\psi)}[\nabla_\psi r_\psi(s_t,a_t)]\\

=\sum_{t=1}^T\mathbb{E}_{s_t\sim p(s_t|O_{1:T},\psi), a_t\sim p(a_t|s_t,O_{1:T},\psi)}[\nabla_\psi r_\psi(s_t,a_t)]]
$$

where

$$
p(s_t|O_{1:T},\psi)\varpropto\alpha(s_t)\beta(s_t)\\

p(a_t|s_t,O_{1:T},\psi)\varpropto\pi(a_t|s_t)=\frac{\beta(s_t,a_t)}{\beta(s_t)}
$$

define $\mu_t(s_t)\varpropto\alpha(s_t)\beta(s_t,a_t)$ as a state-action visitation probability, then

$$
\mathbb{E}_{\tau\sim p(\tau|O_{1:T},\psi)}[\nabla_\psi r_\psi(\tau)]=\sum_{t=1}^T\mu_t^T\nabla_\psi r_\psi
$$

MaxEnt IRL algorithm:

1. Given $\psi$, compute backward message $\beta(s_t,a_t)$ and forward message $\alpha(s_t)$
2. Compute $\mu_t(s_t)\varpropto\alpha(s_t)\beta(s_t,a_t)$, then evaluate $\nabla_\psi\mathcal{L}=\frac{1}{N}\sum_{i=1}^N\nabla_\psi r_\psi(\tau_i)-\sum_{t=1}^T\mu_t^T\nabla_\psi r_\psi$
3. Update $\psi\leftarrow\psi+\alpha\nabla_\psi\mathcal{L}$

In the case of $r_\psi=\psi^Tf$, it optimizes $\max_\psi\mathcal{H}(\pi^{r_\psi})$ and $\mathbb{E}_{\pi^{r_\psi}}[f]=\mathbb{E}_{\pi^*}[f]$

Idea: learn $p(a_t|s_t,O_{1:T},\psi)$ using max-ent RL algorithm, then use it to sample $\{\tau_j\}$ using learned policy.

However, it's expensive to compute train the policy each time, we can use lazy policy optimization

$$
\nabla_\psi\mathcal{L}=\frac{1}{N}\sum_{i=1}^N\nabla_\psi r_\psi(\tau_i)-\frac{1}{\sum_j w_j}\sum_{j=1}^M w_j\nabla_\psi r_\psi(\tau_j)
$$

where

$$
w_j=\frac{p(\tau)\exp(r_\psi(\tau_j))}{\pi(\tau_j)}=\frac{p(s_1)\prod_t p(s_{t+1}|s_t,a_t)\exp(r_\psi(s_t,a_t))}{p(s_1)\prod_t p(s_{t+1}|s_t,a_t)\pi(a_t|s_t)}=\prod_t\frac{\exp(r_\psi(s_t,a_t))}{\pi(a_t|s_t)}
$$

$\pi$ denotes current policy, importance sampling allows us to use samples from lazy-optimized policy to estimate the gradient of the optimal policy according to the current reward

In a higher level, it looks like a game, reward optimization makes demos are more likely, sample less likely, policy optimization makes samples harder to distinguish from demos

Reward $r_\psi$, policy $\pi_\theta$, we can use alternating optimization

$$
\nabla_\psi\mathcal{L}=\frac{1}{N}\sum_{i=1}^N\nabla_\psi r_\psi(\tau_i)-\frac{1}{\sum_j w_j}\sum_{j=1}^M w_j\nabla_\psi r_\psi(\tau_j)\\

\nabla_\theta\mathcal{L}=\frac{1}{M}\sum_{j=1}^M\nabla_\theta\log\pi_\theta(\tau_j)r_\psi(\tau_j)
$$

Like GAN with generator $p_\theta$ and discriminator $p_\psi$

$$
\psi=\argmax_\psi\frac{1}{N}\sum_{x\sim p^*}\log p_\psi(x)+\frac{1}{M}\sum_{x\sim p_\theta}\log(1-p_\psi(x))\\

\theta=\argmax_\theta\frac{1}{M}\sum_{x\sim p_\theta}\log p_\psi(x)
$$

best discriminator $p_\psi$ is $p_\psi(x)=\frac{p^*(x)}{p^*(x)+p_\theta(x)}$

We can treat inverse RL as a GAN, optimal policy approach $\pi_\theta(\tau)\varpropto p(\tau)\exp(r_\psi(\tau))$

$$
D_\psi(\tau)=\frac{\frac{1}{Z}p(\tau)\exp(r(\tau))}{p_\theta(\tau)+\frac{1}{Z}p(\tau)\exp(r(\tau))}=\frac{\frac{1}{Z}\exp(r(\tau))}{\prod_t\pi_\theta(a_t|s_t)+\frac{1}{Z}\exp(r(\tau))}
$$

then optimize $\psi$ by

$$
\psi\leftarrow\argmax_\psi\mathbb{E}_{\tau\sim p^*}[\log D_\psi(\tau)]+\mathbb{E}_{\tau\sim\pi_\theta}[\log(1-D_\psi(\tau))]
$$

we don't need importance sampling, since the ratio is already contained in $Z$. But $Z$ is intractable, we can optimize $Z$ w.r.t same objective (see [https://arxiv.org/pdf/1611.03852](https://arxiv.org/pdf/1611.03852))
