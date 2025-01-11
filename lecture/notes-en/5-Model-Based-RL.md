## Model-based RL

Learn the transition dynamics, then figure out how to choose actions

If we know the transition dynamics, how can we use it to optimize the policy ?

$$
a^*=\arg\max_a J(s,a)\quad s.t.\quad s_{t+1}=f(s_t,a_t)
$$

Discrete case: Monte Carlo Tree Search (MCTS), just the random version of dynamic programming

### Trajectory optimization

we want to find a sequence of actions that maximize the reward, which is equivalent to solve

$$
\min_{u_1,\dots,u_T}\sum_{t=1}^T c(x_t,u_t)\quad s.t.\quad x_{t+1}=f(x_t,u_t)\\

=\min_{u_1,\dots,u_T}c(x_1,u_1)+c(f(x_1,u_1),u_2)+\dots+c(f(f(\dots),\dots),u_T)
$$

Linear case: LQR, suppose

$$
c(x_t,u_t)=\frac{1}{2}\begin{bmatrix}x_t\\u_t\end{bmatrix}^TC_t\begin{bmatrix}x_t\\u_t\end{bmatrix}+\begin{bmatrix}x_t\\ u_t \end{bmatrix}^T c_t\quad C_t=\begin{bmatrix}C_{x_T,x_T} & C_{x_T,u_T}\\ C_{u_T,x_T} & C_{u_T,u_T}\end{bmatrix},c_t=\begin{bmatrix}c_{x_T}\\ c_{u_T}\end{bmatrix}\\

f(x_t,u_t)=F_t\begin{bmatrix}x_t\\ u_t\end{bmatrix}+f_t
$$

if we only want to solve for $u_T$ only, $u_T=-C_{u_T,u_T}^{-1}(C_{u_T,x_T}x_T+c_{u_T})=K_Tx_T+k_T$, which is a linear function of $x_T$, then we can solve $x_T$ by minimizing the following cost function

$$
V(x_T)=const+\frac{1}{2}\begin{bmatrix}x_T\\ K_Tx_T+k_T\end{bmatrix}^TC_T\begin{bmatrix}x_T\\ K_Tx_T+k_T\end{bmatrix}+\begin{bmatrix}x_T\\ K_Tx_T+k_T\end{bmatrix}^Tc_T
$$

repeating the process backwards, we can get the optimal action sequence. We can also approximate a nonlinear system as a linear-quadratic system using Taylor expansion

### Learning dynamics

1. run base policy $\pi_0(a_t|s_t)$, collect data $\mathcal{D}=\{(s,a,s')_i\}$

   repeat $N$ times:
2. learn dynamics model $f_{\theta}(s,a)$ by minimizing $\sum_i||f_{\theta}(s_i,a_i)-s'_i||^2$
3. plan through $f(s,a)$ to choose actions
4. execute the first planned action, observe resulting state $s'$
5. add a batch of $(s,a,s')$ to $\mathcal{D}$

First take uncertain actions, then become more certain through learning. However, this is suboptimal, we only observe $s_1$ in open-loop case, we can use closed-loop control to improve the performance

$$
\pi=\arg\max_{\pi}\mathbb{E}_{\tau\sim p(\tau)}[\sum_t r(s_t,a_t)]
$$

where $p(\tau)=p(s_1)\prod_{t=1}^Tp(s_{t+1}|s_t,a_t)p(a_t|s_t)$, we just backpropagate directly into the policy

1. run base policy $\pi_0(a_t|s_t)$, collect data $\mathcal{D}=\{(s,a,s')_i\}$
2. learn dynamics model $f_{\theta}(s,a)$ by minimizing $\sum_i||f_{\theta}(s_i,a_i)-s'_i||^2$
3. pick $s_i$ from $\mathcal{D}$, use $f_{\theta}(s,a)$ to make short rollouts from them
4. use both real data and simulated data to improve $\pi_{\theta}$ with off-policy RL
5. run $\pi_{\theta}$, collect data $(s,a,s')$ and add to $\mathcal{D}$, go to 2 until convergence

Problem: when optimizing, we neither optimize over the real data nor the simulated data, we optimize over the mixture of them, which is not guaranteed to converge

### Dyna-Style algorithms

online Q-learning that performs model-free RL with a model

1. given state $s$, pick action $a$ using exploration policy
2. observe reward $r$ and next state $s'$, get transition $(s,a,s',r)$
3. update model $\hat{p}(s'|s,a)$ and $\hat{r}(s,a)$ using $(s,a,s',r)$
4. $Q(s,a)\leftarrow Q(s,a)+\alpha\mathbb{E}_{s',r\sim\mathcal{D}}(r+\gamma\max_{a'}Q(s',a')-Q(s,a))$

   repeat $K$ times:
5. sample $(s,a)$ from $\mathcal{B}$, simulate $s'$ using $\hat{p}(s'|s,a)$ and $\hat{r}(s,a)$
6. $Q(s,a)\leftarrow Q(s,a)+\alpha\mathbb{E}_{s'\sim\hat{p}(s'|s,a),r\sim\hat{r}(s,a)}(r+\gamma\max_{a'}Q(s',a')-Q(s,a))$

MBA, MVE, MBPO: similar as Dyna-Style algorithms, but totally use model to generate data for policy optimization

### Successor Representations

Suppose reward is action-dependent (this is reasonable in some way by defining $\mathcal{S'}=\mathcal{S}\times\mathcal{A}$ and suppose reward return a step later), then

$$
V^\pi(s_t)=\sum_{t'=t}^\infty\gamma^{t'-t}\mathbb{E}_{p(s_{t'}|s_t)}[r(s_{t'})]=\sum_s\sum_{t'=t}^\infty\gamma^{t'-t}p(s_{t'}=s|s_t)r(s)
$$

define $\mu_{s_i}^\pi(s_t)=(1-\gamma)\sum_{t'=t}^\infty\gamma^{t'-t}p(s_{t'}=s_i|s_t)$, which is the probability of reaching state $s_i$ from state $s_t$ under policy $\pi$, then

$$
V^\pi(s_t)=\frac{1}{1-\gamma}\sum_s\mu_{s_i}^\pi(s_t)r(s_i)=\frac{1}{1-\gamma}\mu^{\pi}(s_t)^T r
$$

this is called successor representation. Similar as Bellman equation

$$
\mu^{\pi}_{s_i}(s_t)=(1-\gamma)\delta(s_t=s_i)+\gamma\mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t),a_t\sim\pi(a_t|s_t)}[\mu^{\pi}_{s_i}(s_{t+1})]
$$

However, if number of states is much larger, it's hard to compute $\mu^{\pi}$, we can use Successor features: $\psi^{\pi}_j(s_t)=\sum_s\mu^{\pi}_{s}(s_t)\phi_j(s)$, then if $r(s)=\sum_j\phi_j(s)w_j=\phi(s)^T w$

$$
V^\pi(s_t)=\frac{1}{1-\gamma}\mu^{\pi}(s_t)^T r=\frac{1}{1-\gamma}\psi^\pi(s_t)^T w
$$

if number of features is much less than the number of states, learning them is much easier. From definition

$$
\psi_j^{\pi}(s_t)=\phi_j(s_t)+\gamma\mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t),a_t\sim\pi(a_t|s_t)}[\psi_j^{\pi}(s_{t+1})]
$$

also construct a $Q$-function like version:

$$
\psi^{\pi}_j(s_t,a_t)=r(s_t)+\gamma\mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t),a_{t+1}\sim\pi(a_{t+1}|s_{t+1})}[\psi_j^{\pi}(s_{t+1},a_{t+1})]\\

Q^{\pi}(s_t,a_t)=\sum_j\psi_j^{\pi}(s_t,a_t)w_j=\psi^{\pi}(s_t,a_t)^T w
$$

Using successor features to recover many $Q$-functions

1. Train $\psi^{\pi_k}(s_t,a_t)$ for policies $\pi_k$ via Bellman equation
2. Get reward samples $\{s_i,r_i\}$, then solve $\arg\min_w\sum_i||\phi(s_i)^T w-r_i||^2$
3. Recover $Q^{\pi_k}(s_t,a_t)=\psi^{\pi_k}(s_t,a_t)^T w$

### Uncertainty-Aware Neural Net Models:

Two types of uncertainty: aleatoric uncertainty (inherent noise in the system) and epistemic uncertainty (uncertainty in the model)

We can use output entropy to measure aleatoric uncertainty, use multiple models to measure epistemic uncertainty

Bayesian Neural Networks:

For standard neural networks, we optimize

$$
L(D,w)=\sum_{x_i,y_i\in D}||y_i-f(x_i,w)||^2+\lambda\sum_d w_d^2\\

\log p(D,w)=\sum_{x_i,y_i\in D}\log\mathcal{N}(y_i|f(x_i,w),I)+\sum_d\mathcal{N}(w_d|0,\frac{1}{\sqrt{\lambda}})
$$

which overfits the data when $p(D_{\text{data}})$ is different from $p(D_{\text{test}})$. However, Bayesian Inference computes the posterior distribution of $w$ given $D$ from prior distribution $p(w)$

$$
p(w|D)=\frac{p(D|w)p(w)}{p(D)}=\frac{p(D|w)p(w)}{\int p(D|w')p(w')\text{d}w'}
$$

using the posterior distribution, we can compute the predictive distribution

$$
p(\hat{y}(x)|D)=\int p(\hat{y}(x)|w)p(w|D)\text{d}w=\mathbb{E}_{p(w|D)}[p(\hat{y}(x)|w)]
$$

which means sampling networks from $p(w|D)$, we can quantify our uncertainty about these things, e.g., by computing their variance.

The difficulty is computing the integral, we can use Monte Carlo sampling

$$
p(D)=\mathbb{E}_{p(w)}(p(D|w))\approx\frac{1}{N}\sum_{i=1}^Np(D|w_i)\\

p(\hat{y}(x)|d)=\mathbb{E}_{p(w)}[p(\hat{y}(x)|w)\frac{p(d|w)}{p(d)}]\approx\frac{1}{N}\sum_{i=1}^Np(\hat{y}(x)|w_i)\frac{p(d|w_i)}{p(d)}
$$

but since $w$ is high-dimensional, sample with low variance is difficult. We can use Markov Chain Monte Carlo (MCMC) to sample: use the Markov chain to generate candidate samples and then stochastically accept them with probability

$$
a=q(w'|w_t)=\min(1,\frac{p(w',D)}{p(w_t,D)})
$$

Metropolis-Hastings algorithm, we can choose $q(w'|w_t)=\mathcal{N}(w_t,\sigma^2)$

Variational Inference: approximate the posterior distribution with a simpler distribution $q_\phi(w)$, then minimize the KL-divergence between $q_\phi(w)$ and $p(w|D)$

$$
d_{KL}(q_\phi(w)||p(w|D))=\mathbb{E}_{q_\phi(w)}[\log\frac{q_\phi(w)}{p(w|D)}]=\mathbb{E}_{q_\phi(w)}[\log q_\phi(w)-p(w,D)]+\log p(D)
$$

we can minimize the KL-divergence by maximizing the evidence lower bound (ELBO)

$$
\mathcal{L}(\phi)=\mathbb{E}_{q_\phi(w)}[p(w,D)-\log q_\phi(w)]=\mathbb{E}_{x,y\in D}\mathbb{E}_{q_\phi(w)}[\log p(\hat{y}(x)=y|w)+\log p(w)-\log q_\phi(w)]
$$

we pick $\mathcal{N}(\mu,\sigma)$ as $q_\phi(w)$, then we can compute the ELBO

$$
\mathcal{L}(\phi)=\frac{1}{N}\sum_{i=1}^N\nabla_{\mu,\sigma}\mathbb{E}_{w\sim q_{\mu,\sigma}}[\log p(\hat{y}(x_i)=y_i|w)+\log p(w)-\log q_{\mu,\sigma}(w)]\\

=\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^S \nabla_{\mu,\sigma}\log p(\hat{y}(x_i)=y_i|\mu+\sigma\epsilon_j)+\log p(\mu+\sigma\epsilon_j)
$$

after computing $\phi^*$, we compute $p(\hat{y}(x)|D)$ by $\mathbb{E}_{q_{\phi^*}(w)}[p(\hat{y}(x)|w)]$

### Latent Space Models

Observation model: $p(o_t|s_t)$, high-dimensional, not dynamic

dynamics model: $p(s_{t+1}|s_t,a_t)$, low-dimensional, dynamic

reward model: $p(r_t|s_t,a_t)$

To build a latent space model

$$
\max_\phi\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\mathbb{E}_{(s_t,s_{t+1})\sim p(s_t,s_{t+1}|o_{1:T

},a_{1:T})}[\log p_\phi(s_{t+1,i}|s_{t,i},a_{t,i})+\log p_\phi(o_{t,i}|s_{t,i})]
$$

learn approximate posterior $q_\psi(s_{t}|o_{1:t},a_{1:t})$ as encoder ($q_\psi(s_t,s_{t+1}|o_{1:T},a_{1:T})$ is most accurate but hard to learn, $q_\psi(s_t|o_t)$ is simplest but less accurate)

For simplicity, consider single-step encoder $q_\psi(s_t|o_t)$, our goal is

$$
\max_{\phi,\psi}\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\mathbb{E}_{s_t\sim q_\psi(s_t|o_t),s_{t+1}\sim q_\psi(s_{t+1}|o_{t+1})}[\log p_\phi(s_{t+1,i}|s_{t,i},a_{t,i})+\log p_\phi(o_{t,i}|s_{t,i})]
$$

if $q_\psi(s_t|o_t)$ is deterministic, which means $q_\psi(s_t|o_t)=\delta(s_t=g_\psi(o_t))$

$$
\max_{\phi,\psi}\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T[\log p_\phi(g_\psi(o_{t+1,i})|g_\psi(o_{t,i}),a_{t,i})+\log p_\phi(o_{t,i}|g_\psi(o_{t,i}))]
$$

backpropagate to train, we can also add reward model to the objective

$$
\max_{\phi,\psi}\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T[\log p_\phi(g_\psi(o_{t+1,i})|g_\psi(o_{t,i}),a_{t,i})+\log p_\phi(o_{t,i}|g_\psi(o_{t,i}))+\log p_\phi(r_{t,i}|g_\psi(o_{t,i}),a_{t,i})]
$$

learn from $p(o_{t+1}|o_t,a_t)$ directly is also a good choice
