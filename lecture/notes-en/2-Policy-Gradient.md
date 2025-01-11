## Policy gradient

learning objective

$$
J(\theta)=\mathbb{E}_{(s,a)\sim p_{\theta}(s,a)}[r(s,a)]=\int p_{\theta}(\tau)r(\tau)d\tau\\

\nabla_{\theta}J(\theta)=\int\nabla_{\theta}p_{\theta}(\tau)r(\tau)d\tau=\mathbb{E}_{\tau\sim p_{\theta}}[\nabla_{\theta}\log p_{\theta}(\tau)r(\tau)]\\

=\mathbb{E}_{\tau\sim p_{\theta}}[\left(\sum_{t=1}^T\nabla_{\theta}\log\pi_{\theta}(a_t|s_t)\right)\left(\sum_{t=1}^T r(s_{t},a_{t})\right)]
$$

compute gradients by sampling

To reduce variance, subtract a baseline from reward (don't affect the expectation because $\mathbb{E}_{\tau\sim p_{\theta}}[\nabla_{\theta}\log\pi_{\theta}(a_t|s_t)]=0$), which is expected reward

$$
b=\frac{\mathbb{E}_{\tau\sim p_{\theta}}[\left(\sum_{t=1}^T r(s_{t},a_{t})\right)\left(\sum_{t=1}^T\nabla_{\theta}\log\pi_{\theta}(a_t|s_t)\right)^2]}{\mathbb{E}_{\tau\sim p_{\theta}}[\left(\sum_{t=1}^T\nabla_{\theta}\log\pi_{\theta}(a_t|s_t)\right)^2]}
$$

However, the baseline is hard to compute, most of time we don't use such way

Causality: policy at time $t'$ can't affect reward at time $t<t'$, $\log\pi_{\theta}(a_t|s_t)$ shouldn't be affected by $r(s_{t'},a_{t'})$ for $t'<t$

$$
\nabla_{\theta}J(\theta)=\mathbb{E}_{\tau\sim p_{\theta}}[\left(\sum_{t=1}^T\nabla_{\theta}\log\pi_{\theta}(a_t|s_t)\left(\sum_{t'=t}^T r(s_{t'},a_{t'})\right)\right)]
$$

this is the same as the original equation because

$$
\mathbb{E}_{\tau\sim p_{\theta}}[\sum_{t=1}^T\nabla_{\theta}\log\pi_{\theta}(a_t|s_t)\left(\sum_{t'<t}r(s_{t'},a_{t'})\right)]=0
$$

we only use causality to reduce variance

Off-policy with importance sampling:

$$
\nabla_{\theta'}J(\theta')=\mathbb{E}_{\tau\sim p_{\theta}}[\frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)}\left(\sum_{t=1}^T\nabla_{\theta'}\log\pi_{\theta'}(a_t|s_t)\left(\sum_{t'=t}^T r(s_{t'},a_{t'})\right)\right)]\\

=\mathbb{E}_{\tau\sim p_{\theta}}\left[\left(\sum_{t=1}^T\nabla_{\theta'}\log\pi_{\theta'}(a_t|s_t)\right)\left(\prod_{t'=1}^T\frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_{\theta}(a_{t'}|s_{t'})}\right)\left(\sum_{t'=t}^T r(s_{t'},a_{t'})\right)\right]
$$

ignore reward at time $t'$ when computing gradient at time $t>t'$

$$
\nabla_{\theta'}J(\theta')=\mathbb{E}_{\tau\sim p_{\theta}}\left[\left(\sum_{t=1}^T\nabla_{\theta'}\log\pi_{\theta'}(a_t|s_t)\right)\left(\prod_{t'=1}^t\frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_{\theta}(a_{t'}|s_{t'})}\right)\left(\sum_{t'=t}^T r(s_{t},a_{t})\prod_{t''=t}^{t'}\frac{\pi_{\theta'}(a_{t''}|s_{t''})}{\pi_{\theta}(a_{t''}|s_{t''})}\right)\right]
$$

this equation is not trivial to compute

we can ignore the last term $\prod_{t''=t}^{t'}\frac{\pi_{\theta'}(a_{t''}|s_{t''})}{\pi_{\theta}(a_{t''}|s_{t''})}$, get a policy iteration algorithm. However, $\prod_{t'=1}^t\frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_{\theta}(a_{t'}|s_{t'})}$ might be exponentially small, since when sampling from $\pi_{\theta}$, $\pi_{\theta}(a_{t'}|s_{t'})$ is likely to be large.

First-order approximation of IS: sample $(s_{i,t},a_{i,t})$ from $\pi_{\theta}(s_t,a_t)$, then

$$
\nabla_{\theta}J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\frac{\pi_{\theta'}(s_{i,t},a_{i,t})}{\pi_\theta (s_{i,t},a_{i,t})}\nabla_{\theta'}\log\pi_{\theta'}(a_{i,t}|s_{i,t})\hat{Q}_{i,t}\\

\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}{\pi_\theta (a_{i,t}|s_{i,t})}\nabla_{\theta'}\log\pi_{\theta'}(a_{i,t}|s_{i,t})\hat{Q}_{i,t}
$$

You can see the intuition  [here](https://github.com/Hidden-Hyperparameter/RL-notes/blob/master/lecture/notes/5-policy_grad.md#with-first-order-approximation)

Rescale the gradient to avoid large jump

$$
\theta'\leftarrow\arg\max_{\theta'}(\theta'-\theta)^T\nabla_{\theta}J(\theta)\quad\text{s.t.}\quad D_{KL}(\pi_{\theta'},\pi_{\theta})\leq\epsilon
$$

Define fisher information matrix $F(\theta)=\mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}\log\pi_{\theta}(a|s)\nabla_{\theta}\log\pi_{\theta}(a|s)^T]$, then $D_{KL}(\pi_{\theta'},\pi_{\theta})=\frac{1}{2}(\theta'-\theta)^TF(\theta)(\theta'-\theta)$, using Lagrange multiplier, we get

$$
\theta'\leftarrow\theta+\alpha^j \sqrt{\frac{2\epsilon}{\nabla_\theta J(\theta)^TF(\theta)^{-1}\nabla_\theta J(\theta)}} F(\theta)^{-1}\nabla_{\theta}J(\theta)
$$

and we can choose $j$ by backtracking line search, which is the smallest value that satisfies the constraint and improves the objective function

#### Trust region policy optimization (TRPO):

define $\eta(\tilde{\pi})=\mathbb{E}_{\tau\sim\tilde{\pi}}[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)]$, which we want to maximize, then define

$$
A_{\pi}(s,a)=Q_{\pi}(s,a)-V_{\pi}(s)=\mathbb{E}_{s'\sim\mathcal{T}(s|s,a)}[r(s,a)+\gamma V_{\pi}(s')-V_{\pi}(s)]
$$

which represents the advantage of taking action $a$ at state $s$ under policy $\pi$, then

$$
\mathbb{E}_{s,a\sim\tilde{\pi}}[\sum_{t=0}^{\infty}\gamma^t A_{\pi}(s_t,a_t)]=\mathbb{E}_{s,a\sim\tilde{\pi}}[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)]-\mathbb{E}_{s_0\sim p(s_0)}[V_{\pi}(s_0)]\\

=\eta(\tilde{\pi})-\eta(\pi)
$$

so

$$
\eta(\tilde{\pi})=\eta(\pi)+\mathbb{E}_{s,a\sim\tilde{\pi}}[\sum_{t=0}^{\infty}\gamma^t A_{\pi}(s_t,a_t)]\\

=\eta(\pi)+\sum_{t=0}^{\infty}\sum_s\gamma^t P(s_t=s|\tilde{\pi})\sum_a\tilde{\pi}(a|s)A_{\pi}(s,a)
$$

define weighted sum of state probablity $\rho_{\tilde\pi}(s)=\sum_{t=0}^{\infty}\gamma^t P(s_t=s|\tilde{\pi})$, then

$$
\eta(\tilde{\pi})=\eta(\pi)+\sum_s\rho_{\tilde\pi}(s)\sum_a\tilde{\pi}(a|s)A_{\pi}(s,a)\\

=\eta(\pi)+\frac{1}{1-\gamma}\mathbb{E}_{s\sim\rho_{\tilde\pi},a\sim\tilde{\pi}}[A_{\pi}(s,a)]
$$

Two tricks:

1. estimate $\rho_{\tilde\pi}(s)$ using $\rho_{\pi}(s)$, because the difference between $\pi$ and $\tilde{\pi}$ is small
2. importance sampling (first order approximation)

$$
\eta(\tilde{\pi})\approx\eta(\pi)+\frac{1}{1-\gamma}\mathbb{E}_{s\sim\rho_{\pi},a\sim\pi}\left[\frac{\tilde{\pi}_\theta(a|s)}{\pi_\theta (a|s)}A_{\pi}(s,a)\right]=L_{\pi}(\tilde{\pi})
$$

we can discover $L_\pi(\tilde{\pi})$ and $\eta(\tilde{\pi})$ have same value and gradient at $\tilde{\pi}=\pi$, and define $\epsilon=\max_{s,a}|A_{\pi}(s,a)|$, then

$$
\eta(\tilde{\pi})\geq L_{\pi}(\tilde{\pi})-\frac{4\epsilon\gamma}{(1-\gamma)^2}D^{\max}_{KL}(\pi,\tilde{\pi})
$$

where max is over all state $s$, i.e. $D^{\max}_{KL}(\pi,\tilde{\pi})=\max_s D_{KL}(\pi(\cdot|s),\tilde{\pi}(\cdot|s))$, we can prove optimize the lower bound is equivalent to optimize the original objective function, we can simply use KL-divergence to estimate it. Our optimization problem becomes

$$
\max_{\tilde{\pi}}\left[\mathbb{E}_{s\sim\rho_{\pi},a\sim\pi}\left[\frac{\tilde{\pi}_\theta(a|s)}{\pi_\theta (a|s)}A_{\pi}(s,a)\right]-\frac{4\epsilon\gamma}{1-\gamma}D^{\max}_{KL}(\pi,\tilde{\pi})\right]
$$

if we use penalty coefficient $\frac{4\epsilon\gamma}{1-\gamma}$, then we can find our step size is very small, so we can use a constraint optimization problem

$$
\max_{\tilde{\pi}}\mathbb{E}_{s\sim\rho_{\pi},a\sim\pi}\left[\frac{\tilde{\pi}_\theta(a|s)}{\pi_\theta (a|s)}A_{\pi}(s,a)\right]\quad\text{s.t.}\quad D^{\max}_{KL}(\pi,\tilde{\pi})\leq\delta
$$

but since $D^{\max}_{KL}(\pi,\tilde{\pi})=\max_s D_{KL}(\pi(\cdot|s),\tilde{\pi}(\cdot|s))$, the number of constraints is infinite, we can simplify it by

$$
\max_{\tilde{\pi}}\mathbb{E}_{s\sim\rho_{\pi},a\sim\pi}\left[\frac{\tilde{\pi}_\theta(a|s)}{\pi_\theta (a|s)}A_{\pi}(s,a)\right]\quad\text{s.t.}\quad D^{\rho_\pi}_{KL}(\pi,\tilde{\pi})\leq\delta
$$

then see the optimization step above to get the estimate of $D^{\rho_\pi}_{KL}(\pi,\tilde{\pi})$, or maximize

$$
\mathcal{L}(\theta',\lambda)=\mathbb{E}_{s\sim\rho_{\pi},a\sim\pi}\left[\frac{\tilde{\pi}_\theta(a|s)}{\pi_\theta (a|s)}A_{\pi}(s,a)\right]-\lambda(D^{\rho_\pi}_{KL}(\pi,\tilde{\pi})-\delta)
$$

we maximize $\mathcal{L}(\theta',\lambda)$ w.r.t. $\theta'$ and then update $\lambda\leftarrow\lambda+\alpha(D_{KL}(\pi,\tilde{\pi})-\delta)$. After that, we update value function, and repeat the process.

Intuition: raise $\lambda$ if constraint is violated too much, else lower it

#### Proximal policy optimization (PPO):

Clip the ratio of new policy and old policy

$$
L_{\text{clip}}(\theta)=\mathbb{E}_{s,a\sim\pi_{\theta}}\left[\min\left(\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}A_{\pi_{\theta}}(s,a),\text{clip}(\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)},1-\epsilon,1+\epsilon)A_{\pi_{\theta}}(s,a)\right)\right]
$$

Adaptive KL Penalty Coefficient: $D_{KL}$ too small, reduce $\beta$, $D_{KL}$ too large, increase $\beta$

$$
L_{\text{penalty}}(\theta)=\mathbb{E}_{s,a\sim\pi_{\theta}}\left[\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}A_{\pi_{\theta}}(s,a)-\beta D_{KL}(\pi_{\theta},\pi_{\theta'})\right]
$$
