## Actor-Critic

lower variance due to critic, but not unbiased if critic is not perfect

$$
\nabla_{\theta}J(\theta)=\mathbb{E}_{\tau\sim p_{\theta}}[\left(\sum_{t=1}^T\nabla_{\theta}\log\pi_{\theta}(a_t|s_t)A^{\pi}(s_{t},a_{t})\right)]
$$

just fit $V^{\pi}(s)$ to estimate $A_{\pi}(s,a)=Q_{\pi}(s,a)-V_{\pi}(s)=\mathbb{E}_{s'\sim\mathcal{T}(s,a)}[r(s,a)+\gamma V_{\pi}(s')-V_{\pi}(s)]$ with parameters $\phi$, with supervised learning

$$
\mathcal{L}(\phi)=\mathbb{E}_{s\sim\pi_{\theta}}[(\hat{V}_\phi^\pi(s_i)-y_i)^2]
$$

where $y_i$ is reward from state $s_i$ to the end of the trajectory

batch actor-critic algorithm:

1. sample a batch of trajectories $\tau=\{s_i,a_i\}$ from $\pi_{\theta}(a|s)$
2. use the batch to update $\hat{V}_\phi^\pi(s)$ by $\nabla_\phi\mathcal{L}(\phi)$
3. evaluate $\hat{A}^{\pi}(s,a)=r(s,a)+\gamma\hat{V}_\phi^\pi(s')-\hat{V}_\phi^\pi(s)$
4. update $\theta$ by $\nabla_{\theta}J(\theta)$

To simplify calculation $y_i\approx r(s_i,a_i)+\gamma\hat{V}_\phi^\pi(s_{i+1})$

online actor-critic algorithm, update $\phi$ and $\theta$ after each sampled trajectory:

1. sample a trajectory $\tau=\{s_i,a_i,s',r\}$ from $\pi_{\theta}(a|s)$
2. update $\hat{V}_\phi^\pi(s)$ using target $r+\gamma\hat{V}_\phi^\pi(s')$
3. evaluate $\hat{A}^{\pi}(s,a)=r(s,a)+\gamma\hat{V}_\phi^\pi(s')-\hat{V}_\phi^\pi(s)$
4. update $\theta$ by $\nabla_{\theta}J(\theta)=\nabla_{\theta}\log\pi_{\theta}(a|s)\hat{A}^{\pi}(s,a)$

off-policy actor-critic algorithm:

1. take action $a\sim\pi_{\theta}(a|s)$, get $(s,a,s',r)$ store in $\mathcal{R}$
2. sample a batch of trajectories $\tau=\{s_i,a_i,s'_i,r_i\}$ from $\mathcal{R}$
3. update $\hat{Q}_\phi^\pi(s_i,a_i)$ using target $r+\gamma\hat{Q}_\phi^\pi(s'_i,a'_i)$ for each $s_i,a_i$, $a'_i\sim\pi_{\theta}(a'_i|s'_i)$
4. update $\theta$ by $\nabla_{\theta}J(\theta)=\frac{1}{N}\sum_{i=1}^N\nabla_{\theta}\log\pi_{\theta}(a_i^{\pi}|s_i)\hat{Q}_\phi^\pi(s_i,a_i^{\pi})$ where $a_i^{\pi}=\pi_\theta(a|s_i)$

Changes:

1. The reason why we use $\hat{Q}_\phi^\pi(s_i,a_i)$ instead of $\hat{V}_\phi^\pi(s_i)$ is estimating $V$ using target $r+\gamma\hat{V}_\phi^\pi(s')$ is not accurate, since $s_i'$ is not the state after taking action $a_i^{\pi}$, but the state after taking action $a_i^{\pi_{\theta_{\text{old}}}}$
2. Advantage function here is $\hat{A}^{\pi}(s,a)=\hat{Q}_\phi^\pi(s_i,a_i)-\mathbb{E}_{a\sim\pi_{\theta}}[\hat{Q}_\phi^\pi(s_i,a)]$, but computing the expectation is hard, so we use $\hat{Q}_\phi^\pi(s_i,a_i)$ instead. (use more data to reduce the variance instead of baseline)
3. It's worth noting that step 4 here sample trajectories from $\mathcal{R}$, not from $\pi_{\theta}$. The policy we get above is an optimal solution on a boarder distribution, not on $\pi_{\theta}$, but we trust the generalization ability of neural network

To keep estimator unbiased, pick **state-dependent baseline** $V_\phi^\pi(s)$

$$
\nabla_{\theta}J(\theta)=\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_{\theta}\log\pi_{\theta}(a_i^{\pi}|s_i)\left(\left(\sum_{t'=t}^T\gamma^{t'-t}r(s_{i,t'},a_{i,t'})\right)-V_\phi^\pi(s_{i,t})\right)
$$

or **action-dependent baseline** $Q_\phi^\pi(s,a)$, provide second term to keep unbiased and reduce variance (since we can sample from $\pi_{\theta}$ efficiently), we only need a few samples to compute the first term.

$$
\nabla_{\theta}J(\theta)=\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_{\theta}\log\pi_{\theta}(a_i^{\pi}|s_i)\left(\hat{Q}_{i,t}-Q_\phi^\pi(s_{i,t},a_{i,t})\right)+\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_{\theta}\mathbb{E}_{a\sim\pi_{\theta}(\cdot|s_{i,t})}[Q_\phi^\pi(s_{i,t},a)]
$$

n-step returns: control bias and variance trade-off

$$
\hat{A}_n^{\pi}(s_t,a_t)=\sum_{t'=t}^{t+n-1}\gamma^{t'-t}r(s_{t'},a_{t'})+\gamma^n V_\phi^\pi(s_{t+n})-V_\phi^\pi(s_t)
$$

Generalized Advantage estimation: cut everywhere all at once, use weighted combination of n-step returns, pick weight $w_n=(1-\lambda)\lambda^{n-1}$

$$
\hat{A}_{\text{GAE}}^{\pi}(s_t,a_t)=\sum_{n=1}^{\infty}w_n\hat{A}_n^{\pi}(s_t,a_t)=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\left(\sum_{t'=t}^{t+n-1}\gamma^{t'-t}r(s_{t'},a_{t'})+\gamma^n V_\phi^\pi(s_{t+n})-V_\phi^\pi(s_t)\right)\\

=(1-\lambda)\sum_{t'=t}^\infty\gamma^{t'-t}r(s_{t'},a_{t'})\sum_{n=t'-t+1}^\infty\lambda^{n-1}+\sum_{t'=t}^\infty(\gamma\lambda)^{t'-t}(\gamma V_\phi^\pi(s_{t'+1})-V_\phi^\pi(s_{t'}))\\

=\sum_{t'=t}^\infty(\gamma\lambda)^{t'-t}(r(s_{t'},a_{t'})+\gamma V_\phi^\pi(s_{t'+1})-V_\phi^\pi(s_{t'}))=\sum_{t'=t}^\infty(\gamma\lambda)^{t'-t}A^{\pi}(s_{t'},a_{t'})
$$

which have a good interpretation
