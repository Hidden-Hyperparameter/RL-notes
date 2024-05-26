# 2-Imitation Learning

**Behavioral Cloning**

$$
\theta^\star = \argmax_{\theta}\log \pi_\theta(a=\pi^\star(s)|s)
$$

**Dagger**

$$
\mathcal{D}= \mathcal{D}\cup \left\{(s_t,a_t^\star=\pi^{\star}(s_t))|t=1,2,\cdots,T\right\}
$$

# 5-Policy Gradient

**Policy Gradient with Causality**

$$
\nabla_\theta J(\theta)\approx \frac{1}{N}\sum_{n=1}^N \sum_{t=1}^T \nabla_\theta\log \pi_\theta(a_t|s_t)\left(\sum_{t'=t}^T r(s_{t'},a_{t'})\right)=\frac{1}{N}\sum_{n=1}^N \sum_{t=1}^T \nabla_\theta\log \pi_\theta(a_t|s_t)\hat{Q}^{\pi_\theta}_{n,t}
$$

# 6-Actor-Critic

**Vanilla Actor-Critic with discount**

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=1}^T\nabla_{\theta}\log \pi_\theta(a_t|s_t)\left(r(s_t,a_t)+\gamma V^{\pi_\theta}(s_{t+1})-V^{\pi_\theta}(s_t)\right)\right]
$$

$$
\hat{L}(\phi)=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T}\left(\text{SG}[\hat{y}_{n,t}]-V^{\pi_\theta}_{\phi}(s_{n,t})\right)^2=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T}\left(r(s_{n,t},a_{n,t})+ \text{SG}[\gamma V^{\pi_\theta}_{\phi}(s_{n,t+1})]-V^{\pi_\theta}_{\phi}(s_{n,t})\right)^2
$$

**Off-Policy Actor-Critic Algorithm**

$$
L_{\text{new}}(\phi)=\sum_{\text{batch}}\left(r(s_{t},a_{t})+\gamma Q^{\pi_\theta}_{\phi}(s_{t+1},a_{t+1})-Q^{\pi_\theta}_{\phi}(s_{t},a_t)\right)^2
$$

1. 利用现在的policy $\pi_\theta$ 走一步，得到$\{s_t,a_t,s_{t+1},r(s_t,a_t)\}$，存入Replay Buffer；
2. 从Replay Buffer中随机取一个batch，用这些数据训练 $Q^{\pi_\theta}_{\phi}$（见前面的目标函数）。这个过程中，对每一组数据我们需要采样一个新的action $a_{t+1}$；
3. 取 $A^{\pi_\theta}(s_t,a_t)=Q^{\pi_\theta}_\phi(s_t,a_t)$；
4. 计算 $\nabla_\theta J(\theta)=\sum_{\text{batch}}\nabla_{\theta}\log \pi_\theta(\tilde{a_t}|s_t)A^{\pi_\theta}(s_t,\tilde{a_t})$，其中$\tilde{a_t}$是从$\pi_\theta$中采样得到的；
5. 用这个梯度更新 $\theta$。

**State-Dependent Baseline**

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)\left[\left(\sum_{t'=t}^T \gamma^{t'-t} r(s_{t'},a_{t'})\right)-V^{\pi_\theta}_\phi(s_t)\right]\right]
$$

**Control Variates**

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)\left(\hat{Q}^{\pi_\theta}_{t}-Q^{\pi_\theta}_\phi(s_t,a_t)\right)\right]+\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)Q^{\pi_\theta}_\phi(s_t,a_t)\right]
$$

**N-Step Return**

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=1}^T \nabla_{\theta}\log \pi_\theta(a_t|s_t)\left[\left(\sum_{t'=t}^{t+n-1} \gamma^{t'-t} r(s_{t'},a_{t'})\right)+\gamma^n V_\phi^{\pi_\theta}(s_{t+n})-V^{\pi_\theta}_\phi(s_t)\right]\right]
$$


**Generalized Advantage Estimation**

$$
A_{\text{GAE}}^{\pi_\theta}(s_t,a_t)=\sum_{n=1}^\infty \lambda^{n-1} A^{\pi_\theta}_n(s_t,a_t)
$$
