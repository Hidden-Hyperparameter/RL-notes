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

(Don't use it in practice!)

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

(**Very good** in practice!)

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

# 7-Value Function

**Value Iteration**

$$
V^\pi(s_t)\leftarrow \max_{a_t}\left\{r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}(s_{t+1})]\right\}
$$

**Q Iteration**

$$
V^\pi(s_t)\leftarrow \max_{a_t}\left\{r(s_t,a_t)+\gamma \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)}[V^{\pi}(s_{t+1})]\right\}
$$

**Fitting Q Iteration Algorithm**

重复：
1. 收集数据集，包含一系列的$\{s,a,s',r\}$；
2. 重复$K$次：
    1. 根据公式计算$Q$的目标；
    2. 作$S$次梯度下降，最小化拟合误差

**Vanilla Q Learning Algorithm**

重复：
1. 从环境中根据某种policy采样一个$(s,a,s',r)$；
2. 计算**一个** $[Q^\pi(s,a)]^\star=r(s,a)+\gamma\max_{a'}Q^{\pi}_\phi(s',a')$
3. 对 $L=([Q^\pi(s,a)]^\star-Q^{\pi}_\phi(s,a))^2$作**一步**梯度下降。

**DQN Algorithm**

```python
while True:
    for _ in range(N):
        InteractWithEnv()
        for _ in range(K): # usually K=1
            GetTargetFromBuffer()
            GradStep()
    phi_0=phi
```

# 8-Q Learning

**Double Q-learning**

$$
Q_\phi(s_t,a_t)\leftarrow r(s_t,a_t)+\gamma Q_{\phi_0}(s_{t+1},\arg \max Q_{\textcolor{red}{\phi}}(s_{t+1},a))
$$


**DDPG Algorithm**

重复：
1. 从环境中根据某种policy采样一个$(s,a,s',r)$，加入replay buffer $B$；
2. 从replay buffer取出一个batch(相当于$K=1$)，计算目标 $[Q(s,a)]^\star=r(s,a)+\gamma Q_{\phi_0}(s',\pi_{\theta_0}(s'))$；
3. 对 $L(\phi)=\sum_{(s,a)}([Q(s,a)]^\star-Q_\phi(s,a))^2$作一步梯度下降；
4. 对 $L(\theta)=-\sum_s Q_\phi(s,\pi_\theta(s))$做一步梯度下降；
5. 更新$\phi_0,\theta_0$，可以使用隔$N$次更新一次的方法，也可以使用Polyak平均的方法。

# 9-Advanced Policy Gradients

**Natural Gradients**

$$
\theta_1\leftarrow \theta_0 + \eta \frac{F^{-1}g}{\sqrt{g^TF^{-1}g}},g=\nabla_\theta J(\theta)
$$

# 10 Optimal Control & Planning

**Cross Entropy Method**

重复：
1. 随机采样 $\mathbf{A}_1,\mathbf{A}_2,\cdots,\mathbf{A}_N$（这里$\mathbf{A}=(a_1,\cdots,a_T)$）；
2. 计算 $J(\mathbf{A}_1),J(\mathbf{A}_2),\cdots,J(\mathbf{A}_N)$；
3. 保留最好的 $M$ 个数据点，更新 $p(\mathbf{A})$ 使得它更接近这 $M$ 个数据点的分布。一般地，$p(\mathbf{A})$ 取为高斯分布。

**Monte Carlo Tree Search**

```python
def MCTS(root):
    while True:
        # Selection & Expansion
        node = TreePolicy(root)
        # Simulation
        reward = DefaultPolicy(node)
        # Backpropagation
        update(node,reward)

def update(node,reward):
    while True:
        node.N += 1
        node.Q += reward
        if node.parent is None:
            break
        node = node.parent

def TreePolicy(expanded):
    """UCT Tree Policy"""
    if not root.fully_expanded():
        return root.expand()
    else:
        best_child = argmax([Score(child) for child in root.children])
        return TreePolicy(best_child)

def Score(node):
    return node.Q/node.N + C*sqrt(log(node.parent.N)/node.N)
```

**LQR**

- Backward Pass

1. 初始化：$\mathbf{Q}_T=\dbinom{A_T\quad B_T}{B_T^T\quad C_T},\mathbf{q}_T=\dbinom{x_T}{y_T}$
2. 对$t=T,T-1,\cdots,1$：
    1. $a_t=-C_t^{-1}(y_t+B_ts_t)$
    2. $V_t=A_t-B_t^TC_t^{-1}B_t, \quad v_t=xt-B_t^TC_t^{-1}y_t$
    3. $\dbinom{A_{t-1}\quad B_{t-1}}{B_{t-1}^T\quad C_{t-1}}\texttt{ += } \mathbf{F}_{T-1}^TV_{t}\mathbf{F}_{T-1}$（对应 $\mathbf{Q}_{t-1}$）； $\dbinom{x_{t-1}}{y_{t-1}}\texttt{ += } (\mathbf{F}_{T-1}^TV_{t}^T+v_{t}^T)\mathbf{F}_{T-1}$ （对应$\mathbf{q}_{t-1}$）

- Forward Pass

对$t=1,2,\cdots,T$:

1. $a_t=-C_t^{-1}(y_t+B_ts_t)$
2. $s_{t+1}=\mathbf{F_t}\binom{s_t}{a_t}+f_t+f_t$

**iLQR**

重复：
1. 计算各个偏导数；
2. 运行LQR的backward pass，得到$a_t$关于$s_t$的线性关系的系数（即前面的$a_t=-C_t^{-1}(y_t+B_ts_t)$）；
3. 运行LQR的forward pass（可以引入$\alpha$以保证收敛性）。但这一步计算$s_t$的时候，我们**必须采用真实的演化函数$f$**，而不是线性近似。