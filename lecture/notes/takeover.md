# 2-Imitation Learning

**Behavioral Cloning**

$$
\theta^\star = \argmax_{\theta}\log \pi_\theta(a=\pi^\star(s)|s)
$$

**Dagger**

$$
\mathcal{D}= \mathcal{D}\cup \left\{(s_t,a_t^\star=\pi^{\star}(s_t))|t=1,2,\cdots,T\right\}
$$

# 3-4

![](./assets/not_implement.png)

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

**Model Predictive Control**

重复:

1. 根据当前的state $s_t$，把它当成第一步，对$\sum_{t'=t}^{t+T_0}c(s_{t'},a_{t'})$运行iLQR，得到最优的一系列action $a_t,a_{t+1},\cdots,a_{t+T_0}$；
2. 扔掉后面的所有action，保留$a_t$，并和环境进行一步交互。

# 11 Model-Based RL


**Model-Based RL Algorithm** (a.k.a. Model Predictive Control,MPC)

1. 用某种baseline policy $\pi_0$（比如随机采样）获得一个数据集$D=\{(s,a,s')\}$；
2. 重复：
    1. 在$D$上学习一个动力学模型$f(s,a)$，使得$f(s,a)\approx s'$；
    2. 使用某种planning的方法来更新最优策略$\pi$。一般地，random shooting就足够了；
    3. 运行当前的最新策略$\pi$ **仅一步**, 收集 **一组（不是一个）** 数据，加入数据集：$D=D\cup \{(s,a,s')\}$；

**Uncertainty**

核心关系：

$$
p(s_{t+1}|s_t,a_t)=\mathbb{E}_{\theta\sim p(\theta|D)}[p(s_{t+1}|s_t,a_t;\theta)]
$$

Algorithm:

1. 对每一个ensemble中的模型$\theta$：
    1. 对$t=1,2,\cdots,T-1$，**不断用这一个** $\theta$计算$s_{t+1}$；
    2. 计算$J$。
2. 计算各个得到的$J$的平均值。

## 12 Model Based Method with Policy

**MBPO Algorithm**

```python
while True:
    # sample data
    if q_net is None: # first epoch, random sample
        env_data = SampleFromEnv(policy=random)
    else:
        env_data = SampleFromEnv(policy=ExplorePolicy(q_net))
    dynamic_buffer.AddData(env_data)
    q_buffer.AddData(env_data)

    # train dynamic model
    dynamic_model.Train(dynamic_buffer.Sample())
    for _ in range(K):
        # add additional data to Q learning data buffer
        start_point = dynamic_buffer.Sample(n)['states'] # n: a hyperparameter, usually = 1
        path = dynamic_model.GetTrajectories(
            start=start_point,
            length=T, 
            policy=q_net
        )
        q_buffer.AddData(path)

        # get Q-learning data
        train_q_data = q_buffer.Sample()
        for _ in range(S):
            q_net.GradStep(train_q_data)
            q_net.Update()
```

**Dyna Algorithm**

重复：

1. 运行某种policy（需要exploration）获得一些$(s,a,s',r)$，加入replay buffer $B$；
2. 用这一组数据更新一步dynamic model；
3. 用这一组数据更新一次Q function；
4. 重复$K$次：
    1. 从buffer中取出一组数据$(s,a)$。
    2. 从model给出的概率 $p(s'|s,a)$ 中采样 **几个** $s'$，并用期待值再次更新Q function。

**Successor Representation**

$$
r(h)=\sum_j w_j\phi_j(h)
$$

$$
\psi_j (h_t):=\sum_h \phi_j(h)p_{\text{fut}}(h|h_t)
$$

$$
Q^\pi(h_t)=\frac{1}{1-\gamma} \sum_j w_j\psi_j (h_t)
$$

$$
\psi_j (h_t)=(1-\gamma)\phi_j(h_t)+\gamma\sum_{h'}p_\pi(h'|h_t)\psi_j(h')
$$

**Algorithm**

1. 收集一系列和环境交互的数据$(s,a,r)$，训练参数矩阵$w$使得$|r(s,a)-\sum_j w_j\phi_j(s,a)|^2$被最小化；
2. 选定很多很多个policy $\pi_1,\cdots,\pi_k$
3. 对$i=1,\cdots,k$：
    1. 根据当前的策略$\pi$，使用上面的递推关系，类似Q-learning地训练$\psi_j$；
    2. 利用$\psi$和$w$计算$Q(s,a)=\frac{1}{1-\gamma} \sum_j w_j\psi_j (s,a)$。
4. 选出一个表现超过前面所有policy的policy $\pi^\star$。具体地，令

$$
\pi^\star(s)=\arg\max_a \max_i \sum_j w^{\pi_i}_j\psi^{\pi_i}_j(s,a) 
$$

