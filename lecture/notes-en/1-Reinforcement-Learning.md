## Reinforcement learning

Q-function

$$
Q^{\pi}(s_t,a_t)=\sum_{t'=t}^T\mathbb{E}_{\tau\sim p_{\pi}}[r(\tau)|s_t,a_t]
$$

Value function

$$
V^{\pi}(s_t)=\mathbb{E}_{a_t\sim\pi(a_t|s_t)}[Q^{\pi}(s_t,a_t)]=\mathbb{E}_{a_t\sim\pi(a_t|s_t)}[Q^{\pi}(s_t,a_t)]
$$

Four types of RL algorithms

- Policy gradient: differentiate the objective function

$$
\mathbb{E}_{(s,a)\sim p_{\theta}(s,a)}[r(s,a)]\quad\theta\leftarrow\theta+\alpha\nabla_{\theta}\mathbb{E}_{(s,a)\sim p_{\theta}(s,a)}[r(s,a)]
$$

- Value-based: estimate value function and Q-function $Q^{\pi}(s_t,a_t)$, set $\pi(s)=\arg\max_a Q^{\pi}(s,a)$
- Actor-critic: estimate value function and Q-function, then update policy $\pi(a_t|s_t)$ by $\nabla_{\theta}\mathbb{E}(Q^{\pi}(s_t,a_t))$
- Model-based RL: estimate transition probability $\mathcal{T}$ and reward function $r$, use it for planning and policy optimization (Backpropagation through actions or policy)

Off policy: improve policy without generating new samples
On policy: need to generate new samples even if the policy changes a little

Common assumptions: full observability, episodic learning (policy gradient methods, model-based RL methods), continuity or smoothness