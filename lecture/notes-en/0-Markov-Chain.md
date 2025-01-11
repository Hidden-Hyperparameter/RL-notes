## Markov Chain

$\mathcal{M}=\{\mathcal{S},\mathcal{A}, \mathcal{T}, r\}$, $r:\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}$

$\mu_{t,j}=p(s_t=j),\xi_{t,k}=p(a_t=k),\mathcal{T}_{i,j,k}=p(s_{t+1}=i|s_t=j,a_t=k)$

$$
\mu_{t+1,i}=\sum_{i,k}\mu_{t,j}\xi_{t,k}\mathcal{T}_{i,j,k}
$$

Partially observable Markov decision process

$\mathcal{M}=\{\mathcal{S},\mathcal{A},\mathcal{O}, \mathcal{T}, \mathcal{E}, r\}$， $\mathcal{O}$ observation space, $\mathcal{E}$ emission probability $p(o_t|s_t)$

$$
p_{\theta}(s_1,a_1,\dots,s_T,a_T)=p(s_1)\prod_{t=1}^T \pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t)\\
\theta^*=\arg\max_{\theta}\sum_{s_1,a_1,\dots,s_T,a_T}\mathbb{E}_{\tau\sim p_{\theta}}[r(\tau)]=\arg\max_{\theta}\sum_{t=1}^T\mathbb{E}_{s_t, a_t\sim p_{\theta}(s_t,a_t)}[r(s_t,a_t)]
$$

stationary distribution $\mu=\mathcal{T}\mu$, infinite horizon case

$$
\theta^*=\arg\max_{\theta}\frac{1}{T}\sum_{t=1}^T\mathbb{E}_{s_t, a_t\sim p_{\theta}(s_t,a_t)}[r(s_t,a_t)]\rightarrow \arg\max_{\theta}\mathbb{E}_{(s,a)\sim p_{\theta}(s,a)}[r(s,a)]
$$
