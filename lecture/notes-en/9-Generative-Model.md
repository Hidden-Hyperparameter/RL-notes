## Generative Model

Latent variable models: generate using $p_\theta(x|z)\sim\mathcal{N}(f(z;\theta),I)$

$p(x)=\int p(x|z)p(z)dz$, where $p(z)$ is a prior. Using maximum likelihood

$$
\theta\leftarrow\arg\max_\theta\sum_i\log p_\theta(x_i)=\arg\max_\theta\sum_i\log\int p_\theta(x_i|z)p(z)dz
$$

however, the integral is intractable, one way to estimate is

$$
\theta\leftarrow\arg\max_\theta\sum_i\mathbb{E}_{z\sim p(z|x_i)}\left[\log p_\theta(x_i,z)\right]
$$

intuition: we choose a $z$ with highest probability given $x_i$, then use $z$ to estimate the integral. However, $z$ is high-dimensional, we sample from $p(z|x_i)$

Left problem: how to sample from $p(z|x_i)$ ?

Variational Inference:

$$
\log p(x_i)=\log\int p_\theta(x_i|z)p(z)dz=\log\mathbb{E}_{z\sim q_i(z)}\left[\frac{p_\theta(x_i|z)p(z)}{q_i(z)}\right]\\

=\mathbb{E}_{z\sim q_i(z)}\left[\log p_\theta(x_i|z)+\log p(z)\right]+\mathcal{H}(q_i)+D_{KL}(q_i(z)||p_\theta(z|x_i))\\

=\mathcal{L}(p_\theta, q_i)+D_{KL}(q_i(z)||p_\theta(z|x_i))
$$

we want to use $\mathbb{E}_{z\sim q_i(z)}\log\left[\frac{p_\theta(x_i|z)p(z)}{q_i(z)}\right]$ to estimate $\log p(x_i)$, the difference is KL-divergence. We want to minimize the difference

Minimizing $D_{KL}(q_i(z)||p(z|x_i))$ is equivalent to maximizing $\mathbb{E}_{z\sim q_i(z)}\log\left[\frac{p(x_i|z)p(z)}{q_i(z)}\right]$

Choose $q_i(z)$ to be a Gaussian $\mathcal{N}(\mu_i,\Sigma_i)$, our algorithm work as

1. sample $z\sim q_i(z)$, update $\theta$ by $\nabla_\theta\mathcal{L}(p_\theta,q_i)\approx\nabla_\theta\log p_\theta(x|z)$
2. update $q_i$ to maximize $\mathcal{L}(p_\theta,q_i)$

Amortized Variational Inference: Instead of learning $q_i(z)$ for each $x_i$, we learn $q_\phi(z|x)=\mathcal{N}(\mu_\phi(x),\Sigma_\phi(x))$ to approximate $p(z|x)$

Representation Learning:

1. Train VAE on states in replay buffer $\mathcal{R}$
2. Run RL, using $z$ as the state instead of $s$

Conditional Models: use decoder $p_\theta(x|y,z;\theta)$ and encoder $q_\phi(z|x,y,\phi)$, where $y_i$ is the label

$$
\mathcal{L}_i=\mathbb{E}_{z\sim q_\phi(z|x_i,y_i)}[\log p_\theta(x_i|y_i,z)+\log p(z|y_i)]+\mathcal{H}(q_\phi(z|x_i,y_i))
$$
