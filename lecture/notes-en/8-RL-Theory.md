## RL Theory

RL Theory problems: If I use this algorithm with $N$ samples, $k$ iterations, how is the result ? If I use this exploration strategy, how high is my regret ?

### Model based

Oracle exploration: for every $(s,a)$, sample $s'\sim P(s'|s,a)$ $N$ times, $\hat{P}(s'|s,a)=\frac{\# (s,a,s')}{N}$. Given a strategy $\pi$, use $\hat{P}$ to estimate $\hat{Q}^\pi$:

1. how close is $\hat{Q}^\pi$ to $Q^\pi$ ? $||\hat{Q}^\pi(s,a)-Q^\pi(s,a)||_\infty\leq\epsilon$
2. how close is $\hat{Q}^*$ if we learn it using $\hat{P}$ ? $||Q^*(s,a)-\hat{Q}^*(s,a)||_\infty\leq\epsilon$
3. how good is the resulting policy $\hat{\pi}$ ? $||Q^*(s,a)-Q^{\hat{\pi}}(s,a)||_\infty\leq\epsilon$

Using Hoeffding's inequality, we can get the sample complexity

$$
||\hat{P}(s'|s,a)-P(s'|s,a)||_1\leq c\sqrt{\frac{|S|\log(1/\delta)}{N}}
$$

with probability $1-\delta$

Use $P$ denote the transition matrix $P(s'|s,a)$, $\Pi$ denote probability distribution over policies, hence

$$
Q^\pi=r+\gamma PV^\pi\\

V^\pi=\Pi Q^\pi
$$

hence denote $P^\pi=P\Pi$, and

$$
Q^\pi-\hat{Q}^\pi=(I-\gamma P^\pi)^{-1}r-(I-\gamma\hat{P}^\pi)^{-1}r\\

=\gamma(I-\gamma\hat{P}^\pi)^{-1}(P^\pi-\hat{P}^\pi)Q^\pi=\gamma(I-\gamma\hat{P}^\pi)^{-1}(P-\hat{P})V^\pi
$$

where $(I-\gamma\hat{P}^\pi)^{-1}$ is evaluation operator, $(P-\hat{P})$ is the estimation error, and $V^\pi$ is the true value

Lemma: given $P^\pi$ and any vector $v\in\mathbb{R}^{|S||A|}$, since $P^\pi$ is a stochastic matrix, we have

$$
||(I-\gamma P^\pi)^{-1}v||_\infty\leq\frac{||v||_\infty}{1-\gamma}
$$

simplify above equation

$$
||Q^\pi-\hat{Q}^\pi||_\infty\leq\frac{\gamma}{1-\gamma}||(P-\hat{P})V^\pi||_\infty\leq\frac{\gamma}{1-\gamma}\left(\max_{s,a}||P(\cdot|s,a)-\hat{P}(\cdot|s,a)||_1\right)||V^\pi||_\infty
$$

w.l.o.g suppose $R_{\max}=1$, then $||V^\pi||_\infty\leq\frac{1}{1-\gamma}$, use union bound

$$
\Pr\{\max_{s,a}||P(\cdot|s,a)-\hat{P}(\cdot|s,a)||_1\leq c\sqrt{\frac{|S|\log(1/\delta)}{N}}\}\geq1-|S||A|\delta
$$

hence with probability $1-\delta$

$$
||Q^\pi-\hat{Q}^\pi||_\infty\leq c\frac{\gamma}{(1-\gamma)^2}\sqrt{\frac{|S|\log(|S||A|/\delta)}{N}}
$$

Note: this establishes for any fixed policy $\pi$, since the uncertainty of $\hat{P}$ is independent of $\pi$

Return the problem above

$$
||Q^*(s,a)-\hat{Q}^*(s,a)||_\infty\leq ||\sup_{\pi}Q^\pi-\sup_{\pi}\hat{Q}^\pi||_\infty\leq\sup ||Q^\pi-\hat{Q}^\pi||_\infty\leq c\frac{\gamma}{(1-\gamma)^2}\sqrt{\frac{|S|\log(|S||A|/\delta)}{N}}\\

||Q^*(s,a)-Q^{\hat{\pi}}(s,a)||_\infty\leq ||Q^*(s,a)-\hat{Q}^{\hat{\pi}}||_\infty+||\hat{Q}^{\hat{\pi}}-Q^{\hat{\pi}}||_\infty\leq2c\frac{\gamma}{(1-\gamma)^2}\sqrt{\frac{|S|\log(|S||A|/\delta)}{N}}
$$

### Model Free

Analyzing fitted $Q$-iteration: define $\hat{T}Q=r+\gamma P\max_a Q$ as Bellman operator

$$
\hat{Q}_{k+1}\leftarrow\arg\min_Q||\hat{Q}-\hat{T}\hat{Q}_k||_\infty
$$

no convergence guarantee if using $||\cdot||_2$

Error come from sampling error $T\neq\hat{T}$ and approximation error $\hat{Q}_{k+1}\neq\hat{T}\hat{Q}_k$

$$
|\hat{T}Q(s,a)-TQ(s,a)||=|\hat{r}(s,a)-r(s,a)+\gamma\left(\mathbb{E}_{\hat{P}(s'|s,a)}[\max_{a'}Q(s',a')]-\mathbb{E}_{P(s'|s,a)}[\max_{a'}Q(s',a')]\right)\\

\leq |\hat{r}(s,a)-r(s,a)|+\gamma||\hat{P}(\cdot|s,a)-P(\cdot|s,a)||_1||Q||_\infty\\

\leq c_1 R_{\max}\sqrt{\frac{\log1/\delta}{2N}}+c_2 ||Q||_\infty\sqrt{\frac{\log1/\delta}{N}}
$$

Obviously, $||Q||_\infty=O(\frac{1}{1-\gamma}R_{\max})$. Using union bound

$$
\Pr\left\{||\hat{T}Q-TQ||_\infty\leq c\frac{R_{\max}}{1-\gamma}\sqrt{\frac{\log |S||A|/\delta}{N}} \right\} \geq1-\delta
$$

Approximation error

$$
||\hat{Q}_{k}-Q^*||_\infty\leq ||\hat{Q}_{k}-T\hat{Q}_{k-1}||_\infty+||T\hat{Q}_{k-1}-TQ^*||_\infty\\

\leq\epsilon_{k-1}+\gamma ||\hat{Q}_{k-1}-Q^*||_\infty
$$

using fact that $T$ is a $\gamma$-contraction mapping, then

$$
\lim_{k\to\infty}||\hat{Q}_{k}-Q^*||_\infty\leq\lim_{k\to\infty}\left(\sum_{i=0}^{k-1}\gamma^i\epsilon_{k-i-1}+\gamma^k||\hat{Q}_0-Q^*||_\infty\right)\leq\frac{1}{1-\gamma}\max_k\epsilon_k
$$

$\epsilon_k$ measures how much $\hat{Q}_{k+1}$ deviates from $T\hat{Q}_k$

$$
||\hat{Q}_{k+1}-T\hat{Q}_k||_\infty\leq ||\hat{Q}_{k+1}-\hat{T}\hat{Q}_k||_\infty+||\hat{T}\hat{Q}_k-T\hat{Q}_k||_\infty
$$

first term depends on learning process, second term can be estimated using previous result
