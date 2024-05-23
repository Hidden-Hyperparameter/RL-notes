# Notations
- $a=\pi^{\star}(s)$: the expert policy gives $a$ when the state is $s$
- $\pi_\theta$: the policy we are trying to learn
- Use $|p_1-p_2|$ to denote the total variance distance between $p_1$ and $p_2$: $|p_1-p_2|=\sum_{x}|p_1(x)-p_2(x)|$

# Behavior Cloning Analysis

## Distribution Distance
**Assumptions.**
- $\forall (a,s), \pi_\theta(a\ne \pi^{\star}(s)|s)\le \epsilon$

**Conclusion**: for arbitrary $t$,

$$
|p_{\pi_\theta}-p_{\pi^{\star}}|\le 2\epsilon t.
$$

**Proof**. Both the expert and our model take $t$ actions to reach $s_t$ from $s_0$. Thus, there are at least probability $\lambda=(1-\epsilon)^t$ such that the model make decision the same as the expert. This gives

$$
p_{\pi_\theta}(s_t)=\lambda p_{\pi^{\star}}(s_t)+(1-\lambda)p_{\text{fail}}(s_t),
$$

where $p_{\text{fail}}(s_t)$ is some unknown probability. However,
$$
|p_{\pi_\theta}-p_{\pi^{\star}}|=(1-\lambda)|p_{\text{fail}}-p_{\pi^{\star}}|,
$$

and we know that the total variance distance is bounded by $2$. Notice that $1-\lambda\le 1-(1-\epsilon)^t\le \epsilon t$, we are done.

## The total cost

Define $c_t$ to be the cost function given by
$$
c_t(s_t,a_t)=\begin{cases}0&,a_t=\pi^{\star}(s_t)\\1&,\text{otherwise}\end{cases}.
$$

**Conclusion**: 

$$
S=\sum_{t\le T} E_{s_t\sim p_{\pi_\theta}}[c_t(s_t,a_t)]=\mathcal{O}(\epsilon T^2).
$$

**Proof**. 

$$
S=\sum_{t\le T} E_{s_t\sim p_{\pi_\theta}}[c_t(s_t,a_t)]=\sum_{t\le T} \sum_{s_t}p_{\pi_\theta}(s_t)c_t(s_t,a_t)
$$

$$
\le \sum_{t\le T} \sum_{s_t}p_{\pi^\star}(s_t)c_t(s_t,a_t)+\sum_{t\le T} \sum_{s_t}|p_{\pi_\theta}(s_t)-p_{\pi^{\star}}(s_t)|c_t(s_t,a_t)
$$

Now, we use the previous result to get

$$
S\le \sum_{t\le T} \sum_{s_t}p_{\pi^\star}(s_t)\cdot \epsilon+\sum_{t\le T} 2\epsilon t = \mathcal{O}(\epsilon T^2).
$$

(这里有一点不太明白，为什么不能直接对$p_{\pi_\theta}(s_t)$作上面第一项的估计)