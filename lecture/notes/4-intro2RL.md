# Markov Decision Process (MDP)

A Markov Chain is 
$$
\mathcal{M}=\{S,T\}
$$

where $S$ is a set of states and $T$ is a transition matrix. If we denote the distribution as a matrix:
$$
\mu_{t,s}=p_t(s),\forall s\in S
$$

The transition is given by
$$
\mu_{t+1,s'}=\sum_{s\in S}T(s',s)\mu_{t,s}=(T\mu_t)_{s'}
$$

A **Markov Decision Process** is 
$$
\mathcal{M}=\{S,A,T,r\}
$$

where $S$ is the state space, $A$ is the action space, and the $T$ is the transition matrix which states
$$
p_{t+1}(s')=\sum_{s,a}T(s',s,a)p_t(s)p^{(A)}_t(a)
$$

where $p^{(A)}_t(a)$ is the probability of taking action $a$ at time $t$.


$r$ is the **reward function**, which gives a reward
$$
r(s,a)\in \mathbb{R}
$$
for any given state-action pair.

In **Partially Observed MDP**, observation $o_t$ is included in the system, which is stochastically determined by $s_t$: a probability distribution $O(o_t|s_t)$.

For a MDP, the **goal** is to maximize