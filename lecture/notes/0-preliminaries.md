# General

本仓库是我学习Berkeley的课程[CS285(DRL)](http://rail.eecs.berkeley.edu/deeprlcourse/)时候记录的自学笔记。

本笔记基本可以说包含了PPT和讲课上的大部分内容，也成一个体系。我个人所希望的也是，经过我们大家一起合作，仅参考本笔记就可以完成RL入门知识的学习，比听课更加快速，又比单纯看PPT更为轻松。

笔记文件可以在[这里](./lecture/notes)找到（为了最好的阅读体验，建议下载仓库并用VS code阅读）。

本笔记的架构：每一讲有一个单独的笔记。此外，[takeaway](./lecture/notes/takeaway.md)总结了所有讲的要点，类似于一个“cheatsheet”。

此外，如果有些笔记引用了作业的内容，你可以在[这里](../../homework_repo/)找到。这就是CS285这一课程本身的作业。

# For Contributors

本笔记基本由我一个人完成，因此肯定有大量的错误。同时，我只是RL领域的初学者，还未入门，因此对很多算法的个人理解也很可能有偏差。如果你发现了错误，或者有任何建议，欢迎提出issue或者PR！

如果这个仓库逐渐有了公开的性质（而不是只有我自己看），可能会考虑推出一个英文版本。（Not Implemented）

# Table Of Contents

Takeaway/Cheatsheet: [Here](./lecture/notes/takeaway.md)

0. Preliminaries (just this file)
1. What is RL (not implemented)
2. Imitation Learning [Here](./lecture/notes/2-imitation_learning.md)
3. Pytorch Basics [(Not complete)](./lecture/notes/3-pytorch.md)
4. Introduction to RL [Here](./lecture/notes/4-intro2RL.md)
5. Policy Gradients [Here](./lecture/notes/5-policy_grad.md)
6. Actor Critic Algorithms [Here](./lecture/notes/6-actor-critic.md)
7. Value Function Methods [Here](./lecture/notes/7-value_func.md)
8. Q Learning (advanced) [Here](./lecture/notes/8-Q_learning.md)
9. Advanced Policy Gradients [Here](./lecture/notes/9-advanced_policy_grad.md)
10. Optimal Control and Planning [Here](./lecture/notes/10-optimal_control_planning.md)
11. Model-based RL [Here](./lecture/notes/11-model-based.md)
12. Model-based RL with a Policy [Here](./lecture/notes/12-model-based-with-policy.md)
13. Exploration (1) [Here](./lecture/notes/13-exploration_1.md)
14. Exploration (2) [Here](./lecture/notes/14-exploration_2.md)
15. Offline RL (1) [Here](./lecture/notes/15-offline-RL_1.md)
16. Offline RL (2) [Here](./lecture/notes/16-offline-RL_2.md)
17. RL Theory [Here](./lecture/notes/17-RL-theory.md)
18. Variational AutoEncoder [Here](./lecture/notes/18-vae.md)
19. Soft Optimality [Here](./lecture/notes/19-soft-optimality.md)
20. Inverse RL [Here](./lecture/notes/20-IRL.md)
21. RL and Language Models [Here](./lecture/notes/21-RL-LM.md)
22. Transfer Learning and Meta Learning [Here](./lecture/notes/22-transfer-meta.md)
23. Challenges & Open Problems [Here](./lecture/notes/23-challenge.md)


# Preliminaries

学习RL，我们需要什么？

1. 一些DL的基本知识。 这个介绍DL的[repo](https://github.com/szjzc2018/dl)是一个非常好的仓库，欢迎给它点star。

2. 做好**记号混乱**的心理准备。如果学习过DL，就应该发现以下的场景是十分常见的：
    - 一个符号有多个完全不同的含义；
    - 多个符号代表完全相同的含义；
    - 前一页的符号在后一页变了；
    - 期待值不说明从哪个分布采样；
    - 多个不同的概率分布全部用记号$p$表示；
    - 还有最常见的：公式存在重要的typo。比如，在某些地方，$p^\star$和$p_\star$代表两个完全不同的意思，但又在某一处一不小心写反了。

我们会**尽量避免**这些现象发生，但必须先打好这一预防针。这也不是任何人的问题——很多试图把这些记号变得清楚整洁的尝试都会大概率因为发现公式变得长得离谱以至于令人无法忍受而告终。所以，在混乱的记号中理解它们的“深意”，这一能力也是RL即将为我们培训的，十分重要的技能之一:)

tl;dr: 接下来，是几个十分基础的介绍或问题，旨在介绍RL的基本概念，建立起一个“形象”。如果您已经了解RL的基本任务，完全可以跳过。

# What is Reinforcement Learning?

> Reinforcement learning (RL) is an **interdisciplinary area** of **machine learning** and **optimal control** concerned with how an intelligent agent **ought to take actions** in a **dynamic environment** in order to **maximize the cumulative reward**. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. (Wikipedia)

Wikipedia给出的对RL的定义十分直观。我们来总结一下，一个一般的RL问题由是**agent**（或者称为**policy**）和环境**environment**构成：
- agent能做的是在某个给定的状态（**state**）下面给出操作（**action**）；
- 每作出一个action，环境会给出一个反应，也就是环境会进入下一个状态；同时，我们认为agent会得到一个环境赋予的**reward**；
- agent的目标是最大化无限轮这样的操作之后的**cumulative reward**（当然，实际中可能会在若干轮后停止）。这一reward通常由环境给出。
- 环境的特性是如何根据state,action给出下一个state和reward。这称为环境的动力学（**dynamics**）。

我们为什么要定义这样的一个问题？当然是因为，这一抽象很好地描述了生活中的许多和人们所认同的“智能”相关的现象。
- 人们可以训练动物，使得它们做出一些特定的动作。如何训练呢？比如，小狗站起来，就喂它一把狗粮吃，否则就吓唬它一下；久而久之，小狗就会学会站起来。人们会说，“这个小狗真聪明！”——为什么？因为，它其实处理了一个很复杂的RL问题：
    - 它的action是十分复杂的，也许可以建模为大脑中成千上万神经元的刺激信号；
    - 它的dynamics更是复杂。从神经元如何将信号传输到一根根神经上，进一步控制肌肉的收缩；这再进一步和复杂的物理世界交互，由弹力、摩擦力、重力等等控制的一个极度耦合、非线性的偏微分方程进行时间演化；最终，它得到的reward只是0或1；
    - 在失败一次之后它不会放弃，因为它最大化的是累积的reward，它学会利用把历史的失败教训存在它的“参数”里，作出更好的决策，最终达到目标。
- 人类自己就更是如此。比如说，在高中，我们会根据考试的成绩或排名（某种“reward”）反思，进一步改进自己的学习方法这一policy；在人际交往中，我们也会根据对方做的事情来调整自己的行为，等等。

甚至对于新生的AI，和“环境”的交互也是“智能”最终的体现。

> 在OpenAI推出GPT3的时候，研究者们看了看它，说：“这个模型真不错”；在推出ChatGPT的时候，连你的奶奶都跑过来跟你说：“这个模型真不错”！你也看出来了——这之间是有很大的差距的。这一差距是什么？就是RL。正是RL让GPT3从一个普通的大语言模型变成一个可以和人对话、交互并了解人的喜好的"chatbot"。 ——Eric Mitchell, on the guest lecture of CS285

因此，我们更加清楚了RL这一领域的重要性。实际上，它的发展也主要集中在两个领域：
- RL作为一个算法问题，如何解决？
- 我们是否可以用RL的思想来解释人类或动物的行为？

当然，作为学习计算机的人，我们更关心的是第一个问题。本笔记就会从最基本的思想出发，带着大家一起解锁各个算法，建立一个解决RL问题的较全面的框架。但在开始之前，我们先来看几个常见的问题。

# We already have Physics. Why do we need RL (for control) anyway?

> Q: 你搞个RL模型下下棋玩玩游戏差不多得了，我不懂不多作评价。连开车、发射火箭，你也要用你那个什么算法来？你模型几亿个参数我不懂，但物理规律在哪里呢？你发射火箭的时候，大到牛顿定律、拉格朗日方程，小到空气阻力系数甚至火箭的质量，在你的一堆只有线性函数和relu的模型里哪一个体现出来了？我凭什么相信你的甚至要指明随机数种子的算法，而不相信我几千个方程式严格联立求解出来的纯物理的控制方法呢？
>
> A: 物理的方法当然合理，因为它们相当于人类完成了RL的"pretraining"。我们也无法否认物理方法在各种领域应用广泛，比如发射火箭，就可以完全通过物理的控制来完成。
>
> 但是，物理方法主要有两个问题。第一，你能否完全保证一切都在物理的控制范围之内吗？比如，对于自动驾驶，当你获得一张驾驶座前拍的一张照片，单纯的物理学并不能帮你把你前方的人、车、交通标志、道路都分清楚，确定下来。所以，这里必须就要有ML的方法介入了。
>
> 第二，就算对于完全完美的物理系统，纯粹物理的方法也不一定可行。比如，你为一只机器狗装了无数的传感器，它现在可以精准地定位它脚下沙滩上的每一颗沙子的位置、形状、大小和摩擦系数。再假设你经过研究可以高效地求解这个复杂的物理系统。但是假如突然脚下出现了一块小石头，你的系统就完全错误了。发射火箭能成功，是因为在高空意外很少见；而一旦意外出现，出一个错，动力学系统的混沌效应就会让物理的控制方法很难恢复。对于同样的问题，相反，对于大部分的RL算法，它把环境抽象为一个黑盒，发生意外情况的时候，agent会感知，并且恢复。这就是RL的优势所在。

物理和RL的关系，就像图片识别中专家方法和卷积网络的关系一样。作为二十一世纪的现代化公民，我们应该逐渐意识到神经网络的参数里蕴含的无穷能力了。

# What's the difference between RL and DL (in the root)?

RL和DL有什么区别呢？当然，我们知道，RL和环境有关，也有随机性；但它毕竟也是一个对确定的objective进行优化的过程。那么，RL和我们之前接触的DL（或者说，supervised learning和unsupervised learning）在**实质性**上真的有不同之处吗？

## From the results

在具体详细地分析之前，我们先从结果上来看，用一句话来引入：

> Generative models are impressive because the images look like something **a person might draw**; RL models are impressive because **no person had thought of their strategies before**. ——Sergey Levine, the instructor of CS285（你可以看一看他的Google Scholar首页来获得亿点震撼）

这句话很好地概括了RL和DL（在目标上）的区别。
- DL所做的是**模仿**，这件事情**有标准答案**。比如，对于generative model，它的最终任务就是学会数据集分布$p(x)$。人们夸赞它，夸赞它模仿地惟妙惟肖，就像是人一样。
- 而RL所做的是**决策**，这件事情**没有标准答案**。我们也已经看到，agent的目的是最大化cumulative reward。如何能最大化这一reward呢？每一步又该take什么action呢？这不仅没有答案，甚至在理论上都不一定是唯一的。人们夸赞RL agent，夸赞它们作出的举动是如此的“新颖”，与普通人的方法完全不同，甚至于超越了人类的认知。
    - 比如，在当年AlphaGo战胜李世石的时候，agent给出了著名的"Move 37"，当时所有的围棋专家都无法理解这一步的意义，这就是RL的魅力所在。

当然，从目标上来看，还有一个更为重要的区别：**generalization**。我们在这里暂且不描述，当我们去实践RL的具体任务的时候，我们就会真正体会到它们的差异。（如果你很好奇，可以直接去看看最后一讲的[总结](./23-challenge.md#generalization)）

## From the process

从（训练）过程上，RL和DL也有很多细节上的差异。这些差异或多或少是重要的，你也可以发现，很多RL“算法”就是为了减轻这些差异带来的问题而设计的。

- **数据来源**：
    - 在DL中，我们被给定训练数据集，我们在上面训练我们的模型；
    - 在RL中，开始我们一无所有。我们需要自己想如何和环境交互，获得数据。我们还要平衡获得新数据的过程和在老数据上面做类似DL的训练的过程，即平衡**exploration**和**exploitation**。
- **数据的关联性**：即使数据都采集完成了，训练也并不相同：
    - 在DL中，我们认为数据是i.i.d.的。可以发现很多DL算法实际上隐式地依赖（或默认了）这一点；
    - 但在RL中，数据之间是**相关**的。因为我们和环境的交互方式是，每一次环境给我们一个next state，我们就在那一个state上面继续活动。也就是说，我们采集到的数据是一条轨迹（**trajectory**）。在这样的数据里面，前后肯定是有关联的。
- **额外的和环境交互的代价**：
    - 你之后跑RL训练的时候，你会发现你的神经网络出奇的小（比如，就几千个参数），但训练又出奇的慢。为什么？实际上很大的时间花费在了`env.step(actions)`这一函数上。这一函数的作用就是，输入action，环境计算并给你next state和reward。如果你使用`gym`这个包，那么恭喜你，这一计算必须在CPU上使用`np.ndarray`进行；你还总是需要把`np.ndarray`和`torch.Tensor`相互转化，这都是你的GPU利用率很低的原因。
    - 这其实已经算好了，因为我们至少还只是用simulator在电脑上操作；如果要训练在实际世界里的机器人，那一秒就是货真价实的一秒！
    - 因此，在RL，必须注意一件事叫做**sample efficiency**，也就是和环境交互的次数不应该过多。这又会进一步和sample的质量形成tradeoff……

至此，应该可以看出DL和RL的多方面区别了。其实，这里列举的还不是全部，我们可以在具体的算法中进一步思考它们的区别。

# Wait... So what do you mean by "Deep" RL?

的确——RL实际上具有比你的想象更加悠久的历史，远远早于神经网络的发现。但是，当时人们的模型一般都是线性的。就是最近，人们才想到**把RL和DL结合起来**，用神经网络来表示policy。这就是Deep RL。

直观上，把RL搞得Deep一些肯定是有益无害；但事实上有很subtle的事情会发生。我们会在后面的笔记中进一步讨论这一点。

# Anyway... Let's get [started](./2-imitation_learning.md)!