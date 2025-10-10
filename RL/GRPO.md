# 组相对策略优化（GRPO）

在对大模型进行微调的环节中，强化学习具有不可替代的重要作用。当前，广泛应用的近端策略优化算法（PPO）在应用大规模模型时，遭遇了沉重的计算和存储压力。鉴于PPO算法需要构建一个与策略模型规模大体相当的价值网络来对优势函数进行评估，这便引发了显著的内存（显存）成本和计算成本。

而且PPO算法在更新策略的过程中，极有可能导致策略分布产生剧烈波动，进而波及训练的稳定性。鉴于上述种种问题，DeepSeek提出了一种创新的强化学习算法——组相对策略优化算法（GRPO）。其目标在于降低对价值网络的依赖程度，与此同时确保策略更新的稳定性与高效性。

![img](assets/v2-88f04c9c365012f4c053aab965084216_1440w.jpg)

从上图可以看出，GRPO减少了价值函数，有别与PPO需要像那样添加额外的价值函数近似，转而直接采用多个采样输出的平均奖励当作Baseline，这使得训练资源的使用量得到了显著削减。

去除Value Function，Reward直接对单个Q生成的Response进行打分，归一化后，作为替代的优势函数。

$$\begin{aligned}\mathcal{J}_{GRPO}(\theta)&=\mathbb{E}[q\sim P(Q),\{o_i\}_{i=1}^G\sim\pi_{\theta_{old}}(O|q)]\\&\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left\{\min\left[\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta,id}(o_{i,t}|q,o_{i,<t})}\hat{A}_{i,t},clip\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{odd}}(o_{i,t}|q,o_{i,<t})},1-\epsilon,1+\epsilon\right)\hat{A}_{i,t}\right]-\beta\mathbb{D}_{KL}[\pi_{\theta}||\pi_{ref}]\right.\end{aligned}$$

