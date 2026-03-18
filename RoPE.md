> RoPE 核心思想：通过在复数域（或 2D 平面）中对词向量进行旋转，使得两个向量的内积（Attention 分数）仅仅依赖于它们的相对位置差 $(m - n)$ 

## 1. 核心数学直觉：从 2D 平面开始

假设我们现在的词向量只有二维（$d=2$），一个 Query 向量 $q$ 和一个 Key 向量 $k$。

在没有任何位置信息时，它们的注意力打分就是简单的内积：

$$\langle q, k \rangle = q^T k$$

现在，我们要把位置信息 $m$ 注入到 $q$ 中，把位置信息 $n$ 注入到 $k$ 中。RoPE 的做法是：**不把位置信息加上去，而是把向量转起来。**

我们把这 2D 向量看作复平面上的复数：$q = |q|e^{i\alpha}$，$k = |k|e^{i\beta}$。

现在给它们分别乘以一个旋转因子（$\theta$ 是一个预设的常量角度）：

- 将 $q$ 旋转 $m\theta$ 角度：$q_m = |q|e^{i(\alpha + m\theta)}$
- 将 $k$ 旋转 $n\theta$ 角度：$k_n = |k|e^{i(\beta + n\theta)}$

神奇的事情在计算 Attention（内积）时发生了！复数的内积等于第一个数乘以第二个数的共轭：

$$\langle q_m, k_n \rangle = \text{Re}(q_m k_n^*) = \text{Re}\left( |q|e^{i(\alpha + m\theta)} \cdot |k|e^{-i(\beta + n\theta)} \right)$$

合并指数项后，结果变成了：

$$\langle q_m, k_n \rangle = |q||k| \cos(\alpha - \beta + (m - n)\theta)$$

**公式里的绝对位置 $m$ 和 $n$ 凭空消失了，只剩下了一个 $(m - n)$！** 这意味着，只要两个词的相对距离保持不变，它们之间的 Attention 分数就不会因为它们在句子中的绝对位置改变而发生任何变化。这就完美实现了“用绝对位置的计算方式，达到了相对位置的效果”。

------

## 2. 推广到多维：高维空间的“齿轮”

当然，大模型的词向量不可能是 2D 的，通常是 4096 维甚至更高。

RoPE 的处理方式非常简单粗暴：**把 $d$ 维向量两两分组，切分成 $d/2$ 个 2D 平面。**

对于位置 $m$ 的词向量 $x$，我们用一个块对角矩阵（Block Diagonal Matrix）来对它进行旋转：

$$R_m = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\ \sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\ 0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\ 0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2} \end{pmatrix}$$

然后让 $x_{rotated} = R_m x$。

这里的每一个 $\theta_i$ 频率都不同（通常也是按指数衰减 $\theta_i = 10000^{-2i/d}$）。这就好比一堆转速不同的齿轮，前面的维度转得飞快（捕捉局部、近距离的关系），后面的维度转得像蜗牛一样慢（捕捉全局、长距离的关系）。

------

## 3. 工程落地：PyTorch 中的极致性能榨取

在写代码（比如 MiniMind）时，如果你真的去构建上面那个巨大且充满大量 0 的稀疏矩阵 $R_m$ 来做矩阵乘法，那**计算量和显存开销会让你直接原地爆炸**。

所以在实际的工业界代码中，我们绝不会用矩阵乘法，而是利用三角函数的性质，将其转化为**元素级相乘（Element-wise multiplication）**：

$$R_m x = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \cos m\theta_2 \\ \cos m\theta_2 \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \sin m\theta_2 \\ \sin m\theta_2 \end{pmatrix}$$

你看这个公式里的第二项，它把原始向量相邻的元素互换了位置，并且加了负号（这在代码里通常被封装成一个叫 `rotate_half` 的函数）。这样一来，时间复杂度直接从 $O(d^2)$ 降维打击到了 $O(d)$。

这就是 RoPE 能一统大模型江湖的原因：**理论极度优雅，工程极度高效。**
