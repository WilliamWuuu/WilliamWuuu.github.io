---
title: 'Decision Tree & Random Forest'
date: 2024-11-27
permalink: /posts/2024/11/Decision Tree & Random Forest/
tags:
  - Machine Learning
  - Decision Tree
---

This is a blog of my learning notes about decision tree and random forest. Enjoy! ^_^

# Overview

- 决策树是一种依托于策略抉择而建立起来的树，他代表的是对象属性与对象值之间的一种映射关系。树中每个节点表示某个对象，而每个分叉路径则代表的某个可能的属性值，从根节点到叶节点所经历的路径对应一个判定测试序列。
- 决策树可以是二叉树或非二叉树，也可以把他看作是 if-else 规则的集合，也可以认为是在特征空间上的条件概率分布。
- 决策树在机器学习模型领域的特殊之处，在于其信息表示的清晰度。决策树通过训练获得的 “知识”，直接形成层次结构。这种结构以这样的方式保存和展示知识，即使是非专家也可以很容易地理解。

---
**决策树的优点**：
- 决策树算法中学习简单的决策规则建立决策树模型的过程非常容易理解。
- 决策树模型可以可视化，非常直观。
- 应用范围广，可用于分类和回归，而且非常容易做多类别的分类。
- 能够处理数值型和连续的样本特征。
  
---
**决策树的缺点**：
- 很容易在训练数据中生成复杂的树结构，造成过拟合。剪枝可以缓解过拟合的负作用，常用方法是限制树的高度、叶子节点中的最少样本数量。
- 学习一棵最优的决策树被认为是NP-Complete问题。实际中的决策树是基于启发式的贪心算法建立的，这种算法不能保证建立全局最优的决策树(引入随机森林随机能缓解这个问题)。
  
---
一棵树，目的和作用是决策。一般来说，每个节点上都保存了一个切分，输入数据通过切分继续访问子节点，直到叶子节点，就做出了决策。可以认为是根据数据的某个维度进行切分，不断重复这个过程。当然，如果切分的顺序不同，会得到不同的树。

叶子节点越少，往往决策树的泛化能力越高，所以可以认为训练决策树的一个目标是减少决策树的叶子节点 。

# 决策树分类算法

## 基于 ID3 算法的决策分析

- ID3 是由 J.Ross Quinlan 在1986年开发的一种基于决策树的分类算法。该算法是以信息论为基础，以信息熵核信息增益为衡量标准，从而实现对数据的归纳分类。根据信息增益运用自顶向下的贪心策略是 ID3 建立决策树的主要方法。
- 运用 ID3 算法的主要优点是建立的决策树的规模比较小，查询速度比较快。这个算法建立在“奥卡姆剃刀”的基础上，即越是小型的决策树越优于大的决策树。但是，该算法在某些情况下生成的并不是最小的树型结构。
  
---
**基本概念**
- 信息量
  
	信息量在是作为信息“多少”的度量。假设我们听到了两件事，分别如下：
		
		事件A：巴西队进入了2018世界杯决赛圈。
		事件B：中国队进入了2018世界杯决赛圈。
	
	仅凭直觉来说，事件B的信息量比事件A的信息量要大。究其原因，是因为事件A发生的概率很大，事件B发生的概率很小。所以当越不可能的事件发生了，我们获取到的信息量就越大。越可能发生的事件发生了，我们获取到的信息量就越小。那么我们可以得出以下规律：
		
		1. 信息量和事件发生的概率相关，事件发生的概率越低，传递的信息量越大。
		2. 信息量应当是非负的，必然发生的事件的信息量为零（必然事件是必然发生的，所以没有信息量。几乎不可能事件一旦发生，具有近乎无穷大的信息量）。
		3. 两个事件的信息量可以相加，并且两个独立事件的联合信息量应该是他们各自信息量的和。
	
	因此，在已知事件 $X_i$ 发生的情况下，我们定义 $X_i$ 所含有的信息量为:

	$$I(X_i)=-\log _a P(X_i)$$

	其中，如果是以 $2$ 为底数，单位是 $bit$ ；如果以 $e$ 为底数，单位是 $nat$ ；如果以 $10$ 为底数，单位是 $det$ 。

- 信息熵
  
	信息熵是接受信息量的平均值，用于确定信息的不确定程度，是随机变量的均值。信息熵越大，信息就越凌乱或传输的信息越多。处理信息是一个让信息的熵减少的过程。
	
	假设 $X$ 是一个离散的随机变量，且它的取值范围是 $x_1,x_2,\dots,x_n$ ，那么 $X$ 的熵定义为：

	$$H(X)=\sum\limits _i P(x_i)I(x_i)=-\sum\limits _iP(x_i)\log _2 P(x_i)$$

- 条件熵
  
	在决策树的切分里，事件 $x_i$ 可以认为是在样本中出现某个标签/决策。于是 $P(x_i)$ 可以用所有样本中某个标签出现的频率来代替。但我们求熵是为了决定采用哪一个维度进行切分，因此有一个新的概念条件熵：

  $$H(X \vert Y)=\sum\limits _{y\in Y}p(y)H(x \vert Y=y)$$

	这里我们认为 $Y$ 就是用某个维度进行切分，那么 $y$ 就是切成的某个子集合于是 $H(X \vert Y=y)$ 就是这个子集的熵。因此可以认为条件熵是每个子集合的熵的一个加权平均/期望。

- 信息增益
  
	信息熵表示的是不确定度。均匀分布时，不确定度最大，熵就最大。当选择某个特征对数据集进行分类时，分类后的数据集信息熵会比分类前的小，其差值表示为信息增益。信息增益用于度量属性 $A$ 对降低样本集合 $X$ 熵的贡献大小。信息增益可以衡量某个特征对分类结果的影响大小。信息增益越大，越适用对X进行分析。
	
	有了信息熵的定义后，信息增益的概念便很好理解了，表示分类后，数据整体信息熵的差值。我们假设特征集中有一个离散特征 $a$，它有 $V$ 个可能的取值 $a_1,a_2,…,a_V$ ，如果使用特征 $a$ 来对样本 $D$ 进行划分，那么会产 $V$ 个分支节点，其中第 $v$ 个分支节点中包含的样本集。我们记为 $D^v$。于是，可计算出特征 $a$ 对样本集 $D$ 进行划分所获得的信息增益为：

	$$Gain(D,a)=H(D)-H(D \vert a)=H(D)-\sum\limits _{v=1}^V\frac{\lvert D^v \rvert}{\lvert D \rvert}H(D^v)$$

	即特征 $a$ 对样本集 $D$ 进行划分所获得的信息增益为样本集 $D$ 的信息熵减去经过划分后各个分支的信息熵之和。由于每个分支节点，所包含的样本数不同，所有在计算每个分支的信息熵时，需要乘上对应权重 $\frac{\lvert D^v \rvert}{\lvert D \rvert}$ ，即样本数越多的分支节点对应的影响越大。

---
**算法流程**

$\rightarrow$ 对当前样本集合计算出所有属性信息的信息增益。

$\rightarrow$ 先选择信息增益最大的属性作为测试属性，将测试属性相同的样本转化为同一个子样本。

$\rightarrow$ 若子样本集的类别属性只含有单个属性，则分支为叶子节点，判断其属性值并标上相应的符号，然后返回调用处，否则对子样本递归调用本算法。

---
**优缺点**
- 优点
> 1. 结构简单；
> 2. 清晰易懂；
> 3. 灵活方便；
> 4. 不存在无解的危险；
> 5. 可以利用全部训练例的统计性质进行决策，从而抵抗噪音。

- 缺点
> 1. ID3算法采用信息增益来选择最优划分特征，然而人们发现，信息增益倾向与取值较多的特征，对于这种具有明显倾向性的属性，往往容易导致结果误差；
> 2. ID3算法没有考虑连续值，对与连续值的特征无法进行划分；
> 3. ID3算法无法处理有缺失值的数据；
> 4. ID3算法没有考虑过拟合的问题，而在决策树中，过拟合是很容易发生的；
> 5. ID3算法采用贪心算法，每次划分都是考虑局部最优化，而局部最优化并不是全局最优化，当然这一缺点也是决策树的缺点，获得最优决策树本身就是一个NP难题，所以只能采用局部最优。

## 基于 C4.5 算法的决策分析

C4.5 是 J.Ross Quinlan 基于 ID3 算法改进后得到的另有一个分类决策树算法。C4.5 算法继承了 ID3 算法的优点，且改进后的算法产生的分类规则易于理解，准确率高。同时，该算法也存在一些缺点，如算法效率低，值仅适合用于能够驻留于内存的数据集。正在 ID3 算法的基础上，C4.5 算法进行了以下几点改进：
- 用信息增益率来选择属性，克服了ID3算法选择属性时偏向选择取值多的属性的不足
- 在决策树的构造过程中进行剪枝，不考虑某些具有很好元素的节点。
- 能够完成对联系属性的离散化处理。
- 能够对不完整数据进行处理。

那么以信息增益作为准则来进行划分属性有什么缺点？
	
假设有14个样本的数据，增加一列特征 $data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]$，根据公式，可以算出对应的信息增益为：$Gain(D,data) = 0.940$。
	
可见，它所对应的信息增益远大于其他特征，所以我们要以 $data$ 特征作为第一个节点的划分。这样划分的话，将产生14个分支，每个分支对应只包含一个样本，每个分支节点的纯度已达到最大，也就是说每个分支节点的结果都非常确定。但是，这样的决策树肯定不是我们想要的，因为它根本不具备任何泛化能力。

为了避免这个问题，我们引入增益率这个概念，改为使用增益率来作为最优划分特征的标准。

---
**信息增益率**
	
同样假设有数据集 $D$，以及特征 $a$，它有 $V$ 个可能的取值 $a^1,a^2,\dots,a^V$ ，如果数据集 $D$ 在以特征 $a$ 作为划分特征时，增益率定义为:

$$Gainratio(D,a)=\frac{Gain(D,a)}{IV(a)}$$

其中：

$$IV(a)=-\sum\limits^{V}_{v=1}\frac{|D^v|}{|D|}\log _2\frac{|D^v|}{|D|}$$
	
上述增益类公式中，就是特征a本身的信息熵，也就是对应根据特征a的可能取值，所对应求得的信息熵。特征a对应的种类越多，则其值通常可能会越大，从而增益率越小。这样就可以避免信息增益中队可取数目比较多的特征有所偏好的缺点。
	
然而增益率又会便好取值数目比较少的属性，因此又有一个启发性规则：先从候选划分属性中找出信息增益高于平均水平的属性，再从其中选择增益率最高的。
## 基于分类回归树 (CART) 的决策划分
在数据挖掘中，决策树主要有两种类似：分类树和回归树。分类树的输出是样本的类别，回归树的输出是一个实数。分类和回归树，即 CART (Classification And Regression Tree)，最先由 Breiman 等提出，也属于一类决策树。CART算法由决策树生成和决策树剪枝两部分组成：
- 决策树生成：基于训练数据集生成决策树，生成的决策树要尽量的大
- 决策树剪枝：用验证数据集对以生成的树进行剪枝并选择最优子树，这时用损失函数最小作为剪枝的标准。

CART算法既可以用于创建分类树，也可以用于创建回归树。CART算法的重要特点包含以下三个方面：
- 二分 (Binary Split)：在每次判断过程中，都是对样本数据进行二分。CART 算法是一种二分递归分割技术，把当前样本划分为两个子样本，使得生成的每个非叶子结点都有两个分支，因此CART算法生成的决策树是结构简洁的二叉树。由于 CART 算法构成的是一个二叉树，它在每一步的决策时只能是“是”或者“否”，即使一个 feature 有多个取值，也是把数据分为两部分
- 单变量分割 (Split Based on One Variable)：每次最优划分都是针对单个变量。
- 剪枝策略：CART 算法的关键点，也是整个 Tree-Based 算法的关键步骤。剪枝过程特别重要，所以在最优决策树生成过程中占有重要地位。有研究表明，剪枝过程要比树的生成过程更为重要，对于不同的划分标准生成的最大树 (Maximum Tree)，在剪枝之后都能够保留最重要的属性划分，差别不大。反而是剪枝方法对于最优树的生成更为关键。

CART树生成就是递归的构建二叉决策树的过程，对回归使用平方误差最小化准则，对于分类树使用基尼指数 (Gini index) 准则，进行特征选择，生成二叉树。

---
**基尼系数**
基尼系数同样也是表述混乱的程度。我们用基尼指数 $Gini(D)$ 表示集合 $D$ 的不确定性，基尼指数 $Gini(D,A)$ 表示数据集D经过特征A划分以后集合D的不确定性。基尼指数越大说明我们的集合不确定性就越大。

给定样本集 $D$，假设有 $K$ 个类，样本点属于第 $k$ 个类的概率为 $p_k$ ，则概率分布的基尼指数定义为：

$$Gini(D)=\sum\limits^K_{k=1}p_k(1-p_k)=1-\sum\limits^K_{k=1}p_k^2$$

根据基尼指数定义，可以得到样本集合D的基尼指数：

$$Gini(D)=1-\sum\limits^K_{k=1}(\frac{|C_k|}{|D|})^2$$

其中 $C_k$ 表示数据集D中属于第k类的样本子集。 

基尼值反映了从数据集随机抽取两个样本，其类别标记不一致的概率。因此基尼值越小，数据集纯度越高。

对属性 $a$ 进行划分，则属性 $a$ 的基尼系数定义为：

$$Gini\_ index(D,a)=\sum\limits^V_{v=1}\frac{|D^v|}{|D|}Gini(D^v)$$

如果数据集 $D$ 根据特征 $A$ 在某一取值 $a$ 上进行分割，得到 $D_1, D_2$ 两部分后，那么在特征 $A$ 下集合 $D$ 的基尼系数：

$$Gini\_ index(D,a)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)$$

于是，我们在候选属性集合 $A$ 中，选择使得划分后基尼系数最小的属性作为最优划分属性，即：

$$a_*=\arg\limits_{a\in A}\min Gini\_ index(D,a)$$

其中算法的停止条件有：
- 节点的样本个数小于预定阈值。
- 样本集的 $Gini$ 系数小于预定阈值 (此时样本基本属于同一类)，或基本没有更多特征。

## 基于随机森林的决策分类

- 随机森林是包含多个决策树的分类器。随机森林算法是由 Leo Breiman 和 Adele Cutler 发展推论出的。随机森林由很多决策树组成，且这些决策树之间*没有关联*。
- 随机森林就是通过集成学习的思想将多棵树集成的一种算法，它的基本单元是决策树，而它的本质属于机器学习的一个分支——集成学习 (Ensemble Learning) 方法。集成学习就是使用一系列学习器进行学习，并将各个学习方法通过某种特定的规则进行整合，以获得比单个学习器更好的学习效果的一种机器学习方法。集成学习通过建立几个模型，并将它们组合来解决单一预测问题。它的工作原理主要是生成多个分类器或模型，各自独立的学习和做出预测。
- 随机森林是由多棵决策树构成的。对于每棵树，它们使用的训练集是采用放回的方式从总的训练集中采样出来的。而在训练每棵树的节点时，使用特征是从所有特征中，采用按照一定比例随机的无回放的方式抽取的。

---
**Bagging算法(Bootstrap aggregating)**
- Bagging 是由 Leo Breiman 于1994年提出的一种集成学习算法。Bagging 算法可与其他分类、回归算法结合，提高其准确率、稳定性的同时，透过降低结果的变异数，避免过拟合的发生。

- 给定一个大小为 $n$ 的训练集 $D$，Bagging 算法从中均匀、有放回地 (Bootstrap) 选出大小为 $m$ 的子集 $D_i$ 作为新的训练集。在这 $m$ 个训练集上使用分类、回归等算法，则可得到 $m$ 个模型 (弱学习器)，再透过取平均值、取多数票等结合策略，即可得到 Bagging 的结果 (强学习器)。
- 一次含 $m$ 个样本的训练集的随机采样中，一个样本每次被采集到的概率是 $\frac{1}{m}$。一个样本经过 $m$ 次采样都没有被采中的概率 $P=(1-\frac{1}{m})^m$。当 $m\rightarrow \infty$时， $P=\frac{1}{e}\approx 0.368$ 。也就是说，在 Bagging 的每轮随机采样中，训练集中大约有 36.8% 的数据没有被采样集采集中，我们常常称之为袋外数据(Out Of Bag, OOB)。这些数据没有参与训练集模型的拟合，因此可以用来检测模型的泛化能力。

- Bagging 的集合策略也比较简单，对于分类问题，通常使用简单投票法，得到最多票数的类别或者类别之一为最终的模型输出。对于回归问题，通常使用简单平均法，对多个弱学习器得到的回归结果进行算术平均得到最终的模型输出。
- 由于Bagging算法每次都进行采样来训练模型，因此泛化能力很强，对于降低模型的方差很有作用。当然对于训练集的拟合程度就会差一些，也就是模型的偏倚会大一些。

---
**Random Forest**
- 根据 Bagging算法的思想， 我们将 CART 决策树作为弱学习器。对于普通的决策树，我们会在节点上所有的 $n$ 个样本特征中选择一个最优的特征来做决策树的左右子树划分，但是 RF 通过随机选择节点上的一部分样本特征，这个数字小于 $n$，假设为 $n_{sub}$ 。然后在这些随机选择的 $n_{sub}$ 个样本特征中，选择一个最优的特征来做决策树的左右子树划分。这样进一步增强了模型的泛化能力。减小 $n_{sub}$ ，则模型的方差变小，偏差变大。 反之则模型的方差变大，偏差变小。
- 随机森林背后的思想是，每棵树的预测可能都相对较好，但可能对部分数据过拟合。如果构造很多树，并且每棵树的预测都很好，但都以不同的方式过拟合，那么我们可以对这些树的结果取平均值来降低过拟合。既能减少过拟合又能保持树的预测能力，这可以在数学上严格证明。
- 为了实现这一策略，我们需要构造许多决策树。每棵树都应该对目标值做出可以接受的预测，还应该与其他树不同。随机森林的名字来自于将随机性添加到树的构造过程中，以确保每棵树都各不相同。随机森林中树的随机化方法有两种：一种是通过选择用于构造树的数据点，另一种是通过选择每次划分测试的特征。我们来更深入地研究这一过程。

- 构造随机森林
  1. 确定森林中树的个数。这些树在构造时彼此完全独立，算法对每棵树进行不同的随机选择，以确保树和树的训练集之间是有区别的。
  2. 接下来，基于这个新创建的数据集来构造决策树。但是，要对我们在介绍决策树时描述的算法稍作修改。在每个结点处，算法随机选择特征的一个子集，并对其中一个特征寻找最佳测试，而不是对每个结点都寻找最佳测试。选择的特征个数由 max_features 参数来控制。每个结点中特征子集的选择是相互独立的，这样树的每个结点可以使用特征的不同子集来做出决策。由于使用了自助采样，随机森林中构造每棵决策树的数据集都是略有不同的。由于每个结点的特征选择，每棵树中的每次划分都是基于特征的不同子集。这两种方法共同保证随机森林中所有树都不相同。

- 在这个过程中的一个关键参数是 max_features。如果我们设置 max_features 等于 n_features，那么每次划分都要考虑数据集的所有特征，在特征选择的过程中没有添加随机性（不过自助采样依然存在随机性）。如果设置 max_features 等于1，那么在划分时将无法选择对哪个特征进行测试，只能对随机选择的某个特征搜索不同的阈值。因此，如果 max_features 较大，那么随机森林中的树将会十分相似，利用最独特的特征可以轻松拟合数据。如果 max_features 较小，那么随机森林中的树将会差异很大，为了很好地拟合数据，每棵树的深度都要很大。

---
**优缺点**
- *优点*
> 1. 随机森林算法能解决分类与回归两种类型的问题，并在这两个方面都有相当好的估计表现；
> 2. 随机森林对于高维数据集的处理能力令人兴奋，它可以处理成千上万的输入变量，并确定最重要的变量，因此被认为是一个不错的降维方法。此外，该模型能够输出变量的重要性程度，这是一个非常便利的功能。
> 3. 在对缺失数据进行估计时，随机森林是一个十分有效的方法。就算存在大量的数据缺失，随机森林也能较好地保持精确性。
> 4. 当存在分类不平衡的情况时，随机森林能够提供平衡数据集误差的有效方法。
> 5. 模型的上述性能可以被扩展运用到未标记的数据集中，用于引导无监督聚类、数据透视和异常检测；
> 6. 随机森林算法中包含了对输入数据的重复自抽样过程，即所谓的 bootstrap 抽样。这样一来，数据集中大约三分之一将没有用于模型的训练而是用于测试，这样的数据被称为 out of bag samples，通过这些样本估计的误差被称为 out of bag error。
> 7. 研究表明，这种 out of bag 方法的与测试集规模同训练集一致的估计方法有着相同的精确程度，因此在随机森林中我们无需再对测试集进行另外的设置。
> 8. 训练速度快，容易做成并行化方法

- *缺点*
> 1. 随机森林在解决回归问题时并没有像它在分类中表现的那么好，这是因为它并不能给出一个连续型的输出。当进行回归时，随机森林不能够作出超越训练集数据范围的预测，这可能导致在对某些还有特定噪声的数据进行建模时出现过度拟合。
> 2. 对于许多统计建模者来说，随机森林给人的感觉像是一个黑盒子——你几乎无法控制模型内部的运行，只能在不同的参数和随机种子之间进行尝试。

---
**随机森林的构造方法**
- 随机森林的建立基本由随机采样和完全分裂两部分组成。
	1. 随机采样
		随机森林对输入的数据进行行、列的采样，但两个采样的方法有所不同。对于行采样，采用的方法是有回放的采样，即在采样得到的样本集合中，可能会有重复的样本。假设输入样本为 $N$ 个，那么采样的样本也是 $N$ 个，这样使得在训练时，每科树的输入样本都不是全部的样本，所以相对不容易出现 over-fitting。对于列采样，采用的方式是按照一定的比例无放回的抽取，从 $M$ 个 feature 中，选择 m 个样本 ($n<<M$)。
	
	2. 完全分裂
		在形成决策树的过程中，决策树的每个节点都要按完全分裂的方式来分裂，直到节点不能再分裂。采用这种方式建立出的决策树的某一叶子节点要么是无法继续分裂的，要么里面的所有样本都是指向同一个分类。

- 接下来介绍每科树的构造方法，步骤如下：
	
	$\rightarrow$ 用 $N$ 表示训练集的个数，M表示变量的数目
	
	$\rightarrow$ 用 $m$ 来表示当在一个节点上做决定时会用到的变量的数量
	
	$\rightarrow$ 从 $N$ 个训练案例中采用可重复取样的方式，取样 $N$ 次，形成一组训练集，并使用这棵树来对剩余变量预测其类别，并对误差进行计算。
	
	$\rightarrow$ 对于每个节点，随机选择 $m$ 个基于词典上的变量。根据这 $m$ 个变量，计算其最佳的分割方式。
	
	$\rightarrow$ 对于森林中的每棵树都不用采用剪枝技术，每棵树都能完整生长。

- 森林中任意两棵的相关性与森林中棵树的分类能力是影响随机森林分类效果(误差率)的两个重要因素。任意两棵树之间的相关性越大，错误率越大，每棵树的分类能力越强，整个森林的错误率越低。
## 决策树的剪枝
- 决策树很容易发生过拟合，可以改善的方法有：
> 1. 通过阈值控制终止条件，避免树形结构分支过细。
> 2. 通过对已经形成的决策树进行剪枝来避免过拟合。
> 3. 基于Bootstrap的思想建立随机森林。

---
**剪枝的分类**
- 预剪枝
	
	预剪枝即是指在决策树的构造过程中，对每个节点在划分前需要根据不同的指标进行估计，如果已经满足对应指标了，则不再进行划分，否则继续划分。
	
	树的增长不能是无限的，因此需要设定一些条件，若树的增长触发某个设定条件时，则树的增长需要进行停止继续。这些设定的条件称作停止条件 (Stopping Criteria), 常用的 停止条件如下：
	> 1. 直接指定树的深度
	> 2. 直接指定叶子节点个数
	> 3. 直接指定叶子节点的样本数
	> 4. 对应的信息增益量
	> 5. 拿验证集中的数据进行验证，看分割前后，精度是否有提高。
	
	由于预剪枝是在构建决策树的同时进行剪枝处理，所以其训练时间开销较少，同时可以有效的降低过拟合的风险。
	
	但是，预剪枝可能会给决策树带来欠拟合的风险。1，2，3，4指标不用过多解释，对于5指标来说，虽然当前划分不能导致性能提高，但是在此基础上的后续划分，完全有可能使性能显著提高(全局最优不一定每一个局部都最优)。

- 后剪枝
	
	后剪枝先根据训练集生成一颗完整的决策树，然后根据相关方法进行剪枝。
	
	常用的一种方法是，自底向上对非叶子节点进行考察，同样拿验证集中的数据，来根据精度进行考察。看该节点划分前和划分后，精度是否有提高，如果划分后精度没有提高，则剪掉此子树，将其替换为叶子节点。
	
	相对于预剪枝来说，后剪枝的欠拟合风险很小，同时，泛化能力往往要优于预剪枝，但是，因为后剪枝先要生成整个决策树后，然后才自底向上依次考察每个非叶子节点，所以训练时间长。

---
**后剪枝算法**
- Reduced-Error Pruning (错误率降低剪枝，REP)
	
	REP 算法是最简单的后剪枝方法之一，它需要使用一个剪枝验证集来对决策树进行剪枝， 用训练集合来训练数据。 通常取出可用样例的三分之一用作验证集合，用剩余三分之二作训练集合。 
	
	决定是否修剪某个结点的步骤如下：
	> 1. 删除以该结点作为根结点的子树。
	> 2. 使该结点成为叶子结点。
	> 3. 赋予该结点关联的训练数据的最常见分类。
	> 4. 当修剪后的树对于验证集合的性能与原来的树相同或优于原来的树时，该结点才真正被删除。
	
	利用训练集合过拟合的性质，使训练集合数据能够对其进行修正，反复进行上述步骤，采用自底向上的方法处理结点，将那些能够最大限度地提高验证集合的精度的结点删去，直到进一步修剪有害（修剪会减低验证集合的精度）为止。
	
	在数据量较少的情况下很少应用REP方法。该方法趋于过拟合，这是因为训练数据集中存在的特性在剪枝过程中都被忽略，当剪枝数据集比训练数据集小得多时这个问题需要特别注意。

- Pessimistic Error Pruning (悲观剪枝，PEP)  (用于 C4.5)
	
	PEP 剪枝算法是在 C4.5 决策树算法中提出的，该方法基于训练数据的误差评估，因此比起 REP 剪枝法，它不需要一个单独的测试数据集。但训练数据也带来错分误差偏向于训练集，因此需要加入修正 $\frac{1}{2}$ (惩罚因子，用常数 $0.5$)，是自上而下的修剪。 之所以叫“悲观”，可能正是因为每个叶子结点都会主观加入一个惩罚因子，“悲观”地提高误判率。
	
	悲观错误剪枝法是根据剪枝前后的误判率来判定子树的修剪。该方法引入了统计学上连续修正的概念弥补 REP 中的缺陷，在评价子树的训练错误公式中添加了一个常数，假定每个叶子结点都自动对实例的某个部分进行错误的分类。
	
	把一颗子树(具有多个叶子节点)的分类用一个叶子节点来替代的话，在训练集上的误判率肯定是上升的，但是在新数据上不一定。于是我们需要把子树的误判计算加上一个经验性的惩罚因子。对于一颗叶子节点，它覆盖了 $N$ 个样本，其中有 $E$ 个错误，那么该叶子节点的误判率为 $\frac{E+0.5}{N}$，这个 $0.5$ 就是惩罚因子。
	
	那么对于一颗拥有 $L$ 个叶子结点的子树，该子树的误判率估计为：

	$$e=\sum\limits_{i\in L}\frac{E_i+0.5}{N_i}=\frac{\sum\limits_{i\in L}(E_i+0.5)}{\sum\limits_{i\in L}N_i}$$

	这样的话，我们可以看到一颗子树虽然具有多个子节点，但由于加上了惩罚因子，所以子树的误判率计算未必占到便宜。剪枝后内部节点变成了叶子节点，其误判个数 $J$ 也需要加上一个惩罚因子，变成 $J+0.5$。使用训练数据，子树总是比替换为一个叶节点后产生的误差小，但是使用修正后有误差计算方法却并非如此，当子树的误判个数大过对应叶节点的误判个数一个标准差之后，就决定剪枝，即满足：

	$$E(SubErrCount)-var(SubErrCount)>E(LeafErrCount)$$

	这里出现的标准差如何计算呢？
	
	我们假定一棵子树错误分类一个样本的值为 $1$，正确分类一个样本的值为 $0$，该子树错误分类的概率(误判率)为 $e$，则每分类一个样本都可以近似看作是一次伯努利试验，覆盖 $N$ 个样本的话就是做 $N$ 次独立的伯努利试验，因此，我们可以把子树误判次数近似看成是服从 $B(N, e)$ 的二项分布。因此，我们很容易估计出子该树误判次数均值和标准差：

	$$\exp(SubtreeErrCount)=N\times e$$

	$$var(SubtreeErrCount)=\sqrt{N\times e\times (1-e)}$$

	当然并不一定非要大一个标准差，可以给定任意的置信区间，我们设定一定的显著性因子，就可以估算出误判次数的上下界。对于给定的置信区间，采用下界估计作为规则性能的度量。这样做的结果是对于大的数据集合，该剪枝策略能够非常接近观察精度，随着数据集合的减小，离观察精度越来越远。该剪枝方法尽管不是统计有效的，但是在实践中有效。
	
	PEP 采用自顶向下的方式 将符合上述不等式的非叶子结点裁剪掉。该算法看作目前决策树后剪枝算法中精度比较高的算法之一，同时该算法仍存在一些缺陷。首先，PEP算法是唯一使用自顶向上剪枝策略的后剪枝算法，但这样的方法有时会导致某些不该被剪掉的某结点的子结点被剪掉。虽然PEP方法存在一些局限性，但是在实际应用中表现出了较高的精度。

- Minimum Error Pruning (最小误差剪枝，MEP)
	
	一棵树的好坏用如下式子衡量：

	$$W_{\alpha}(T)=W(T)+\alpha C(T)$$

	其中 $W(T)$ 表示该树误差的衡量； $C(T)$ 表示对树大小的衡量(可以用树的终端节点个数代表)； $\alpha$ 表示两者的平衡系数，其值越大树越小，反之树越大。
	
	为了利用该准则来进行剪枝主要有如下两个步骤：
		
	$\rightarrow$ 找到完整树的一些子树{ $T_i \vert i=1,2,3,\dots,m$ }
		
	$\rightarrow$ 分别计算出每棵树的 $W_{\alpha}(T_i)$，选择书中具有最小的 $W_{\alpha}(T_i)$ 的树。