---
title:          "Boundary-Value PDEs Meet Higher-Order Differential Topology-aware GNNs"
title_zh:       "边值偏微分方程与高阶微分拓扑感知图神经网络"
date:           2025-10-23 00:00:00 +0800
selected:       true
pub:            "Neural Information Processing Systems (NeurIPS)"
# pub_pre:        "Submitted to "
# pub_post:       'Under review.'
pub_last:       ' <span class="badge badge-pill badge-publication badge-success">Spotlight</span>'
pub_date:       "2025"

abstract: >-
  Recent advances in graph neural network (GNN)-based neural operators have demonstrated significant progress in solving partial differential equations (PDEs) by effectively representing computational meshes. However, most existing approaches overlook the intrinsic physical and topological meaning of higher-order elements in the mesh, which are closely tied to differential forms. In this paper, we propose a higher-order GNN framework that incorporates higher-order interactions based on discrete and finite element exterior calculus. The time-independent boundary value problems (BVPs) in electromagnetism are instantiated to illustrate the proposed framework. It can be easily generalized to other PDEs that admit differential form formulations. Moreover, the novel physics-informed loss terms, integrated form estimators, and theoretical support are derived correspondingly. Experiments show that our proposed method outperforms the existing neural operators by large margins on BVPs in electromagnetism. Our code is available at https://github.com/Supradax/Higher-Order-Differential-Topology-aware-GNN.
abstract_zh: >-
  近年来，基于图神经网络（GNN）的神经算子通过有效表示计算网格，在求解偏微分方程（PDE）方面取得了显著进展。然而，现有方法大多忽略了网格中高阶元素所蕴含、且与微分形式密切相关的物理与拓扑意义。本文基于离散外微分与有限元外微分理论，提出了一个能够建模高阶交互的高阶 GNN 框架，并以电磁学中的时不变边值问题（BVP）为例进行说明。该框架可以自然推广到其他具有微分形式表述的 PDE。我们进一步推导了相应的物理信息损失、积分形式估计器及理论支撑。实验表明，在电磁学边值问题上，该方法显著优于现有神经算子。
# cover:          /assets/images/covers/cover3.jpg
authors:
  - Yunfeng Liao
  - Yangxin Wu
  - Xiucheng Li
links:
  Paper: https://neurips.cc/virtual/2025/poster/118187
  Code: https://github.com/Supradax/Higher-Order-Differential-Topology-aware-GNN
---
