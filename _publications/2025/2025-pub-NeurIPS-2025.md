---
title:          "Boundary-Value PDEs Meet Higher-Order Differential Topology-aware GNNs"
date:           2025-10-23 00:00:00 +0800
selected:       true
pub:            "Neural Information Processing Systems (NeurIPS)"
# pub_pre:        "Submitted to "
# pub_post:       'Under review.'
# pub_last:       ' <span class="badge badge-pill badge-publication badge-success">Spotlight</span>'
pub_date:       "2025"

abstract: >-
  Recent advances in graph neural network (GNN)-based neural operators have demonstrated significant progress in solving partial differential equations (PDEs) by effectively representing computational meshes. However, most existing approaches overlook the intrinsic physical and topological meaning of higher-order elements in the mesh, which are closely tied to differential forms. In this paper, we propose a higher-order GNN framework that incorporates higher-order interactions based on discrete and finite element exterior calculus. The time-independent boundary value problems (BVPs) in electromagnetism are instantiated to illustrate the proposed framework. It can be easily generalized to other PDEs that admit differential form formulations. Moreover, the novel physics-informed loss terms, integrated form estimators, and theoretical support are derived correspondingly. Experiments show that our proposed method outperforms the existing neural operators by large margins on BVPs in electromagnetism. Our code is available at https://github.com/Supradax/Higher-Order-Differential-Topology-aware-GNN.
# cover:          /assets/images/covers/cover3.jpg
authors:
  - Yunfeng Liao
  - Yangxin Wu
  - Xiucheng Li
links:
  Paper: https://neurips.cc/virtual/2025/poster/118187
  Code: https://github.com/Supradax/Higher-Order-Differential-Topology-aware-GNN
---