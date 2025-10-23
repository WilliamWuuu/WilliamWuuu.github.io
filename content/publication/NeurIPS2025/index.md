---
title: 'Boundary-Value PDEs Meet Higher-Order Differential Topology-aware GNNs'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Yunfeng Liao
  - admin
  - Xiucheng Li

date: '2025-10-23T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
publishDate: '2025-10-23T00:00:00Z'

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ['paper-conference']

# Publication name and optional abbreviated publication name.
publication: In *39th Conference on Neural Information Processing Systems*
publication_short: In *NeurIPS 2025*

abstract: Recent advances in graph neural network (GNN)-based neural operators have
demonstrated significant progress in solving partial differential equations (PDEs) by
effectively representing computational meshes. However, most existing approaches
overlook the intrinsic physical and topological meaning of higher-order elements
in the mesh, which are closely tied to differential forms. In this paper, we propose a
higher-order GNN framework that incorporates higher-order interactions based on
discrete and finite element exterior calculus. The time-independent boundary value
problems (BVPs) in electromagnetism are instantiated to illustrate the proposed
framework. It can be easily generalized to other PDEs that admit differential
form formulations. Moreover, the novel physics-informed loss terms, integrated
form estimators, and theoretical support are derived correspondingly. Experiments
show that our proposed method outperforms the existing neural operators by large
margins on BVPs in electromagnetism. Our code is available at https://github.
com/Supradax/Higher-Order-Differential-Topology-aware-GNN.

tags:
  - Machine Learning
  - AI4Science
  - Partial Differential Equations

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
links:
- name: Link
  url: https://neurips.cc/virtual/2025/poster/118187

# url_pdf: ''
# url_code: ''
# url_dataset: ''
# url_poster: ''
url_project: 'https://github.com/Supradax/Higher-Order-Differential-Topology-aware-GNN'
# url_slides: ''
# url_source: ''
# url_video: ''

---
