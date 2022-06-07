### Sample Code for Simple Neural Networks with Structural and Semantic Contrastive Learning

#### Description
Graph Contrastive Learning (GCL) recently has drawn much research interest for learning generalizable node representations in a self-supervised manner. In general, the contrastive learning process in GCL is performed on top of the representations learned by a graph neural network (GNN) backbone, which transforms and propagates the node contextual information based on its local neighborhoods. However, nodes sharing similar characteristics may not always be geographically close, which poses a great challenge for unsupervised GCL efforts since they have inherent limitations in capturing such global graph knowledge. In this work, we go beyond the existing unsupervised GCL counterparts and address their limitations by proposing a simple yet effective framework -- Simple Neural Networks with Structural and Semantic Contrastive Learning (S$^3$-CL). Notably, by virtue of the proposed structural and semantic contrastive learning algorithms, even a simple neural network is able to learn expressive node representations that preserve valuable global structural and semantic patterns. Our experiments demonstrate that the node representations learned by S$^3$-CL achieve superior performance on different downstream tasks compared to the state-of-the-art GCL methods.
<div align=center><img src="https://github.com/S3CL-Code/S-3-CL/blob/main/overall.png" width="700"/></div>


#### Requirements
```
h5py==2.9.0
pandas==0.25.1
numpy==1.17.2
torch==1.7.0
torch-geometric==1.6.1
munkres==1.0.12
scipy==1.3.1
scikit_learn==0.22.1

```

We will upload more pre-trained models and trained weights soon.
