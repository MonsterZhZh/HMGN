# HMGN
Hierarchical Multi-Granularity Classification based on Residual Gated GCN

## Paper

### [Preprint Version](https://arxiv.org/pdf/2201.03194v2.pdf)
In this paper, we investigate hierarchical multi-granularity classification with training samples labeled at different levels. Accordingly, we propose the combinatorial loss and a hierarchical residual network ([HRN](https://github.com/MonsterZhZh/HRN)) for hierarchical feature interaction.
This repo implements an alternative hierarchical feature interaction network (HMGN) based on the same combinatorial loss.

### Network Architecture
![avatar](GatedGCN.jpg)
1. The trunk net (ResNet-50) produces the feature map;
2. Pre-trained GloVe model generates word vectors for each class in the hierarchy;
3. Word vectors interact with the feature map using a [low-rank bilinear pooling method](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Learning_Semantic-Specific_Graph_Representation_for_Multi-Label_Image_Recognition_ICCV_2019_paper.pdf) to generate semantic guided attention coefficients;
4. We perform weighted average pooling over all locations in the feature map to obtain the initial feature vector for each node in the graph;
5. In the graph, each node represents a class in the hierarchy, and children nodes connect to their parents with undirected edges; 
6. We adopt the [residual gated GCN](https://arxiv.org/pdf/1711.07553.pdf) to perform feature interaction between nodes;
7. The node classifier performs element-wise multiplication with the matrix formed by the feature vectors of each node. We perform average pooling for each class on the results to output the final vector used for classification.

## Requirements
* Python 3.7
* Pytorch 1.3.1
* torchvision 0.4.2
* networkx 2.3
* CUDA 10.2

## Experiments
We perform experiments on CUB-200-2011 and compare two different hierarhical networks with the same combinatorial loss. Supporting files can be found in the related repo: [HRN](https://github.com/MonsterZhZh/HRN).

OA (%) results on CUB-200-2011 with the relabeling proportion 0% by comparing two hierarhical networks:
| Levels     | HRN | HMGN |
| ----------- | ----------- |----------- |
| Orders      | 98.67       |98.74 |
| Families   | 95.51        |95.27 |
| Species   | 86.60        |85.27 |
 


## References
1. [Benchmarking graph neural networks](https://arxiv.org/pdf/2003.00982.pdf)
