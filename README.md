# Incremental-Learning
There are implementation of incremental learning in ways of survy approachs, including：
1. Learning without forgetting(LwF), Ref: https://ieeexplore.ieee.org/abstract/document/8107520
2. Learning for large scale (BiC), Ref: https://ieeexplore.ieee.org/document/8954008
3. End-to-End incremental learning (EEIL), Ref: https://arxiv.org/abs/1807.09536
4. Our Method(Train with pseudo examplar produced by CVAE), https://hdl.handle.net/11296/6dw9k2


We set incremental learning task on 2 different dataset with  2 kinds of incremental steps independently：
1. Cifar100: 10 classes per each step (10 steps), and 50 clasess + 5 classes per each step(11steps).
2. ImageNet: 100 classes per each step (10 steps), and 500 clasess + 50 classes per each step(11steps).

In Order to compare approachs with/without old class data storage and with pseudo data, there are following settings:
1. Offline/Baseline: training  with whole classes data so far, which is used as upper bound of incrementl learning performance.
2. Sample: train with few old class sample at each step
3. NoSample: train without old class sample at each step
4. Pseudo: train with pseudo data from CVAE(Conditional Variational Auto-Encoder)




