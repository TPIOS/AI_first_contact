# ADMM 学习与理解

### 1. 背景与简介

	ADMM 全称 alternating direction method of multipliers，中译名称为交替方向乘子法（初学者根本不明白这个名字到底是什么意思）。说是机器学习中比较广泛使用的约束问题最优化方法，就是带有约束条件的优化问题，比如说我们现在即将解决的问题 $|D-X^T*X|$ 的 Frobenius norm 最小值 在 X 需要保证每位至少大于等于 0 的情况下，就是这样的问题。说是什么 ALM 算法的延伸，但这并不是我们的重点。官网上给出的第一行解释为：The *alternating direction method of multipliers* (ADMM) is an algorithm that solves convex optimization problems by breaking them into smaller pieces, each of which are then easier to handle. 简单来讲可以是一个分治策略，但是这里面核心有两个，一是如何 break ；二是如何 handle smaller pieces。

### 2. 阅读正式论文之前

	在阅读正式论文之前，我想我还是先把这个算法的中文名称里面的乘子法搞清楚。