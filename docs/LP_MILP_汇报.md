# LP 与 MILP：Argus Control 模块两种最优化方法

## 一、共同问题背景

Argus 的 control 层要解决的问题：
把每个 `(cell, destination)` 请求分配到某一个 PoP（入口点），
让**所有用户加权端到端延迟之和最小**，同时满足**每个 PoP 的容量约束**。

这是经典的 **Generalized Assignment Problem (GAP，广义分配问题)**。

### 统一的数学模型

```
min   Σ  d_i · c(i, p) · x_{i,p}
     i,p

s.t.  Σ  x_{i,p} = 1        对每个请求 i    （必须分配到某个 PoP）
      p

      Σ  d_i · x_{i,p} ≤ cap_p   对每个 PoP p  （容量不能超载）
      i
```

符号说明：

- `x_{i,p}`：请求 i 是否由 PoP p 承担
- `d_i`：请求 i 的带宽需求
- `c(i, p)`：请求 i 经 PoP p 的端到端延迟成本
- `cap_p`：PoP p 的容量

**LP 与 MILP 的唯一区别：`x_{i,p}` 的取值范围。**

---

## 二、LP 与 MILP 对比

| 维度 | LP | MILP |
| --- | --- | --- |
| 全称 | Linear Programming（线性规划） | Mixed Integer Linear Programming（混合整数线性规划） |
| 变量取值 | x 取 0 到 1 之间的小数 | x 只能取 0 或 1 |
| 优化目标 | 最小化 加权延迟总和 | 最小化 加权延迟总和（与 LP 相同） |
| 约束条件 | 每个请求分配到一个 PoP；每个 PoP 不超容量 | 同 LP |
| 中心思想 | 线性目标加线性约束，最优解落在凸多面体顶点 | LP 基础上强制整数，搜索加剪枝 |
| 主流算法 | Simplex 单纯形法；Interior Point 内点法 | Branch-and-Cut：分支定界加割平面 |
| 复杂度 | 多项式时间，求解快 | NP-hard，最坏指数时间 |
| 解的性质 | 分数解，作为 MILP 的下界 | 真实最优整数解 |
| Argus 用途 | 快速近似（LP 加 argmax 取整）加下界参考 | 最优基准 Ground Truth |
| 求解器 | HiGHS via scipy linprog | HiGHS via scipy milp |
| 经典应用 | 运输、调度、配餐、投资组合、网络流 | 设施选址、背包、TSP、VRP、GAP |

**关键关系**：`OPT_LP ≤ OPT_MILP` —— LP 是 MILP 放宽整数约束后的版本，所以 LP 最优值一定不差于 MILP。
