 ### 必须新增的实验

  1. Test-gradient convergence

  现在论文的 (n_t) 是固定的：219M 用 128，1.14B/LLaDA 用 64；现有曲线只改变 (n_c)。因此不能回答“排序是否是有限 (F_{\rm test}) 的噪声”。

  建议冻结同一个 calibration surrogate，使用嵌套的：

  [
  n_t\in{16,32,64,128,256}
  ]

  至少报告：

  - rank-1 和 diagonal 的 test error；
  - paired margin (e_{\rm diag}-e_{\rm rank1})；
  - 排序在多大 (n_t) 后稳定；
  - 多个独立 test/mask seeds，而不仅是单个 nested prefix。

  R12/R17 保存的 64×64 test Gram 最多能做部分 sanity check；它们没有逐样本的 diagonal contraction，因此不能完整恢复不同 (n_t) 下的 rank-1/diagonal 排序。该实验需要重新算梯度，但完全不需要模型训练。

  2. 实际更新误差与 forgetting：合并成一次实验

  不要分别做两套实验。在同一个 matched adaptation run 中同时记录实际更新和最终 forgetting：

  [
  \delta^\top F_{\rm test}\delta
  =\frac1{n_t}\sum_j(h_j^\top\delta)^2,
  ]

  矩阵无需显式构造。对 rank-1 与 diagonal 的 (\delta^\top Q\delta) 也可直接计算。

  除了审稿人要求的绝对误差，最好同时报告：

  [
  r_\delta(Q)=
  \frac{|\delta^\top(F_{\rm test}-Q)\delta|}
  {\delta^\top F_{\rm test}\delta+\epsilon},
  \qquad
  \frac{\delta^\top Q\delta}
  {\delta^\top F_{\rm test}\delta+\epsilon}.
  ]

  第二项可以看出 (Q) 是过度保护还是保护不足。

  关键是主比较必须在同一个 common update 上，例如相同 GD/replay 产生的 (\delta_{\rm GD})。如果 rank-1 用自己的 (\delta_{\rm rank1})，diagonal 用自己的 (\delta_{\rm diag})，那么 (Q) 同时改变了评价方
  向，会产生内生混淆。方法自己的更新可以作为补充结果。

  现有 GSM8K formal runs 已有相同 checkpoint、Fisher/replay caches 和 forgetting endpoint，因而不必从零搭建。但它们没有保存足够的最终 (\delta) 或 (\delta^\top F_{\rm test}\delta)，所以大概率需要重跑
  adaptation，或者直接接入正在运行的新实验。

  已有的 realized EWC term——rank-1 为 2.9567、diagonal 为 0.0020——只能说明“trace matching 不等于实际约束相同”，不能替代 reviewer 要求的 (F-Q) 指标。相关结果在 GSM8K report (iclr_1/report/
  gsm8k_rank1_behavior_results.md:100)。

  ### Fidelity–forgetting 关联需要避免的错误

  当前的 synthetic continual matrix 和 GSM8K retention 只能作为上下文，因为它们与主几何实验不是同一 checkpoint/task/parameter intervention。现有 R18 结果 (iclr_1/runs/r18_completion_20260907/
  RESULTS.md) 也明确声明没有建立这种因果联系。

  不能直接把三个 layernorm slice 的 fidelity 与 full-parameter forgetting 做相关性——两者作用的参数集合不同。应当选择以下一种：

  - 对与训练完全相同的参数块测 fidelity；
  - 或者只训练被测参数块；
  - 或对所有受保护参数计算完整的 matrix-free、update-weighted fidelity。

  三个 seeds × 两种方法只有六个点，而且 method 与 fidelity 高度混淆，不足以支撑相关性。若要将其作为 Accept 级主结果，最好再引入 calibration size 或 rank 等预先规定的 fidelity 变化，同时保持模型、任
  务、replay 和 regularization strength 一致。