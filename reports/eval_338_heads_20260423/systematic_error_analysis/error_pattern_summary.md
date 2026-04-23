# ICDAS4 ROI 错例分析总结（val/test）

## 1. 各模型错误类型统计

### VAL 集

| model | n_total | n_errors | error_ratio | err_0_A_count | err_0_A_ratio_err | err_A_B_count | err_A_B_ratio_err | err_B_C_count | err_B_C_ratio_err | err_cross_ge2_count | err_cross_ge2_ratio_err | err_adjacent_count | err_adjacent_ratio_err |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| softmax | 663 | 216 | 0.3258 | 154 | 0.7130 | 41 | 0.1898 | 14 | 0.0648 | 7 | 0.0324 | 209 | 0.9676 |
| ord2seq | 663 | 206 | 0.3107 | 146 | 0.7087 | 39 | 0.1893 | 15 | 0.0728 | 6 | 0.0291 | 200 | 0.9709 |
| corn | 663 | 185 | 0.2790 | 119 | 0.6432 | 41 | 0.2216 | 18 | 0.0973 | 7 | 0.0378 | 178 | 0.9622 |
| softmax_ordplus_o2s | 663 | 184 | 0.2775 | 128 | 0.6957 | 37 | 0.2011 | 13 | 0.0707 | 6 | 0.0326 | 178 | 0.9674 |
| softmax_ordplus_o2s_boundary_gpt | 663 | 180 | 0.2715 | 123 | 0.6833 | 38 | 0.2111 | 13 | 0.0722 | 6 | 0.0333 | 174 | 0.9667 |

### TEST 集

| model | n_total | n_errors | error_ratio | err_0_A_count | err_0_A_ratio_err | err_A_B_count | err_A_B_ratio_err | err_B_C_count | err_B_C_ratio_err | err_cross_ge2_count | err_cross_ge2_ratio_err | err_adjacent_count | err_adjacent_ratio_err |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| softmax | 674 | 198 | 0.2938 | 126 | 0.6364 | 47 | 0.2374 | 19 | 0.0960 | 6 | 0.0303 | 192 | 0.9697 |
| ord2seq | 674 | 214 | 0.3175 | 148 | 0.6916 | 31 | 0.1449 | 18 | 0.0841 | 17 | 0.0794 | 197 | 0.9206 |
| corn | 674 | 205 | 0.3042 | 124 | 0.6049 | 54 | 0.2634 | 13 | 0.0634 | 14 | 0.0683 | 191 | 0.9317 |
| softmax_ordplus_o2s | 674 | 190 | 0.2819 | 126 | 0.6632 | 36 | 0.1895 | 15 | 0.0789 | 13 | 0.0684 | 177 | 0.9316 |
| softmax_ordplus_o2s_boundary_gpt | 674 | 177 | 0.2626 | 111 | 0.6271 | 41 | 0.2316 | 16 | 0.0904 | 9 | 0.0508 | 168 | 0.9492 |

## 2. boundary_gpt 重点分析

### VAL 集

- boundary_gpt 总样本: 663, 错误: 180, 错误率: 0.2715
- 相邻错级: 174 (0.9667 of errors)
- 跨两级及以上: 6 (0.0333 of errors)
- boundary_gpt 错, softmax+ord2seq 都对: 21
- boundary_gpt 对, softmax+ord2seq 都错: 23
- softmax+ord2seq+boundary_gpt 三者都错: 95

最常见误判模式（boundary_gpt）:

| pattern | count |
| :--- | :--- |
| A->0 | 72 |
| 0->A | 51 |
| B->A | 29 |
| A->B | 9 |
| C->B | 8 |
| B->0 | 5 |
| B->C | 5 |
| 0->B | 1 |

### TEST 集

- boundary_gpt 总样本: 674, 错误: 177, 错误率: 0.2626
- 相邻错级: 168 (0.9492 of errors)
- 跨两级及以上: 9 (0.0508 of errors)
- boundary_gpt 错, softmax+ord2seq 都对: 19
- boundary_gpt 对, softmax+ord2seq 都错: 29
- softmax+ord2seq+boundary_gpt 三者都错: 96

最常见误判模式（boundary_gpt）:

| pattern | count |
| :--- | :--- |
| 0->A | 59 |
| A->0 | 52 |
| B->A | 25 |
| A->B | 16 |
| C->B | 10 |
| B->0 | 7 |
| B->C | 6 |
| C->0 | 1 |

## 3. 额外统计

- A↔0 是否最大误判来源: 234/357 (0.6555)
- 相比 static ordplus，boundary_gpt 修正成功最多类别:
| pair | count |
| :--- | :--- |
| A<-0 | 71 |
| 0<-A | 36 |
| B<-A | 20 |
| B<-C | 13 |
| A<-B | 2 |
- 相比 softmax，boundary_gpt 新增错误最多类别:
| pair | count |
| :--- | :--- |
| 0->A | 40 |
| A->0 | 39 |
| B->A | 24 |
| A->B | 3 |
| C->B | 3 |

## 4. 证据驱动原因判断

- 相邻错级占比: 0.9580；跨两级占比: 0.0420。
- 错误中 GT 属于 B/C 的占比(长尾近似): 0.2717。
- 若相邻错级远高于跨级错，通常更支持边界模糊/标注近边界样本导致；若跨级占比较高，常见于 ROI 裁剪或严重语义缺失。