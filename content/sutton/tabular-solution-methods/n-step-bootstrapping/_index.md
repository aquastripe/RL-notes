---
title: "n-step Bootstrapping"
date: 2021-11-14T08:44:09+08:00
draft: false
math: true
---

本章探討如何結合 Monte Carlo (MC) 與 temporal difference (TD)。

{{< toc >}}

## n-step TD Prediction

比較 MC 與 TC 之間的差異。

考慮如何從以 $\pi$ 產生的 episodes 預估 $v_{\pi}$：
- MC: 根據整個 episode 的所有 states 進行更新
- 1-step TD: 只根據下一個 state 進行更新
- n-step TD: 介於兩者的中間，根據下 n 個 state 進行更新
- $\infty$-step TD: 等同於 MC

n-step 的 backup diagram 如下:

![](7.1.png)
- 空心: state
- 實心: action
- 方形: 中止狀態

考慮到如何估計 $S_{t}, R_{t+1}, S_{t+1}, R_{t+2}, \ldots, R_{T}, S_{T}$ 的 value:

已知 MC 的更新式如下：
$$
G_{t} \doteq R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\cdots+\gamma^{T-t-1} R_{T}
$$
其中 $T$ 是最後一個 time step。在此，稱這個量為更新的 **目標** (the **target** of the update)。

1-step TD 的更新目標如下：
$$
G_{t: t+1} \doteq R_{t+1}+\gamma V_{t}\left(S_{t+1}\right)
$$
其中 $V_{t}: \mathcal{S} \rightarrow \mathbb{R}$ 代表 $v_{\pi}$ 在時間 $t$ 時的估計值。

2-step TD 的更新目標可以推廣如下：
$$
G_{t: t+2} \doteq R_{t+1}+\gamma R_{t+2}+\gamma^{2} V_{t+1}\left(S_{t+2}\right)
$$

以此類推，n-step TD 更新目標如下：
$$
G_{t: t+n} \doteq R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1} R_{t+n}+\gamma^{n} V_{t+n-1}\left(S_{t+n}\right)
$$
對所有 $n, t$ 使得 $n \ge 1$ 且 $0 \le t \le T-n$。

所有的 n-step 的 returns 都可以被視為用來近似於 "所有的 returns"，近似的部份為 $V_{t+n-1}(S_{t+n})$。如果 $t+n \ge T$ (亦即：n-step 會考慮到超出中止狀態之後的狀態) 則少掉的項會設定為 0。 

注意：當 $n \gt 1$ 時，n-step returns 會涉及未來的 rewards，所以要到 $t+n$ 後採樣到才能夠計算 $R_{t+n}$ 和 $V_{t+n-1}$。使用 n-step returns 的 state-value 學習演算法如下：
$$
V_{t+n}\left(S_{t}\right) \doteq V_{t+n-1}\left(S_{t}\right)+\alpha\left[G_{t: t+n}-V_{t+n-1}\left(S_{t}\right)\right], \quad 0 \leq t<T
$$
同時，對於所有 $s \ne S_{t}$ 的 state-value 都保持不變。

![](alg-n-step-td.png)

### Error reduction property

n-step returns 使用 $V_{t+n-1}$ 來近似在 $R_{t+n}$ 之後的未知的 rewards。一個重要的性質是：在最糟的狀態下，n-step returns 的期望值保證會比 $V_{t+n-1}$ 更好：
$$
\max _{s}\left|\mathbb{E}_{\pi}\left[G_{t: t+n} \mid S_{t}=s\right]-v_{\pi}(s)\right| \leq \gamma^{n} \max _{s}\left|V_{t+n-1}(s)-v_{\pi}(s)\right|
$$
對所有 $n \ge 1$。

這個性質稱為 **error reduction property**。這個性質可以說明所有的 n-step 方法都收斂到正確的預測值 (predictions)。



### Example 7.1: n-step TD Methods on the Random Walk

參考 [Example 6.2](/RL-notes/sutton/tabular-solution-methods/temporal-difference-learning/#example-62)，以下探討設定多少的 n 結果最好。實驗設定：
- 參數 $n$ 與 $\alpha$
- 計算前 10 個 episodes 的結果
- 實驗重複執行 100 次取平均

結果如下圖：

![](7.2.png)

從這個實驗可以知道，n-step 有機會比兩個極端 (1-step TD 與 MC) 結果更好。
