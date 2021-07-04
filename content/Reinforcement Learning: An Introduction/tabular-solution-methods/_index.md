---
title: "Tabular Solution Methods"
date: 2021-07-04T11:22:06+08:00
draft: false
math: true
resources:
    - name:
      src: ""
      params:
        credits: "[Richard S. Sutton](http://incompleteideas.net/index.html) and [Andrew G. Barto](https://people.cs.umass.edu/~barto/) on [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf)"
---

{{< toc >}}

**表格式解法** (**Tabular Solution Method**) 為 RL 最簡單的形式：當所有的狀態和動作數量少到可以用多維陣列來表示價值函數。這些方法通常可以找到精確解 (exact solution)，也就是精確的 價值函數 和 策略 的極值。

第一章是 RL 的特殊形式：只有一個狀態，稱為 **吃角子老虎機問題** (bandit problems)。

第二章是更通用的問題形式：有限馬可夫決策過程，主要概念包含 **貝爾曼方程 (Bellman equations**；又稱為 **動態規劃方程**) 和 **價值函數**。

第三、四、五章描述三個基本類型的方法來解決 有限馬可夫決策 問題：**動態規劃** (**dynamic programming**)、**蒙特卡羅法 (Monte Carlo methods**)、**時序差分學習法** (**temporal difference learning**)。每個方法各有優劣：

- 動態規劃：需要完整且精確的環境模型
- 蒙特卡羅法：不需要完整且精確的環境模型，但是不適合逐步的增量計算。
    - 增量計算 (incremental computation): 是一個軟體功能，當只有一小塊資料改變的時候，只會對產生變化的部分進行計算和更新，以節省計算時間。(is a software feature which, whenever a piece of data changes, attempts to save time by only recomputing those outputs which depend on the changed data.)
- 時序差分學習法：不需要完整的環境模型，也完全支援增量計算，但是更複雜而難以分析。

第六、七章描述這三種方法如何結合各自的優點。第六章描述如何用 **多步拔靴法** (**自助法**；**自助抽樣法**；**multi-step bootstrapping methods**) 結合 **蒙特卡羅法** 和 **時間差分學習法**。第七章描述如何以 時序差分學習法 結合 模型學習 (model learning) 和 規劃法 (例如動態規劃) 來解決通用的表格式 RL 問題。
