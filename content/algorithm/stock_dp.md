+++
date = '2025-06-29T16:32:01+08:00'
draft = false
title = '股票交易动态规划总结'
+++


---

## 通用 DP 框架

对任意第 `i` 天（0 ≤ i < n）、已完成 `j` 次交易（j 为卖出次数，0 ≤ j ≤ k）、持股状态 `s`（0 = 未持有，1 = 持有），定义：

```
dp[i][j][0] = 第 i 天结束时，进行了 ≤ j 次卖出操作，且当前不持股时的最大收益
dp[i][j][1] = 第 i 天结束时，进行了 ≤ j 次卖出操作，且当前持股时的最大收益
```

* 一次完整交易 = 一次买入 + 一次卖出，我们把卖出次数纳入 j。
* 边界条件：

  * `dp[0][0][0] = 0`
  * `dp[0][0][1] = -price[0]`（第 0 天买入）
  * 对于 j > 0，可令 `dp[0][j][0] = 0`, `dp[0][j][1] = -price[0]`。

通用转移：

```
// 未持有 = 昨天未持有(休息) OR 昨天持有且今天卖出
dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + price[i] - fee_if_any)

// 持有   = 昨天持有(继续持有) OR 昨天不持有且今天买入
dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - price[i])
// 若有冷冻期，买入须参考 dp[i-2][j-1][0]
```

所有题型皆可在此框架下特化。下面依次列出：

---

### 1. LeetCode 121. Best Time to Buy and Sell Stock（k = 1）

**题意**：给定数组 `prices`，只允许完成 **至多 1** 次交易，求最大利润。

**特化**：k = 1，只需 j ∈ {0,1}。可将 `j` 维度消除：

```go
func maxProfit(prices []int) int {

    n := len(prices)
    dp := make([][]int, n)
    for i := range dp {
        dp[i] = make([]int, 2)
    }

    dp[0][0] = 0
    dp[0][1] = -prices[0]
    for i := 1; i < n; i++ {

        dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
      // 只允许买入一次，所以 dp[i-1][0] - prices[i] 不再适用，永远只能从第一天买入
        dp[i][1] = max(- prices[i], dp[i-1][1])
    }
    return dp[n-1][0]
} 
```

**复杂度**：时间 O(n)，空间 O(n)。

---

### 2. LeetCode 122. Best Time to Buy and Sell Stock II（k = ∞）

**题意**：交易次数不限，求最大利润。

```go
func maxProfit(prices []int) int {

    dp := make([][]int, len(prices))

    for i := range dp {
        dp[i] = make([]int, 2)
    }

    dp[0][0] = 0
    dp[0][1] = -prices[0]

    for i := 1; i < len(prices); i++ {
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        dp[i][1] = max(dp[i-1][0] - prices[i], dp[i-1][1])
    }

    return dp[len(prices)-1][0]
}
```

**复杂度**：时间 O(n)，空间 O(n)。

---

### 3. LeetCode 123. Best Time to Buy and Sell Stock III（k = 2）

**题意**：最多可完成 **2** 笔交易，求最大利润。

**特化**：k = 2，j ∈ {0,1,2}。
状态维度为 `dp[i][j][0/1]`，j=卖出次数。

```
// 未持有
dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
// 持有
dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])
```

边界同通用，求 `dp[n-1][2][0]`。

```go
func maxProfit(prices []int) int {

    //n 表示天数 k交易次数 1 0 表示持有或者卖出
    n := len(prices)
    dp := make([][][]int, n)

    for i := range dp {
        dp[i] = make([][]int, 3)
    }

    for i := range dp {
        for j := range dp[0] {
            dp[i][j] = make([]int, 2)
        }
    }

    for i := 0; i < n; i++ {

        for k := 2; k >= 1; k-- {

            if i == 0 {
                dp[0][k][0] = 0
                dp[0][k][1] = -prices[0]
                continue
            }

            dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
            dp[i][k][1] = max(dp[i-1][k-1][0] - prices[i], dp[i-1][k][1])

        }

    }

    return dp[n-1][2][0]
}
```

**复杂度**：时间 O(n·k)=O(n)，空间 O(n)。

---

### 4. LeetCode 188. Best Time to Buy and Sell Stock IV（最多 k 次）

**题意**：最多可完成给定 `k` 笔交易。

**特化**：直接使用三维 `dp[i][j][s]`；若 `k > n/2`，可退化为 k = ∞ 问题（122）。

```go
func maxProfit(k int, prices []int) int {
    n := len(prices)
    dp := make([][][]int, n)
    for i := range dp {
        dp[i] = make([][]int, k + 1)
        for j := range dp[i] {
            dp[i][j] = make([]int, 2)
        }
    }


    for i := 0; i < n; i++ {

        for j := k; j >= 1; j-- {
            if i == 0 {

                dp[0][j][0] = 0
                dp[0][j][1] = -prices[0]
                continue
            }

            dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
            dp[i][j][1] = max(dp[i-1][j-1][0] - prices[i], dp[i-1][j][1])
        }

    }
    return dp[n-1][k][0]
}
```

**复杂度**：时间 O(n·k)，空间 O(n·k)。

---

### 5. LeetCode 309. Best Time to Buy and Sell Stock with Cooldown（冷冻期）

**题意**：交易次数无限，但每次卖出后需冷冻 1 天（即隔天不能买）。

```go
func maxProfit(prices []int) int {
    n := len(prices)

    dp := make([][2]int, n)

    dp[0][0] = 0
    dp[0][1] = -prices[0]

    for i := 1; i < n; i++ {

        if i == 1 {
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], -prices[i])
            continue
        }
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        dp[i][1] = max(dp[i-2][0] - prices[i], dp[i-1][1])
    }
    return dp[n-1][0]
}
```

**复杂度**：时间 O(n)，空间 O(n)。

---

### 6. LeetCode 714. Best Time to Buy and Sell Stock with Transaction Fee（手续费）

**题意**：交易次数无限，每笔交易收取固定手续费 `fee`。

**特化**：k = ∞ + 手续费。

```go
func maxProfit(prices []int, fee int) int {
    n := len(prices)
    dp := make([][2]int, n)

    dp[0][0] = 0
    dp[0][1] = -prices[0]
    for i := 1; i < n; i++ {
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i] - fee)
        dp[i][1] = max(dp[i-1][0] -prices[i], dp[i-1][1])
    }
    return dp[n-1][0]
}
```

复杂度：时间 O(n)，空间 O(1)。

## 总结

* 统一 `dp[i][j][s]` 框架，结合 `k` 值、冷冻期、手续费等特性进行特化。
