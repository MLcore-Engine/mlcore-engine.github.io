+++
date = '2025-04-21T11:22:37+08:00'
draft = true
title = '双指针'
+++

双指针是一种常见的算法技巧，广泛应用于数组和链表相关问题中，能够在许多情况下以 O(n) 的时间复杂度解决问题。
- (125. Valid Palindrome)
- (392. Is Subsequence)
- (167. Two Sum II)
- (11. Container With Most Water)
- (5. 3Sum)

---

### 1. 125. Valid Palindrome（有效的回文）

#### 问题描述
给定一个字符串，判断它是否是回文串。需要忽略非字母和数字的字符，且不区分大小写。例如，输入 `"A man, a plan, a canal: Panama"`，输出 `true`。

#### 解题思路
使用双指针，一个指向字符串头部（`left`），一个指向尾部（`right`）。从两端向中间移动，跳过非字母和数字的字符，比较遇到的字母或数字（转换为小写后比较）。如果所有比较都相等，则为回文串。

#### Go 语言实现
```go
func isPalindrome(s string) bool {
    left, right := 0, len(s)-1
    for left < right {
        // 跳过非字母和数字
        for left < right && !isAlnum(s[left]) {
            left++
        }
        for left < right && !isAlnum(s[right]) {
            right--
        }
        // 比较字符（不区分大小写）
        if left < right {
            if toLower(s[left]) != toLower(s[right]) {
                return false
            }
            left++
            right--
        }
    }
    return true
}

// 判断是否为字母或数字
func isAlnum(c byte) bool {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
}

// 将大写字母转换为小写
func toLower(c byte) byte {
    if c >= 'A' && c <= 'Z' {
        return c + 'a' - 'A'
    }
    return c
}
```

---

### 2. 392. Is Subsequence（子序列）

#### 问题描述
给定两个字符串 `s` 和 `t`，判断 `s` 是否是 `t` 的子序列。子序列是指通过删除 `t` 中的某些字符（不改变剩余字符的相对顺序）可以得到 `s`。例如，`s = "abc"`, `t = "ahbgdc"`，输出 `true`。

#### 解题思路
使用双指针，`i` 指向 `s` 的当前位置，`j` 指向 `t` 的当前位置。遍历 `t`，当 `t[j]` 等于 `s[i]` 时，`i` 前进一步。最终如果 `i` 到达 `s` 的末尾，说明 `s` 是 `t` 的子序列。

#### Go 语言实现
```go
func isSubsequence(s string, t string) bool {
    i, j := 0, 0
    for i < len(s) && j < len(t) {
        if s[i] == t[j] {
            i++
        }
        j++
    }
    return i == len(s)
}
```

---

### 3. 167. Two Sum II - Input Array Is Sorted（两数之和 II - 输入数组已排序）

#### 问题描述
给定一个已按升序排序的数组 `numbers` 和一个目标值 `target`，找到两个数的下标（从 1 开始），使得它们的和等于 `target`。例如，`numbers = [2,7,11,15]`, `target = 9`，输出 `[1,2]`。

#### 解题思路
利用数组已排序的特性，使用双指针，一个指向头部（`left`），一个指向尾部（`right`）。计算当前两数之和 `sum`：
- 如果 `sum == target`，返回下标；
- 如果 `sum < target`，`left` 右移；
- 如果 `sum > target`，`right` 左移。

#### Go 语言实现
```go
func twoSum(numbers []int, target int) []int {
    left, right := 0, len(numbers)-1
    for left < right {
        sum := numbers[left] + numbers[right]
        if sum == target {
            return []int{left + 1, right + 1}
        } else if sum < target {
            left++
        } else {
            right--
        }
    }
    return nil // 题目保证有解，此处仅为形式
}
```

---

### 4. 11. Container With Most Water（容器最大容量）

#### 问题描述
给定一个整数数组 `height`，每个元素表示垂直线的高度，求两条线与 x 轴围成的最大面积。例如，`height = [1,8,6,2,5,4,8,3,7]`，输出 `49`。

#### 解题思路
使用双指针，一个指向数组头部（`left`），一个指向尾部（`right`）。面积由宽度（`right - left`）和高度（两线中较短的那个）决定。每次计算当前面积，并移动较短的线（因为移动较长的线不会增加面积），更新最大面积。

#### Go 语言实现
```go
func maxArea(height []int) int {
    left, right := 0, len(height)-1
    maxArea := 0
    for left < right {
        width := right - left
        h := min(height[left], height[right])
        area := width * h
        if area > maxArea {
            maxArea = area
        }
        if height[left] < height[right] {
            left++
        } else {
            right--
        }
    }
    return maxArea
}

```

---

### 5. 15. 3Sum（三数之和）

#### 问题描述
给定一个整数数组 `nums`，找到所有和为 0 的三元组，结果不能包含重复的三元组。例如，`nums = [-1,0,1,2,-1,-4]`，输出 `[[-1,-1,2],[-1,0,1]]`。

#### 解题思路
先对数组排序，固定一个数 `nums[i]`，然后在剩余部分使用双指针（`left` 和 `right`）寻找和为 `-nums[i]` 的两个数。注意跳过重复的数以避免重复结果。

#### Go 语言实现
```go
func threeSum(nums []int) [][]int {
    sort.Ints(nums)
    var result [][]int
    for i := 0; i < len(nums)-2; i++ {
        if i > 0 && nums[i] == nums[i-1] {
            continue // 跳过重复
        }
        target := -nums[i]
        left, right := i+1, len(nums)-1
        for left < right {
            sum := nums[left] + nums[right]
            if sum == target {
                result = append(result, []int{nums[i], nums[left], nums[right]})
                left++
                right--
                // 跳过重复
                for left < right && nums[left] == nums[left-1] {
                    left++
                }
                for left < right && nums[right] == nums[right+1] {
                    right--
                }
            } else if sum < target {
                left++
            } else {
                right--
            }
        }
    }
    return result
}
```

---

### 双指针的常考题型


1. **回文串判断**  
   - 典型题目：125. Valid Palindrome  
   - 特点：从两端向中间移动，跳过无关字符，比较对称位置的字符。

2. **子序列判断**  
   - 典型题目：392. Is Subsequence  
   - 特点：在一个字符串中顺序查找另一个字符串的字符，指针只前进不后退。

3. **两数之和**  
   - 典型题目：167. Two Sum II  
   - 特点：在有序数组中，利用双指针从两端逼近目标值。

4. **容器最大容量**  
   - 典型题目：11. Container With Most Water  
   - 特点：通过移动指针优化面积计算，移动较短的边以寻找更大可能。

5. **三数之和**  
   - 典型题目：15. 3Sum  
   - 特点：结合排序和双指针，固定一个数后在剩余部分寻找符合条件的另外两个数。

### 总结表格

| 题号 | 名称 | 技巧类型 | 典型场景 |
|------|------|---------|---------|
| 125 | Valid Palindrome | 对撞指针 | 判断回文、清洗字符串 |
| 392 | Is Subsequence | 快慢指针 | 子序列判断 |
| 167 | Two Sum II | 对撞指针 | 寻找两数和（已排序） |
| 11 | Container With Most Water | 对撞指针 | 最大容量优化问题 |
| 15 | 3Sum | 排序 + 对撞指针 | 查找三数组合 |