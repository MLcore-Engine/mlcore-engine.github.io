+++
date = '2025-05-30T09:14:55+08:00'
draft = true
title = '哈希表相关题目'
+++


### 哈希表一般解题方法



#### 1. 频率计数法（Frequency Counting）
**适用问题**: Ransom Note、Valid Anagram、Group Anagrams  
**方法**: 使用哈希表统计字符或元素的出现频率。  
- 对于 Ransom Note，检查一个字符串是否能由另一个字符串的字符构成，通过统计频率并逐一扣减。
    -  例子:
    - ransomNote = "aa", magazine = "aab" → 可以，因为 magazine 有 2 个 'a' 和 1 个 'b'，足够拼出 "aa"。
    - ransomNote = "aa", magazine = "ab" → 不行，因为 magazine 只有 1 个 'a'，不够用。
- 对于 Valid Anagram，比较两个字符串的字符频率是否完全相同。
    - 例子:
    - s = "listen", t = "silent" → 是，因为两者的字符（l, i, s, t, e, n）次数相同。
    - s = "hello", t = "world" → 不是，因为字符种类和次数都不一样。
- 对于 Group Anagrams，将频率特征（或排序后的字符串）作为键，将相同频率的字符串分组。
    - 例子:
    - 输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
    - 输出: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
    - 解释: "eat"、"tea"、"ate" 是异位词，"tan"、"nat" 是异位词，"bat" 单独一组。

**关键点**:  c
- 使用 map 统计频率。
- 考虑输入是否仅限于 ASCII（可用数组代替 map）或包含 Unicode（用 map 更通用）。


- **频率计数法 Valid Anagram** 
```go
func isAnagram(s, t string) bool {
    if len(s) != len(t) {
        return false
    }
    count := make(map[rune]int)
    for _, char := range s {
        count[char]++
    }
    for _, char := range t {
        count[char]--
        if count[char] < 0 {
            return false
        }
    }
    return true
}
```
**时间复杂度**: O(n)，n 为字符串长度。  
**空间复杂度**: O(k)，k 为字符集大小。



---

#### 2. 映射与模式法（Mapping and Patterns）
**适用问题**: Isomorphic Strings、Word Pattern  
**方法**: 使用哈希表记录字符或单词之间的映射关系，确保映射一致性。  
- Isomorphic Strings 检查两个字符串是否可以通过某种一致的字符映射相互转换。
    - 例子:
    - s = "egg", t = "add" → 是同构的。
    - 映射：e → a, g → d，每个 e 都对应 a，每个 g 都对应 d，映射一致。
    - s = "foo", t = "bar" → 不是同构的。
    - 映射：f → b, o → a, 但第二个 o 需要映射到 r，不一致。
    - s = "paper", t = "title" → 是同构的。
    - 映射：p → t, a → i, p → t, e → l, r → e，映射一致。
- Word Pattern 检查字符串中的单词序列是否与给定的模式一致。
    - 例子:
    - pattern = "abba", s = "dog cat cat dog" → 是。
    - 映射：a → dog, b → cat，模式一致。
    - pattern = "abba", s = "dog cat cat fish" → 不是。
    - 映射：a → dog, b → cat，但最后一个 a 应为 dog，却遇到 fish，不一致。
    - pattern = "aaaa", s = "dog dog dog dog" → 是。
    - 映射：a → dog，一致。
    - pattern = "abba", s = "dog dog dog dog" → 不是。
    - 映射：a → dog, b → dog，但 a 和 b 不能映射到同一个单词。
- 通常需要两个 map，确保双向映射唯一。

**关键点**:  
- 双向检查映射，避免一对多或多对一的情况。

### GoLang 示例代码

以下是每类方法的 GoLang 实现，帮助你理解和记忆。


- **映射与模式法: Isomorphic Strings**
```go
func isIsomorphic(s, t string) bool {
    if len(s) != len(t) {
        return false
    }
    mapST := make(map[byte]byte)
    mapTS := make(map[byte]byte)
    for i := 0; i < len(s); i++ {
        charS, charT := s[i], t[i]
        if val, ok := mapST[charS]; ok {
            if val != charT {
                return false
            }
        } else {
            mapST[charS] = charT
        }
        if val, ok := mapTS[charT]; ok {
            if val != charS {
                return false
            }
        } else {
            mapTS[charT] = charS
        }
    }
    return true
}
```
**时间复杂度**: O(n)，n 为字符串长度。  
**空间复杂度**: O(k)，k 为字符集大小。



---

#### 3. 查找配对或重复法（Finding Pairs or Duplicates）
**适用问题**: Two Sum、Contains Duplicate II  
**方法**: 使用哈希表记录已见过的元素及其位置或属性。  
- Two Sum 在数组中找两个和为目标值的数，用 map 存储每个数的补数。
    - 例子:
    - 输入: nums = [2, 7, 11, 15], target = 9
    - 输出: [0, 1] 解释: nums[0] + nums[1] = 2 + 7 = 9。
    - 输入: nums = [3, 2, 4], target = 6
    - 输出: [1, 2] 解释: nums[1] + nums[2] = 2 + 4 = 6。
    - 输入: nums = [3, 3], target = 6
    - 输出: [0, 1] 解释: nums[0] + nums[1] = 3 + 3 = 6。
- Contains Duplicate II 检查数组中是否存在距离不超过 k 的重复元素，可用滑动窗口结合 map 或 set。 给你一个整数数组 nums 和一个整数 k，判断数组中是否存在两个不同的下标 i 和 j，使得 nums[i] == nums[j] 并且 |i - j| <= k（即下标差不超过 k）。如果存在，返回 true；否则返回 false。
    - 例子:
    - 输入: nums = [1, 2, 3, 1], k = 3
    - 输出: true
    - 解释: nums[0] = nums[3] = 1，且 |0 - 3| = 3 <= k。
    - 输入: nums = [1, 0, 1, 1], k = 1
    - 输出: true
    - 解释: nums[2] = nums[3] = 1，且 |2 - 3| = 1 <= k。
    - 输入: nums = [1, 2, 3, 1, 2, 3], k = 2
    - 输出: false
    - 解释: 虽然有重复元素 1，但它们的下标差 |0 - 3| = 3 > k。

**关键点**:  
- 对于范围限制问题（如 k），考虑滑动窗口优化空间。

- **查找配对法：Two Sum**
```go
func twoSum(nums []int, target int) []int {
    numMap := make(map[int]int)
    for i, num := range nums {
        complement := target - num
        if idx, ok := numMap[complement]; ok {
            return []int{idx, i}
        }
        numMap[num] = i
    }
    return nil
}
```
**时间复杂度**: O(n)，n 为数组长度。  
**空间复杂度**: O(n)。



---

#### 4. 循环检测法（Cycle Detection）
**适用问题**: Happy Number  
**方法**: 使用集合记录出现过的状态，检测是否进入循环。  
- Happy Number 通过不断计算各位数字平方和，判断是否能达到 1，若出现重复则有循环。
    Input: n = 19
    Output: true
    Explanation:
    12 + 92 = 82
    82 + 22 = 68
    62 + 82 = 100
    12 + 02 + 02 = 1

- **循环检测法：Happy Number**
```go
func isHappy(n int) bool {
    seen := make(map[int]struct{})
    for n != 1 && seen[n] == struct{}{} {
        seen[n] = struct{}{}
        n = getNext(n)
    }
    return n == 1
}

func getNext(n int) int {
    sum := 0
    for n > 0 {
        digit := n % 10
        sum += digit * digit
        n /= 10
    }
    return sum
}
```
**时间复杂度**: O(log n) 或 O(1)，取决于循环长度。  
**空间复杂度**: O(log n) 或 O(1)。

**关键点**:  
- 在 Go 中用 `map[类型]struct{}` 模拟集合。

---

#### 5. 序列检测法（Sequence Detection）
**适用问题**: Longest Consecutive Sequence  
```bash
    Example 1:

        Input: nums = [100,4,200,1,3,2]
        Output: 4
    Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
    Example 2:

        Input: nums = [0,3,7,2,5,8,4,6,0,1]
        Output: 9
    Example 3:

        Input: nums = [1,0,1,2]
        Output: 3
```
**方法**: 使用集合存储所有元素，检查每个可能的序列起点，计算最长连续序列。  
- 只对没有前驱（num-1 不存在）的数字开始检查序列，避免重复计算。


- **序列检测法：Longest Consecutive Sequence**
```go
func longestConsecutive(nums []int) int {

        numSet := make(map[int]bool)
        maxLen := 0

        for _, num := range nums {
            numSet[num] = true
        }

        for num := range numSet {
            if !numSet[num-1] {
                currentNum := num
                currentLen := 1
                for numSet[currentNum+1] {
                    currentNum++
                    currentLen++
                }
                maxLen = max(currentLen, maxLen)
            }
            
        }
    return maxLen

}

```
**时间复杂度**: O(n)，n 为数组长度。  
**空间复杂度**: O(n)。
**关键点**:  
- 优化时间复杂度到 O(n)，每个元素只访问一次。

---

### 解题步骤
1. **识别问题模式**  
   - 看到字符频率或相同性检查 → 频率计数法。
   - 看到映射或模式匹配 → 映射与模式法。
   - 看到找配对或重复 → 查找配对法。
   - 看到循环或重复计算 → 循环检测法。
   - 看到连续性或序列 → 序列检测法。

2. **熟练使用 GoLang 的 map**  
   - `make(map[类型]类型)` 创建哈希表。
   - `map[类型]struct{}` 模拟集合。
   - 通过 `val, ok := map[key]` 检查键是否存在。

3. **分析时间与空间复杂度**  
   - 哈希表方法通常将时间复杂度从 O(n²) 或 O(n log n) 优化到 O(n)。
   - 空间复杂度通常为 O(n) 或 O(k)，k 为有限字符集大小。
---