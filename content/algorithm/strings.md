+++
date = '2025-04-19T12:00:43+08:00'
draft = false
title = 'string类题目'
+++


## 🧠 Go语言实现字符串题目的「知识点图谱」

```
字符串处理
│
├── 1. 基础操作类
│   ├── 去空格 / 提取单词
│   │   ├── 58. Length of Last Word ✔️
│   │   └── 151. Reverse Words in a String ✔️
│   ├── 字符串比较 / 判断
│   │   └── 125. Valid Palindrome ✔️
│   └── 前缀/后缀处理
│       └── 14. Longest Common Prefix ✔️
│
├── 2. 字符统计类
│   ├── 242. Valid Anagram
│   ├── 387. First Unique Character in a String
│   └── 3. Longest Substring Without Repeating Characters
│
├── 3. 反转与重构类
│   ├── 344. Reverse String
│   ├── 541. Reverse String II
│   ├── 151. Reverse Words in a String ✔️
│   └── 917. Reverse Only Letters
│
├── 4. 双指针应用类
│   ├── 680. Valid Palindrome II
│   ├── 11. Container With Most Water (类比)
│   └── 876. Middle of the Linked List (思路相似)
│
├── 5. 模拟类 / 字符串构造
│   ├── 6. ZigZag Conversion
│   ├── 43. Multiply Strings
│   └── 67. Add Binary
│
└── 6. 高级类（KMP、Rabin-Karp、Trie）
    ├── 28. Implement strStr()
    ├── 686. Repeated String Match
    └── 336. Palindrome Pairs
```

---

## 🧩 每类代表题目详解

### ✅ 1. 【基础操作类】  
**58. Length of Last Word**  
- 遍历尾部开始，跳过空格，数出最后一个单词长度。
- 技巧：反向遍历 + 字符判断

**151. Reverse Words in a String**  
- 用 `strings.Fields()` 处理多空格，然后反转。
- 技巧：字符串切割 + 数组反转 + Join

---

### ✅ 2. 【字符统计类】  
**242. Valid Anagram**  
- 判断两个字符串是否是字母重排。
- 技巧：map 或数组统计字母出现次数。

---

### ✅ 3. 【反转与重构类】  
**344. Reverse String**  
- 双指针前后交换字符。

**541. Reverse String II**  
- 每隔 2k 反转前 k 个字符。

---

### ✅ 4. 【双指针应用类】  
**125. Valid Palindrome**  
- 左右双指针跳过非字母数字，比较大小写。

**680. Valid Palindrome II**  
- 允许删一个字符，判断是否仍为回文。

---

### ✅ 5. 【模拟类】  
**67. Add Binary**  
- 二进制字符串模拟加法，处理进位。

**43. Multiply Strings**  
- 字符串模拟乘法，用数组保存每位结果。

---

### ✅ 6. 【高级类（算法类）】  
**28. strStr() 实现 KMP**  
- 找字符串中子串第一次出现的位置。

**686. Repeated String Match**  
- 最少重复次数使得目标字符串是子串。

---

## 📌 常用技巧总结

| 技巧             | 说明                              | 示例题目                         |
|------------------|-----------------------------------|----------------------------------|
| 双指针            | 从两边向中间缩进或左右扫描         | 125, 680                         |
| 字符串切割         | 使用 `strings.Fields`, `split`    | 151                              |
| 字符串拼接         | `strings.Join`, `+`, `bytes.Buffer` | 多数构造类题目                  |
| 字符统计/比较      | 使用数组或哈希表统计字符出现次数   | 242                              |
| 反转数组/字符串     | 前后指针交换元素                   | 344, 541                         |
| 模拟人工计算       | 模拟加法乘法处理每一位             | 43, 67                           |
| KMP 字符匹配算法   | 高效搜索子串位置                   | 28                               |

---

## 更详细的题目总结

### 字符串处理题目 · 知识点图谱

#### 一、基础操作类

- **58. Length of Last Word**
  - 从后向前遍历，跳过空格，计数最后一个单词长度。
- **151. Reverse Words in a String**
  - 分割去多余空格，反转数组，拼接成字符串。
- **125. Valid Palindrome**
  - 忽略非字母数字，忽略大小写，双指针判断回文。
- **14. Longest Common Prefix**
  - 遍历第一个字符串，逐字符与其它字符串对比。

---

#### 二、字符统计类

- **242. Valid Anagram**
  - 判断两个字符串是否是字母异位词，使用哈希表或数组计数。
- **387. First Unique Character in a String**
  - 统计每个字符出现次数，再找第一个出现一次的字符。
- **3. Longest Substring Without Repeating Characters**
  - 滑动窗口，记录字符是否出现，维护最大子串。

---

#### 三、反转与重构类

- **344. Reverse String**
  - 双指针反转字符数组。
- **541. Reverse String II**
  - 每隔 2k 反转前 k 个字符。
- **917. Reverse Only Letters**
  - 双指针跳过非字母，只交换字母。

---

#### 四、双指针应用类

- **125. Valid Palindrome**
  - 左右指针跳过无效字符，比较大小写。
- **680. Valid Palindrome II**
  - 可以删除一个字符后成为回文，尝试跳过左右字符判断。
- **11. Container With Most Water（类比）**
  - 不是字符串题，但双指针策略类似。
- **876. Middle of the Linked List（思路相似）**
  - 快慢指针找中间节点。

---

#### 五、模拟 / 构造类

- **6. ZigZag Conversion**
  - 模拟字符写入不同行。
- **43. Multiply Strings**
  - 模拟乘法每一位，进位处理。
- **67. Add Binary**
  - 模拟二进制加法，处理进位。

---

#### 六、高级匹配类（KMP / Rabin-Karp / Trie）

- **28. Implement strStr()**
  - 实现子串查找（KMP可选）。
- **686. Repeated String Match**
  - 最少重复使得目标成为子串。
- **336. Palindrome Pairs**
  - 字典构造 + 判断拼接是否为回文。

---

#### 常用技巧总结

| 技巧             | 示例题目                         | 描述                            |
|------------------|----------------------------------|---------------------------------|
| 双指针            | 125, 680                         | 左右向中间夹逼                  |
| 切割 + Join      | 151                              | 去多余空格并反转                |
| Hash 计数        | 242, 387                         | 判断异位词/统计出现次数         |
| 滑动窗口          | 3                                | 无重复最长子串                  |
| 模拟人工加法/乘法 | 67, 43                           | 字符串数学模拟题                |
| 正则或 KMP       | 28, 686                          | 子串查找/重复匹配               |

---


注：本文由chatgpt总结