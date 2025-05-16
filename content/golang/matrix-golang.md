+++
date = '2025-05-16T10:23:03+08:00'
draft = false
title = 'matrix-golang'
+++

---

## 常见的矩阵变形总结 Go实现：

| 操作名称        | 说明                                      | 示例变化前 -> 后（以 3x3 为例）   |
| ----------- | --------------------------------------- | ---------------------- |
| 顺时针旋转 90°   | 转置 + 每行左右翻转                             | `[1,2,3]` -> `[7,4,1]` |
| 逆时针旋转 90°   | 转置 + 每列上下翻转                             | `[1,2,3]` -> `[3,6,9]` |
| 顺时针旋转 180°  | 上下翻转 + 每行左右翻转                           | `[1,2,3]` -> `[9,8,7]` |
| 上下翻转        | 直接交换上和下的行                               | `[1,2,3]` -> `[7,8,9]` |
| 左右翻转        | 每一行反转                                   | `[1,2,3]` -> `[3,2,1]` |
| 主对角线转置      | matrix\[i]\[j] ↔ matrix\[j]\[i]         | `[1,2,3]` -> `[1,4,7]` |
| 副对角线转置（反转置） | matrix\[i]\[j] ↔ matrix\[n-1-j]\[n-1-i] | `[1,2,3]` -> `[9,6,3]` |

---

### 1. 主对角线转置（matrix\[i]\[j] ↔ matrix\[j]\[i]）

```go
func transpose(matrix [][]int) {
    n := len(matrix)
    for i := 0; i < n; i++ {
        for j := i + 1; j < n; j++ {
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        }
    }
}
```

---

### 2. 上下翻转（交换行）

```go
func flipUpDown(matrix [][]int) {
    n := len(matrix)
    for i := 0; i < n/2; i++ {
        matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
    }
}
```

---

### 3. 左右翻转（每行逆序）

```go
func flipLeftRight(matrix [][]int) {
    n := len(matrix)
    for i := 0; i < n; i++ {
        for j := 0; j < n/2; j++ {
            matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]
        }
    }
}
```

---

### 4. 逆时针旋转 90 度（转置 + 上下翻转）

```go
func rotateCounterClockwise(matrix [][]int) {
    transpose(matrix)
    flipUpDown(matrix)
}
```

---

### 5. 顺时针旋转 180 度（上下翻转 + 左右翻转）

```go
func rotate180(matrix [][]int) {
    flipUpDown(matrix)
    flipLeftRight(matrix)
}
```

---

### 6. 副对角线转置（matrix\[i]\[j] ↔ matrix\[n-1-j]\[n-1-i]）

```go
func antiDiagonalTranspose(matrix [][]int) {
    n := len(matrix)
    for i := 0; i < n; i++ {
        for j := 0; j < n-i-1; j++ {
            matrix[i][j], matrix[n-1-j][n-1-i] = matrix[n-1-j][n-1-i], matrix[i][j]
        }
    }
}
```

---

## 总结：

| 操作         | 调用组合                         |
| ---------- | ---------------------------- |
| 顺时针旋转 90°  | `transpose + flipLeftRight`  |
| 逆时针旋转 90°  | `transpose + flipUpDown`     |
| 顺时针旋转 180° | `flipUpDown + flipLeftRight` |
| 上下翻转       | `flipUpDown`                 |
| 左右翻转       | `flipLeftRight`              |
| 主对角线转置     | `transpose`                  |
| 副对角线转置     | `antiDiagonalTranspose`      |

---


## Go-Demo：

```go
package main

import (
    "fmt"
)

func printMatrix(matrix [][]int) {
    for _, row := range matrix {
        fmt.Println(row)
    }
    fmt.Println()
}

func cloneMatrix(matrix [][]int) [][]int {
    n := len(matrix)
    newMatrix := make([][]int, n)
    for i := range matrix {
        newMatrix[i] = make([]int, n)
        copy(newMatrix[i], matrix[i])
    }
    return newMatrix
}

func transpose(matrix [][]int) {
    n := len(matrix)
    for i := 0; i < n; i++ {
        for j := i + 1; j < n; j++ {
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        }
    }
}

func flipUpDown(matrix [][]int) {
    n := len(matrix)
    for i := 0; i < n/2; i++ {
        matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
    }
}

func flipLeftRight(matrix [][]int) {
    n := len(matrix)
    for i := 0; i < n; i++ {
        for j := 0; j < n/2; j++ {
            matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]
        }
    }
}

func antiDiagonalTranspose(matrix [][]int) {
    n := len(matrix)
    for i := 0; i < n; i++ {
        for j := 0; j < n-i-1; j++ {
            matrix[i][j], matrix[n-1-j][n-1-i] = matrix[n-1-j][n-1-i], matrix[i][j]
        }
    }
}

func rotate90Clockwise(matrix [][]int) {
    transpose(matrix)
    flipLeftRight(matrix)
}

func rotate90CounterClockwise(matrix [][]int) {
    transpose(matrix)
    flipUpDown(matrix)
}

func rotate180(matrix [][]int) {
    flipUpDown(matrix)
    flipLeftRight(matrix)
}

func main() {
    original := [][]int{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    }

    fmt.Println("Original:")
    printMatrix(original)

    m1 := cloneMatrix(original)
    rotate90Clockwise(m1)
    fmt.Println("Rotate 90° Clockwise:")
    printMatrix(m1)

    m2 := cloneMatrix(original)
    rotate90CounterClockwise(m2)
    fmt.Println("Rotate 90° Counter-Clockwise:")
    printMatrix(m2)

    m3 := cloneMatrix(original)
    rotate180(m3)
    fmt.Println("Rotate 180°:")
    printMatrix(m3)

    m4 := cloneMatrix(original)
    flipUpDown(m4)
    fmt.Println("Flip Up-Down:")
    printMatrix(m4)

    m5 := cloneMatrix(original)
    flipLeftRight(m5)
    fmt.Println("Flip Left-Right:")
    printMatrix(m5)

    m6 := cloneMatrix(original)
    transpose(m6)
    fmt.Println("Transpose (Main Diagonal):")
    printMatrix(m6)

    m7 := cloneMatrix(original)
    antiDiagonalTranspose(m7)
    fmt.Println("Anti-Diagonal Transpose:")
    printMatrix(m7)
}
```

---