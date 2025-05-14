+++
date = '2025-04-21T11:37:26+08:00'
draft = true
title = 'sort包用法'
+++


### 一、`sort` 包

Go 的 `sort` 包提供了一组对切片（slice）排序和搜索的函数，以及一个灵活的接口 `sort.Interface`，可以用于对任何自定义数据结构进行排序。核心功能包括：

- **基础类型快速排序**：`sort.Ints`、`sort.Strings`、`sort.Float64s`
- **通用切片排序**：`sort.Slice`、`sort.SliceStable`
- **稳定排序**：`sort.Stable`
- **反向排序**：`sort.Reverse`
- **二分查找**：`sort.Search` 及其类型专用变体
- **自定义排序接口**：`sort.Interface`

---

### 二、基础类型快速排序

#### 1. `sort.Ints`

```go
func Ints(a []int)
```

- **作用**：对 `[]int` 原地升序排序，采用快速排序算法（平均 O(n log n)、最坏 O(n²)）。
- **示例**：

  ```go
  nums := []int{5, 3, 8, 1, 2}
  sort.Ints(nums)
  fmt.Println(nums) // [1 2 3 5 8]
  ```

- **场景**：对纯整型数据排序时最简便；常见于统计、排行榜、频率计算等。

#### 2. `sort.Strings`

```go
func Strings(a []string)
```

- **作用**：对 `[]string` 原地升序排序，字符串按字典序比较（平均 O(n log n)）。
- **示例**：

  ```go
  names := []string{"Bob", "Alice", "David", "Charlie"}
  sort.Strings(names)
  fmt.Println(names) // [Alice Bob Charlie David]
  ```

- **场景**：对一组名称、关键字、标签等需要字典排序时。

#### 3. `sort.Float64s`

```go
func Float64s(a []float64)
```

- **作用**：对 `[]float64` 原地升序排序。相同算法。
- **示例**：

  ```go
  data := []float64{3.14, 2.71, 1.41, 0.577}
  sort.Float64s(data)
  fmt.Println(data) // [0.577 1.41 2.71 3.14]
  ```

---

### 三、通用切片排序

基础类型函数虽然方便，但在对结构体切片或需要自定义排序规则时就无能为力了。此时可使用：

#### 1. `sort.Slice`

```go
func Slice(x interface{}, less func(i, j int) bool)
```

- **作用**：对任意切片 `x` 排序，调用 `less(i,j)` 决定元素 i 是否应排在元素 j 前面。采用快速排序（非稳定）。
- **示例**：对结构体切片按字段排序

  ```go
  type User struct { Name string; Age int }
  users := []User{
    {"Bob", 30}, {"Alice", 25}, {"Charlie", 28},
  }
  sort.Slice(users, func(i, j int) bool {
    return users[i].Age < users[j].Age
  })
  // users: [{Alice 25} {Charlie 28} {Bob 30}]
  ```

- **技巧**：
  - **闭包捕获**：`less` 闭包中可捕获外部变量，用于多条件排序（先按年龄，再按名字）。
  - **性能考虑**：对大型切片多次调用 `less`，要确保比较逻辑尽量轻量；必要时预先提取字段到临时切片。

#### 2. `sort.SliceStable`

```go
func SliceStable(x interface{}, less func(i, j int) bool)
```

- **作用**：与 `Slice` 类似，但保证相等元素的相对顺序不变（稳定排序）。内部采取归并排序，时间复杂度 O(n log n) 且需要 O(n) 额外空间。
- **示例**：先按城市排序，再按姓名排序，且同城市内保留先后顺序

  ```go
  type Person struct { Name, City string }
  people := []Person{
    {"Bob", "NY"}, {"Alice", "LA"}, {"Charlie", "NY"}, {"David", "LA"},
  }
  // 稳定地先按 City 排序
  sort.SliceStable(people, func(i, j int) bool {
    return people[i].City < people[j].City
  })
  // 结果：[{Alice LA}, {David LA}, {Bob NY}, {Charlie NY}]
  ```

- **场景**：多轮排序中需要保留前次排序结果。

---

### 四、稳定排序与反向排序

#### 1. `sort.Stable`

```go
func Stable(data Interface)
```

- **作用**：对实现了 `sort.Interface` 接口的 `data` 进行稳定排序（归并排序）。
- **示例**：对自定义结构体切片

  ```go
  type Record struct { Key, Value int }
  records := []Record{{1, 100}, {2, 200}, {1, 150}}
  // 按 Key 升序，且同 Key 保留原顺序
  sort.Stable(sort.Reverse(sortByKey(records)))
  ```

  其中 `sortByKey` 返回一个实现了 Interface 的包装类型，详见下文。

#### 2. `sort.Reverse`

```go
func Reverse(data Interface) Interface
```

- **作用**：将任意实现了 `sort.Interface` 的排序顺序反转，常用于降序排序。
- **示例**：对 `[]int` 降序

  ```go
  nums := []int{3,1,4,2}
  sort.Sort(sort.Reverse(sort.IntSlice(nums)))
  fmt.Println(nums) // [4 3 2 1]
  ```

---

### 五、自定义排序：`sort.Interface`

若既不想用 `Slice`，又想用更底层的能力，可直接实现：

```go
type Interface interface {
    Len() int
    Less(i, j int) bool
    Swap(i, j int)
}
```

#### 1. 内置类型实现

- `sort.IntSlice`、`sort.StringSlice`、`sort.Float64Slice`，均实现了 `Interface`，可用于 `sort.Sort`、`sort.Stable`、`sort.Reverse`。

#### 2. 自定义类型示例

```go
type Person struct { Name string; Age int }

// 定义一个 PersonSlice，实现 Interface
type ByAge []Person

func (p ByAge) Len() int           { return len(p) }
func (p ByAge) Less(i, j int) bool { return p[i].Age < p[j].Age }
func (p ByAge) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// 调用
people := []Person{{"Bob",30}, {"Alice",25}, {"Charlie",28}}
sort.Sort(ByAge(people))
```

- **技巧**：
  - 将多字段排序逻辑写在 `Less` 中；必要时可构建多字段比较函数链。
  - 对极端性能场景，可在切片外部维护关键字段数组，减少 `Less` 中结构体访问成本。

---

### 六、二分查找

`sort` 包也提供了高效的二分查找函数，前提是切片已排好序。

#### 1. 通用查找：`sort.Search`

```go
func Search(n int, f func(int) bool) int
```

- **作用**：在 `[0,n)` 范围内查找最小的 `i` 使得 `f(i)==true`，否则返回 `n`。适合查找“第一个满足条件”的场景。
- **示例**：在递增数组中找第一个 ≥ target 的下标

  ```go
  a := []int{1,3,5,7,9}
  target := 6
  i := sort.Search(len(a), func(i int) bool { return a[i] >= target })
  // i == 3，对应 a[3]=7
  ```

#### 2. 类型专用查找

- `SearchInts(a []int, x int) int`
- `SearchStrings(a []string, x string) int`
- `SearchFloat64s(a []float64, x float64) int`

它们内部直接在排序好的切片上执行二分查找，返回下标。

---

### 七、实战技巧汇总

1. **避免中途修改切片**：排序前后切片底层指针不变，如需保持原序列，可先 `dup := append([]T(nil), src...)` 复制一份再排。
2. **稳定 vs 非稳定**：若要多关键字排序，优先用 `SliceStable` 或 `sort.Stable`，分多轮从最次要到最重要字段排序，保证稳定性。
3. **自定义类型排序**：结构体较大时，提前提取排序关键字段到并行小切片，减少 `Less` 中内存访问开销。
4. **反向排序**：对基础切片用包装类型（如 `sort.IntSlice`）配合 `sort.Reverse`，对通用 `Slice` 可将比较函数取反 `less := func(i,j){ return a[i]>a[j] }`。
5. **查找边界**：利用 `Search` 函数快速定位区间上下界（lower/upper bound），可简化区间计数等操作。
6. **接口 vs 函数式**：在代码简洁性和性能之间权衡：小切片可优先用 `sort.Slice`，大规模、多次排序则自定义 `Interface` 并内联 `Less`，更易编译器优化。
7. **并行排序**（Go 1.21+）：可考虑第三方库（如 `golang.org/x/exp/slices`），或自行分块并发排序后归并，但需注意内存和锁开销。
