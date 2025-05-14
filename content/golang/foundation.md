+++
date = '2025-04-04T19:24:42+08:00'
draft = false
title = 'Go 语言基础知识'
+++

# Go 语言基础知识大全

## 目录
1. [环境配置](#环境配置)
2. [基本数据类型](#基本数据类型)
3. [变量和常量](#变量和常量)
4. [运算符](#运算符)
5. [控制结构](#控制结构)
6. [复合数据类型](#复合数据类型)
7. [函数](#函数)
8. [方法和接口](#方法和接口)
9. [错误处理](#错误处理)
10. [并发编程](#并发编程)
11. [输入输出](#输入输出)
12. [包和模块](#包和模块)
13. [最佳实践](#最佳实践)

## 环境配置

### Go 代理设置
```go
go env -w GOPROXY=https://goproxy.cn,direct
go env -w GOPROXY=https://gocenter.io,direct
```

### 私有库配置
```bash
go env -w GOPRIVATE=your-private-repo.com
git config --global url."https://username:token@private-repo.com".insteadOf "https://private-repo.com"
```

## 基本数据类型

### 整数类型
- **有符号整数**: `int8`, `int16`, `int32`(`rune`), `int64`, `int`
- **无符号整数**: `uint8`(`byte`), `uint16`, `uint32`, `uint64`, `uint`, `uintptr`

### 浮点数和复数
```go
// 浮点数
var f1 float32 = 3.14
var f2 float64 = 3.14159

// 复数
var x complex128 = complex(1, 2) // 1+2i
fmt.Println(real(x*y))   // 提取实部
fmt.Println(imag(x*y))   // 提取虚部
```

### 布尔类型
```go
var b bool = true
// && 的优先级高于 ||
if 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' {
    // 处理字母
}
```

### 字符串
```go
// 字符串操作
s := "hello world"
len(s)           // 字节数
utf8.RuneCountInString(s)  // 字符数

// 修改字符串
c := []byte(s)
c[0] = 'H'
s2 := string(c)

// 字符串连接（高效方式）
var buffer bytes.Buffer
buffer.WriteString("hello")
buffer.WriteString(" world")
result := buffer.String()
```

## 变量和常量

### 变量声明的四种方式
```go
// 1. 短变量声明
s := "hello"

// 2. var 声明
var s string

// 3. var 声明带初始化
var s = "hello"

// 4. var 声明带类型和初始化
var s string = "hello"
```

### 常量和 iota
```go
const Pi = 3.14159

// iota 枚举器
type Weekday int
const (
    Sunday Weekday = iota  // 0
    Monday                 // 1
    Tuesday                // 2
    Wednesday              // 3
    Thursday               // 4
    Friday                 // 5
    Saturday               // 6
)
```

## 运算符

### 算术运算符
```go
+ - * / %
++ --
```

### 位运算符
```go
&     // AND
|     // OR
^     // XOR (一元时表示按位取反)
&^    // AND NOT (位清除)
<<    // 左移
>>    // 右移

// 实例
var x uint8 = 1<<1 | 1<<5  // 00100010
var y uint8 = 1<<1 | 1<<2  // 00000110
fmt.Printf("%08b\n", x&y)  // 00000010
fmt.Printf("%08b\n", x|y)  // 00100110
fmt.Printf("%08b\n", x^y)  // 00100100
fmt.Printf("%08b\n", x&^y) // 00100000
```

### 比较运算符
```go
== != < <= > >=
```

### 逻辑运算符
```go
&& || !
```

## 控制结构

### if 语句
```go
if condition {
    // 代码块
} else if condition2 {
    // 代码块
} else {
    // 代码块
}

// if 初始化语句
if err := function(); err != nil {
    return err
}
```

### switch 语句
```go
// 表达式 switch
switch value {
case 1:
    fmt.Println("one")
case 2, 3:
    fmt.Println("two or three")
default:
    fmt.Println("other")
}

// 类型 switch
switch v := x.(type) {
case string:
    fmt.Printf("字符串: %s\n", v)
case int:
    fmt.Printf("整数: %d\n", v)
default:
    fmt.Printf("未知类型\n")
}
```

### for 循环
```go
// 基本 for 循环
for i := 0; i < 10; i++ {
    fmt.Println(i)
}

// while 风格
for condition {
    // 代码块
}

// 无限循环
for {
    // 代码块
}

// range 循环
for index, value := range slice {
    fmt.Printf("%d: %v\n", index, value)
}
```

## 复合数据类型

### 数组
```go
// 数组声明和初始化
var arr [5]int
arr2 := [5]int{1, 2, 3, 4, 5}
arr3 := [...]int{1, 2, 3}  // 编译器推断长度

// 数组是值类型，传递时会复制
func modifyArray(arr *[5]int) {
    arr[0] = 100
}
```

### 切片 (Slice)
```go
// 创建切片
s1 := make([]int, 5)      // 长度为5
s2 := make([]int, 5, 10)  // 长度5，容量10
s3 := []int{1, 2, 3}

// 切片操作
s1 = append(s1, 6)        // 追加元素
copy(dest, src)           // 复制切片

// 切片表达式
s[low:high]      // 包含low，不包含high
s[low:]          // 从low到末尾
s[:high]         // 从开始到high
s[:]             // 整个切片

// 删除元素
func remove(slice []int, i int) []int {
    copy(slice[i:], slice[i+1:])
    return slice[:len(slice)-1]
}
```

### 映射 (Map)
```go
// 创建和初始化
m1 := make(map[string]int)
m2 := map[string]int{
    "alice": 25,
    "bob":   30,
}

// 操作
m1["key"] = value
value := m1["key"]
delete(m1, "key")

// 检查键是否存在
if value, ok := m1["key"]; ok {
    fmt.Println("存在:", value)
}

// 遍历
for key, value := range m1 {
    fmt.Printf("%s: %d\n", key, value)
}
```

### 结构体
```go
// 定义结构体
type Person struct {
    Name string
    Age  int
    addr *Address  // 嵌入指针
}

type Address struct {
    City, Country string
}

// 创建结构体
p1 := Person{"Alice", 25, &Address{"Shanghai", "China"}}
p2 := Person{
    Name: "Bob",
    Age:  30,
}

// 构造函数模式
func NewPerson(name string, age int) *Person {
    return &Person{
        Name: name,
        Age:  age,
    }
}

// 匿名字段（嵌入）
type Employee struct {
    Person        // 匿名字段
    ID     int
}
```

## 函数

### 函数定义
```go
// 基本语法
func functionName(param1 type1, param2 type2) (returnType1, returnType2) {
    return value1, value2
}

// 命名返回值
func divide(a, b int) (quotient, remainder int) {
    quotient = a / b
    remainder = a % b
    return  // 裸return
}

// 可变参数
func sum(nums ...int) int {
    total := 0
    for _, num := range nums {
        total += num
    }
    return total
}
```

### 匿名函数和闭包
```go
// 匿名函数
func() {
    fmt.Println("Hello")
}()

// 闭包
func counter() func() int {
    count := 0
    return func() int {
        count++
        return count
    }
}

c := counter()
fmt.Println(c()) // 1
fmt.Println(c()) // 2
```

### defer 语句
```go
func readFile(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()  // 确保文件关闭
    
    // 读取文件内容
    return nil
}

// defer 的执行顺序是后进先出 (LIFO)
func example() {
    defer fmt.Println("1")
    defer fmt.Println("2")
    defer fmt.Println("3")
    // 输出: 3 2 1
}
```

### 高阶函数
```go
// 函数作为参数
func apply(nums []int, fn func(int) int) []int {
    result := make([]int, len(nums))
    for i, v := range nums {
        result[i] = fn(v)
    }
    return result
}

// 使用
doubled := apply([]int{1, 2, 3}, func(x int) int { return x * 2 })
```

## 方法和接口

### 方法
```go
type Rectangle struct {
    Width, Height float64
}

// 值接收者 - 不能修改接收者
func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

// 指针接收者 - 可以修改接收者
func (r *Rectangle) Scale(factor float64) {
    r.Width *= factor
    r.Height *= factor
}

// 字符串表示方法
func (r Rectangle) String() string {
    return fmt.Sprintf("Rectangle(%v, %v)", r.Width, r.Height)
}
```

### 接口
```go
// 接口定义
type Stringer interface {
    String() string
}

type Sizer interface {
    Size() int
}

// 接口组合
type Printer interface {
    Stringer
    Print()
}

// 空接口
var x interface{}
x = 42
x = "hello"
x = []int{1, 2, 3}
```

### 类型断言和类型开关
```go
// 类型断言
var i interface{} = "hello"
s, ok := i.(string)
if ok {
    fmt.Println("字符串:", s)
}

// 必须断言（可能panic）
s := i.(string)

// 类型开关
switch v := i.(type) {
case string:
    fmt.Printf("字符串: %q\n", v)
case int:
    fmt.Printf("整数: %d\n", v)
default:
    fmt.Printf("未知类型: %T\n", v)
}
```

## 错误处理

### 错误接口
```go
type error interface {
    Error() string
}

// 创建错误
err := errors.New("something went wrong")
err := fmt.Errorf("operation failed: %w", originalErr)  // 错误包装
```

### 错误处理策略

#### 1. 传播错误
```go
func readConfig(filename string) (Config, error) {
    data, err := ioutil.ReadFile(filename)
    if err != nil {
        return Config{}, fmt.Errorf("read config: %w", err)
    }
    // 处理配置...
    return config, nil
}
```

#### 2. 重试机制
```go
func WaitForServer(url string) error {
    const timeout = 1 * time.Minute
    deadline := time.Now().Add(timeout)
    
    for tries := 0; time.Now().Before(deadline); tries++ {
        _, err := http.Get(url)
        if err == nil {
            return nil
        }
        log.Printf("server not responding (%s); retrying...", err)
        time.Sleep(time.Second << uint(tries))
    }
    return fmt.Errorf("server %s failed to respond after %s", url, timeout)
}
```

#### 3. 记录并继续
```go
if err := doSomething(); err != nil {
    log.Printf("warning: %v", err)
    // 继续执行...
}
```

#### 4. 优雅退出
```go
if err := criticalOperation(); err != nil {
    log.Fatalf("critical error: %v", err)
}
```

### panic 和 recover
```go
func safeDivide(a, b int) (result int, err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic occurred: %v", r)
        }
    }()
    
    if b == 0 {
        panic("division by zero")
    }
    result = a / b
    return
}
```

## 并发编程

### Goroutine
```go
// 启动协程
go function()
go func() {
    fmt.Println("Hello from goroutine")
}()

// 避免主程序过早退出
func main() {
    go task()
    time.Sleep(time.Second)  // 不推荐
}
```

### Channel
```go
// 无缓冲通道（同步）
ch := make(chan int)

// 有缓冲通道（异步）
ch := make(chan int, 10)

// 发送和接收
ch <- 42         // 发送
value := <-ch    // 接收
value, ok := <-ch // 接收，ok表示通道是否关闭

// 关闭通道
close(ch)

// 检查通道是否关闭
for value := range ch {
    fmt.Println(value)
}
```

### 通道方向
```go
// 只发送通道
func send(ch chan<- int) {
    ch <- 42
}

// 只接收通道
func receive(ch <-chan int) {
    value := <-ch
}
```

### select 语句
```go
select {
case msg1 := <-ch1:
    fmt.Println("received from ch1:", msg1)
case msg2 := <-ch2:
    fmt.Println("received from ch2:", msg2)
case <-time.After(1 * time.Second):
    fmt.Println("timeout")
default:
    fmt.Println("no channel ready")
}
```

### 同步原语

#### sync.WaitGroup
```go
var wg sync.WaitGroup

for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(i int) {
        defer wg.Done()
        fmt.Println("Worker", i)
    }(i)
}

wg.Wait()  // 等待所有协程完成
```

#### sync.Mutex
```go
type Counter struct {
    mu    sync.Mutex
    value int
}

func (c *Counter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.value++
}

func (c *Counter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.value
}
```

#### sync.RWMutex
```go
type SafeMap struct {
    mu sync.RWMutex
    m  map[string]int
}

func (sm *SafeMap) Read(key string) int {
    sm.mu.RLock()
    defer sm.mu.RUnlock()
    return sm.m[key]
}

func (sm *SafeMap) Write(key string, value int) {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    sm.m[key] = value
}
```

### 并发模式

#### 生产者-消费者
```go
func producer(ch chan<- int) {
    for i := 0; i < 10; i++ {
        ch <- i
    }
    close(ch)
}

func consumer(ch <-chan int) {
    for value := range ch {
        fmt.Println("consumed:", value)
    }
}

func main() {
    ch := make(chan int)
    go producer(ch)
    consumer(ch)
}
```

#### Fan-in/Fan-out
```go
// Fan-out: 分发任务到多个协程
func fanOut(input <-chan int, workers int) []<-chan int {
    outputs := make([]<-chan int, workers)
    for i := 0; i < workers; i++ {
        output := make(chan int)
        outputs[i] = output
        go func() {
            defer close(output)
            for n := range input {
                output <- process(n)
            }
        }()
    }
    return outputs
}

// Fan-in: 合并多个协程的结果
func fanIn(channels ...<-chan int) <-chan int {
    output := make(chan int)
    var wg sync.WaitGroup
    
    for _, ch := range channels {
        wg.Add(1)
        go func(ch <-chan int) {
            defer wg.Done()
            for value := range ch {
                output <- value
            }
        }(ch)
    }
    
    go func() {
        wg.Wait()
        close(output)
    }()
    
    return output
}
```

## 输入输出

### 格式化输出
```go
// Printf 常用格式
%v    // 默认格式
%+v   // 带字段名的结构体
%#v   // Go语法表示
%T    // 类型
%t    // 布尔值
%d    // 十进制整数
%b    // 二进制
%o    // 八进制
%x    // 十六进制（小写）
%X    // 十六进制（大写）
%f    // 浮点数
%e    // 科学计数法（小写e）
%E    // 科学计数法（大写E）
%s    // 字符串
%q    // 带引号的字符串
%p    // 指针
```

### 输入操作
```go
// 标准输入
var name string
var age int
fmt.Scan(&name, &age)
fmt.Scanf("%s %d", &name, &age)

// 从字符串读取
input := "Alice 25"
fmt.Sscanf(input, "%s %d", &name, &age)

// bufio 读取
reader := bufio.NewReader(os.Stdin)
line, err := reader.ReadString('\n')
```

### 文件操作

#### 读取文件
```go
// 一次性读取整个文件
data, err := ioutil.ReadFile("file.txt")
if err != nil {
    log.Fatal(err)
}

// 逐行读取
file, err := os.Open("file.txt")
if err != nil {
    log.Fatal(err)
}
defer file.Close()

scanner := bufio.NewScanner(file)
for scanner.Scan() {
    line := scanner.Text()
    fmt.Println(line)
}

// 使用 bufio.Reader
reader := bufio.NewReader(file)
for {
    line, err := reader.ReadString('\n')
    if err == io.EOF {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    fmt.Print(line)
}
```

#### 写入文件
```go
// 一次性写入
data := []byte("Hello, World!")
err := ioutil.WriteFile("output.txt", data, 0644)

// 使用 bufio.Writer
file, err := os.Create("output.txt")
if err != nil {
    log.Fatal(err)
}
defer file.Close()

writer := bufio.NewWriter(file)
writer.WriteString("Hello, World!\n")
writer.Flush()

// 格式化写入
fmt.Fprintf(file, "Name: %s, Age: %d\n", name, age)
```

#### 文件复制
```go
func copyFile(src, dst string) error {
    source, err := os.Open(src)
    if err != nil {
        return err
    }
    defer source.Close()
    
    destination, err := os.Create(dst)
    if err != nil {
        return err
    }
    defer destination.Close()
    
    _, err = io.Copy(destination, source)
    return err
}
```

### JSON 处理
```go
type Person struct {
    Name string `json:"name"`
    Age  int    `json:"age"`
}

// 编码到 JSON
person := Person{Name: "Alice", Age: 25}
data, err := json.Marshal(person)

// 解码 JSON
var person Person
err := json.Unmarshal(data, &person)

// 从文件读取 JSON
file, err := os.Open("data.json")
defer file.Close()
decoder := json.NewDecoder(file)
err = decoder.Decode(&person)

// 写入 JSON 到文件
file, err := os.Create("output.json")
defer file.Close()
encoder := json.NewEncoder(file)
err = encoder.Encode(person)
```

## 包和模块

### 包的基本概念
```go
// 包声明（每个文件必须有）
package main

// 导入包
import (
    "fmt"
    "strconv"
    "net/http"
    
    // 别名
    str "strings"
    
    // 点导入（导入到当前命名空间）
    . "math"
    
    // 仅执行初始化
    _ "net/http/pprof"
)
```

### Go Modules
```bash
# 初始化模块
go mod init module-name

# 添加依赖
go get github.com/user/repo
go get github.com/user/repo@version

# 更新依赖
go get -u ./...

# 清理未使用的依赖
go mod tidy

# 查看依赖
go list -m all

# 下载依赖到本地缓存
go mod download
```

### go.mod 文件
```go
module example.com/myapp

go 1.19

require (
    github.com/gin-gonic/gin v1.8.1
    gorm.io/gorm v1.23.0
)

replace (
    github.com/old/repo => github.com/new/repo v1.0.0
    github.com/local/repo => ./local/path
)

exclude github.com/broken/repo v1.0.0
```

### 包的初始化
```go
var globalVar = initFunction()

func init() {
    // 包初始化代码
    // 在main函数之前执行
    // 可以有多个init函数
}

func initFunction() int {
    return 42
}
```

## 类型系统

### 类型定义
```go
// 类型别名
type UserID int
type Username string

// 类型定义
type User struct {
    ID   UserID
    Name Username
}

// 接口定义
type Reader interface {
    Read([]byte) (int, error)
}

// 函数类型
type Handler func(http.ResponseWriter, *http.Request)
```

### 类型断言和转换
```go
// 类型转换（同底层类型）
var x int = 42
var y float64 = float64(x)

// 类型断言（接口到具体类型）
var i interface{} = "hello"
s := i.(string)

// 安全类型断言
if s, ok := i.(string); ok {
    fmt.Println(s)
}
```

### 嵌入和组合
```go
// 结构体嵌入
type Engine struct {
    Power int
}

func (e Engine) Start() {
    fmt.Println("Engine started")
}

type Car struct {
    Engine  // 嵌入Engine
    Brand string
}

// Car 自动获得 Engine 的方法
car := Car{Engine{100}, "Toyota"}
car.Start()  // 调用 Engine.Start()
```

## 反射

### 基本反射操作
```go
import "reflect"

func examine(x interface{}) {
    v := reflect.ValueOf(x)
    t := reflect.TypeOf(x)
    
    fmt.Printf("Type: %s\n", t)
    fmt.Printf("Value: %v\n", v)
    fmt.Printf("Kind: %s\n", v.Kind())
}

// 修改值（需要传递指针）
func modify(x interface{}) {
    v := reflect.ValueOf(x).Elem()
    if v.CanSet() {
        v.SetInt(100)
    }
}

var n int = 42
modify(&n)  // n 变为 100
```

### 结构体反射
```go
type Person struct {
    Name string `json:"name" tag:"example"`
    Age  int    `json:"age"`
}

func inspectStruct(x interface{}) {
    v := reflect.ValueOf(x)
    t := reflect.TypeOf(x)
    
    for i := 0; i < v.NumField(); i++ {
        field := t.Field(i)
        value := v.Field(i)
        
        fmt.Printf("Field: %s, Type: %s, Value: %v\n",
            field.Name, field.Type, value)
        
        // 获取标签
        tag := field.Tag.Get("json")
        fmt.Printf("JSON tag: %s\n", tag)
    }
}
```

## 测试

### 单元测试
```go
// math.go
func Add(a, b int) int {
    return a + b
}

// math_test.go
package main

import "testing"

func TestAdd(t *testing.T) {
    result := Add(2, 3)
    expected := 5
    if result != expected {
        t.Errorf("Add(2, 3) = %d; want %d", result, expected)
    }
}

// 表格驱动测试
func TestAddTable(t *testing.T) {
    tests := []struct {
        a, b, expected int
    }{
        {1, 2, 3},
        {0, 0, 0},
        {-1, 1, 0},
    }
    
    for _, test := range tests {
        result := Add(test.a, test.b)
        if result != test.expected {
            t.Errorf("Add(%d, %d) = %d; want %d",
                test.a, test.b, result, test.expected)
        }
    }
}
```

### 基准测试
```go
func BenchmarkAdd(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(1, 2)
    }
}

// 运行基准测试
// go test -bench=.
// go test -bench=BenchmarkAdd -benchmem
```

### 示例测试
```go
func ExampleAdd() {
    fmt.Println(Add(1, 2))
    // Output: 3
}
```

## 工具和命令

### 常用命令
```bash
# 编译和运行
go run main.go
go build
go build -o myapp
go install

# 测试
go test
go test -v
go test -cover
go test -bench=.

# 格式化代码
go fmt ./...
goimports -w .

# 代码检查
go vet ./...
golint ./...
golangci-lint run

# 依赖管理
go mod tidy
go mod vendor
go mod verify

# 查看文档
go doc fmt.Println
godoc -http=:6060

# 性能分析
go test -cpuprofile=cpu.prof
go test -memprofile=mem.prof
go tool pprof cpu.prof
```

### 编译选项
```bash
# 交叉编译
GOOS=linux GOARCH=amd64 go build
GOOS=windows GOARCH=amd64 go build
GOOS=darwin GOARCH=amd64 go build

# 编译优化
go build -ldflags="-s -w"  # 去除符号表和调试信息
go build -tags=release     # 条件编译
```

## 最佳实践

### 命名规范
1. **包名**: 小写，简短，有意义
2. **变量**: 驼峰命名法，首字母小写
3. **常量**: 驼峰命名法或全大写
4. **函数**: 驼峰命名法，首字母大写表示导出
5. **接口**: 通常以 -er 结尾（如 Reader, Writer）

### 代码组织
```go
// 推荐的文件组织方式
myapp/
├── cmd/           # 应用程序入口
│   └── server/
│       └── main.go
├── internal/      # 私有代码
│   ├── config/
│   ├── handler/
│   └── service/
├── pkg/           # 公共库代码
├── api/           # API 定义
├── web/           # Web 资源
├── configs/       # 配置文件
├── scripts/       # 脚本
├── tests/         # 额外的测试文件
├── docs/          # 文档
├── go.mod
├── go.sum
└── README.md
```

### 性能优化
1. **预分配切片容量**
```go
// 好的做法
slice := make([]int, 0, 100)

// 避免频繁扩容
for i := 0; i < 100; i++ {
    slice = append(slice, i)
}
```

2. **字符串拼接**
```go
// 低效
var result string
for _, s := range strings {
    result += s
}

// 高效
var builder strings.Builder
for _, s := range strings {
    builder.WriteString(s)
}
result := builder.String()
```

3. **避免不必要的内存分配**
```go
// 重用切片
func processData(data []int) {
    buffer := make([]int, 0, len(data))
    // 使用 buffer...
    buffer = buffer[:0]  // 重置长度，保留容量
}
```

### 常见陷阱

#### 1. 循环变量捕获
```go
// 错误
for _, item := range items {
    go func() {
        fmt.Println(item) // 总是打印最后一个 item
    }()
}

// 正确
for _, item := range items {
    go func(item string) {
        fmt.Println(item)
    }(item)
}
```

#### 2. defer 在循环中
```go
// 错误 - defer 会在函数结束时执行，可能导致资源泄漏
func processFiles(files []string) {
    for _, file := range files {
        f, err := os.Open(file)
        if err != nil {
            continue
        }
        defer f.Close() // 在循环中，defer 会累积
        // 处理文件...
    }
}

// 正确
func processFiles(files []string) {
    for _, file := range files {
        func() {
            f, err := os.Open(file)
            if err != nil {
                return
            }
            defer f.Close()
            // 处理文件...
        }()
    }
}
```

#### 3. 切片和映射的零值
```go
// 错误
var slice []int
slice[0] = 1  // panic: index out of range

var m map[string]int
m["key"] = 1  // panic: assignment to entry in nil map

// 正确
slice := make([]int, 10)
slice[0] = 1

m := make(map[string]int)
m["key"] = 1
```

### 错误处理最佳实践
1. **不要忽略错误**
2. **在错误中提供上下文**
3. **使用错误包装**（Go 1.13+）
4. **定义自定义错误类型**

```go
// 自定义错误类型
type ValidationError struct {
    Field string
    Value interface{}
    Tag   string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation failed on field '%s' with value '%v'", e.Field, e.Value)
}
```

### 并发安全
1. **使用通道传递数据**
2. **使用互斥锁保护共享状态**
3. **避免共享内存，通过通信共享**
4. **使用 context 控制协程生命周期**

## 常用标准库

### 字符串处理 (strings)
```go
strings.Contains(s, substr)
strings.HasPrefix(s, prefix)
strings.HasSuffix(s, suffix)
strings.Split(s, sep)
strings.Join(slice, sep)
strings.ToLower(s)
strings.ToUpper(s)
strings.TrimSpace(s)
strings.Replace(s, old, new, n)
```

### 时间处理 (time)
```go
now := time.Now()
time.Sleep(time.Second)
future := now.Add(time.Hour)
duration := future.Sub(now)

// 格式化时间
formatted := now.Format("2006-01-02 15:04:05")
// 解析时间
parsed, err := time.Parse("2006-01-02", "2023-01-01")
```

### 正则表达式 (regexp)
```go
pattern := `\d+`
re, err := regexp.Compile(pattern)
if err != nil {
    log.Fatal(err)
}

matched := re.MatchString("abc123def")
matches := re.FindAllString("123 456 789", -1)
replaced := re.ReplaceAllString("abc123def", "XXX")
```

### HTTP 客户端/服务器
```go
// HTTP 服务器
http.HandleFunc("/", handler)
log.Fatal(http.ListenAndServe(":8080", nil))

// HTTP 客户端
resp, err := http.Get("https://api.example.com")
if err != nil {
    log.Fatal(err)
}
defer resp.Body.Close()

body, err := ioutil.ReadAll(resp.Body)
```

### 数据库操作
```go
import "database/sql"
import _ "github.com/lib/pq"

db, err := sql.Open("postgres", connectionString)
defer db.Close()

// 查询
rows, err := db.Query("SELECT id, name FROM users")
defer rows.Close()

for rows.Next() {
    var id int
    var name string
    err := rows.Scan(&id, &name)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("ID: %d, Name: %s\n", id, name)
}

// 插入
result, err := db.Exec("INSERT INTO users (name) VALUES ($1)", "Alice")
```

## 部署和容器化

### Dockerfile
```dockerfile
# 多阶段构建
FROM golang:1.19-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o main .

# 最终镜像
FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/main .
CMD ["./main"]
```

### 优雅关闭
```go
func main() {
    // 创建 HTTP 服务器
    srv := &http.Server{
        Addr:    ":8080",
        Handler: router,
    }
    
    go func() {
        if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Fatalf("listen: %s\n", err)
        }
    }()
    
    // 等待中断信号
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit
    log.Println("Shutdown Server ...")
    
    // 优雅关闭
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    if err := srv.Shutdown(ctx); err != nil {
        log.Fatal("Server Shutdown:", err)
    }
    log.Println("Server exiting")
}
```
