+++
date = '2025-05-28T14:21:33+08:00'
draft = true
title = '字符编码基础知识详解'
+++

## 目录

* [字符与字节的区别](#字符与字节的区别)
* [字符集与字符编码的区别](#字符集与字符编码的区别)
* [Unicode 与 UTF-8 的关系](#unicode-与-utf-8-的关系)
* [常见编码方案简介（ASCII、GB2312、UTF-16、UTF-32 等）](#常见编码方案简介asciigb2312utf-16utf-32-等)
* [各类编码的历史背景与设计动机](#各类编码的历史背景与设计动机)
* [UTF-8 编码规则与原理](#utf-8-编码规则与原理)
* [编程语言中字符与字节的处理（以 Go 为例）](#编程语言中字符与字节的处理以-go-为例)
* [实例说明“一个字符占多个字节”](#实例说明一个字符占多个字节)
* [代码示例：统计字符数、截取字符串与避免乱码](#代码示例统计字符数截取字符串与避免乱码)

## 字符与字节的区别

**字节**（Byte）是计算机存储容量的基本单位，由**8个二进制位**组成。一个字节可以表示0–255这256种不同的状态，用于衡量数据量的大小。**字符**（Character）是指书写系统中的最小符号单元，比如英文字母`A`、数字`5`、汉字`中`、标点符号“,”等。字符本身是逻辑上的概念，与存储多少字节并不固定相关。在不同的编码方式下，一个字符在存储时所占用的字节数是不一样的。例如，一般在 ASCII 或单字节编码中，一个英文字符占用1个字节，而一个汉字往往占用2个或更多字节。下表列出几种常见编码下字符所占字节数示例：

|        编码方式        |      英文字符（A）     |      常用汉字（中）     | 备注                         |
| :----------------: | :--------------: | :--------------: | :------------------------- |
|        ASCII       |    1 字节（0x41）    |         –        | ASCII 只定义英文字母、数字等；不包含汉字。   |
| GB2312/GBK/GB18030 |    1 字节（0x41）    |   2 字节（0xD6D0）   | GB2312 等兼容 ASCII，汉字用双字节表示。 |
|        UTF-8       |    1 字节（0x41）    |  3 字节（0xE4B8AD）  | 中文和其他非 ASCII 字符通常用3字节表示。   |
|       UTF-16       |   2 字节（0x0041）   |   2 字节（0x4E2D）   | BMP（基本多语言平面）内的字符占2字节。      |
|       UTF-32       | 4 字节（0x00000041） | 4 字节（0x00004E2D） | 所有字符固定4字节。                 |

上表可见，在 UTF-8 编码下，英文字符仍然占1个字节，而汉字通常占3个字节。在 UTF-16 编码下，BMP 内的中文字符占2字节；而 UTF-32 则一律用4字节存储任何字符。需要注意的是，字符是符号的抽象，计算机存储时会将字符映射成字节序列，但“字符”本身并不是一个存储单元。

## 字符集与字符编码的区别

**字符集**（Character Set）是字符的集合，定义了都有哪些字符。例如拉丁字母表、汉字集合都属于字符集。一个字符集可以包含某语言的所有字符或符号。**字符编码**（Character Encoding）是在字符集的基础上，给每个字符分配一个唯一的数字（称为码点）并定义如何将这个数字转换成计算机存储的字节序列的规则。换句话说，字符集只是列出了“有哪些字符”；而字符编码则把字符集里的每个字符映射到具体的数字值（代码单元），并规定了该数字在内存或文件中如何以字节形式排列。

简而言之，字符集好比“字符的字典”，字符编码则是“将字典里的字符转换为数字并编码为字节”的方法。例如，Unicode 提供了一个包含全球字符的**字符集**（每个字符都有唯一的码点）；而 UTF-8、UTF-16 等就是对这一字符集进行编码的不同**编码方案**。一个字符集（比如 Unicode）可以对应多种编码方式（UTF-8/UTF-16/UTF-32 等），它们的区别就在于如何安排码点到字节的映射。

## Unicode 与 UTF-8 的关系

**Unicode** 是一种字符集（或称为通用字符集）标准，它为世界上各种语言的字符分配了唯一的码点（一个数字编号）。每个字符（如汉字“中”或表情“😊”）在 Unicode 中都有一个唯一的码点，如 U+4E2D 对应“中”。**UTF-8** 则是针对 Unicode 码点的具体编码方式，将这些码点转换为字节序列存储。通俗地说，Unicode 定义了字符和码点的对照表，而 UTF-8 定义了把这些码点写入文件或网络传输时如何分解成字节。

UTF-8 是一种变长编码：它根据码点大小使用 1 到 4 个字节来表示一个 Unicode 码点。例如，Unicode 码点 U+0041（英文“A”）在 UTF-8 中编码为单字节 `0x41`（和 ASCII 完全一致），而码点 U+4E2D（汉字“中”）在 UTF-8 中编码为三字节 `0xE4 0xB8 0xAD`。这里需要强调的是，**Unicode 不是编码格式**，它更像是一张字符和号码的对照表；而 UTF-8 是实现这一对照表的编码方案之一。正如资料所述：“Unicode 给每个字符分配唯一的码点”；而“UTF-8 将码点编码成 1–4 个字节”。值得注意的是，UTF-8 与 ASCII 向前兼容：Unicode 的前128个码点（即U+0000到U+007F，对应 ASCII）在 UTF-8 中与 ASCII 相同。例如，英文字母“A”在 ASCII 和 UTF-8 中的编码都是单字节0x41。

综上：Unicode 是全球字符的集合和标准，UTF-8 是针对这个集合的可变长字节编码方式，它把码点以`0xxxxxxx`、`110xxxxx 10xxxxxx`等不同格式写入字节流。

## 常见编码方案简介（ASCII、GB2312、UTF-16、UTF-32 等）

* **ASCII**：最早的字符编码标准之一，由美国于1960年代制定，全称“美国信息交换标准代码”。它是7位编码，包含128个字符，包括英文字母、数字、基本符号和控制字符。由于历史原因，ASCII 的前128个字符后来成为 Unicode 和 UTF-8 的前128个码点，实现了向后兼容。后来出现了各种扩展 ASCII（8位编码），如 ISO-8859 系列和 Windows-1252 等，用剩余的128值表示其它语言字符和符号。但即使扩展 ASCII，也无法表示全球多种语言。

* **GB2312 / GBK / GB18030**：这是中国制定的汉字编码标准。GB2312 发布于1980年，定义了 6763 个常用简体汉字和682个符号，用两个字节（双字节）表示一个汉字。GB2312 与 ASCII 向下兼容，即用一个字节表示 ASCII 字符。后来出现 GBK（GB2312的扩展）和 GB18030（更全面的扩展）标准。GBK 向 GB2312 添加了近2万个新字符（包括繁体字和其它符号），GB18030 则成为最新强制性标准，向后兼容 GB2312/GBK 并新增更多汉字和少数民族文字。

* **UTF-16**：Unicode 的一种编码实现方式，使用16位（2字节）为基本单位。对于 BMP（基本多语言平面 U+0000–U+FFFF）内的字符，UTF-16 通常用一个16位单位表示；对于 BMP 以外的字符（即U+10000以上），UTF-16 使用一对“代理项”（surrogate pair）共4字节来表示。因此，UTF-16 是变长编码，常用字符占2字节，扩展字符占4字节。

* **UTF-32**：Unicode 的另一种编码实现方式，使用32位（4字节）固定长度表示每个字符。它等同于 UCS-4。在 UTF-32 中，每个 Unicode 码点都用4字节直接编码，不存在多字节的概念，因此长度计算很简单，但存储效率低。

下表总结了上述编码的一些特点：

| 编码          | 位宽/字节      | 描述                           |
| :---------- | :--------- | :--------------------------- |
| ASCII       | 7位/1字节     | 最早的英文字符编码（128字符），兼容UTF-8。    |
| GB2312      | 16位/2字节    | 中国国家标准简体汉字编码，双字节表示汉字。        |
| GBK/GB18030 | 可变 (1/2/4) | GB2312扩展，支持更多汉字和符号。          |
| UTF-16      | 16位单元      | Unicode编码实现，BMP内2字节，扩展字符4字节。 |
| UTF-32      | 32位/4字节    | Unicode编码实现，所有字符固定4字节。       |

## 各类编码的历史背景与设计动机

在计算机早期（上世纪60–70年代），主要面向英语国家，出现了如 ASCII（1960年代末）这类针对英文的编码标准。随着不同地区对本地语言支持的需求，各国又推出了各种单字节或双字节编码，例如欧洲国家用 ISO-8859 系列，中文环境下出现了 GB2312（1980年）、台湾地区有 Big5，日本有 Shift\_JIS 等。这些编码往往互不兼容，导致跨国交流困难。为解决全球文本交换问题，Unicode 与国际标准 ISO 10646 应运而生。1987年，Xerox 和 Apple 开始制定 Unicode 标准；1989年 ISO 启动 Universal Coded Character Set (UCS)，1990年发布草案。随后，Unicode 和 UCS 相互协调，最终成为今天统一的字符集标准。

UTF-8 的提出则是出于实际的实现需要。1992年，多位工程师（包括 Ken Thompson、Rob Pike 等）合作提出了一种效率高、与 ASCII 兼容的编码方案，称为“文件系统安全的 UCS 转换格式”（后改称 UTF-8），1993年于 USENIX 大会正式发布。UTF-8 的设计动机是：常用字符（如英文字母）仍然保持1字节编码，减少存储；而可以支持所有 Unicode 码点以多字节形式表现。1998年 IETF 将 UTF-8 定为互联网标准格式，到21世纪初，随着全球化进程加快，Unicode/UTF-8 成为绝大多数软件和互联网的默认编码。2003年起，IETF 甚至规定所有 Web 内容应使用 UTF-8 编码，以统一多语言环境。可以说，Unicode 和 UTF-8 的出现彻底解决了字符兼容性问题，实现了不同语言文本的一致表示和交换。

## UTF-8 编码规则与原理

UTF-8 是一种**变长编码**，它使用 1–4 个字节来表示一个 Unicode 码点。UTF-8 的编码规则如下：

* **1 字节编码（U+0000–U+007F）**：格式 `0xxxxxxx`，最高位为 0，其余7位直接存储码点值。这部分与 ASCII 完全相同。
* **2 字节编码（U+0080–U+07FF）**：格式 `110xxxxx 10xxxxxx`。第一个字节以 `110` 开头，后5位存储码点的高位；第二字节以 `10` 开头，后6位存储码点的低位。
* **3 字节编码（U+0800–U+FFFF）**：格式 `1110xxxx 10xxxxxx 10xxxxxx`。（示意图：**首字节**以`1110`开头，后续两个字节都以`10`开头。）
* **4 字节编码（U+10000–U+10FFFF）**：格式 `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx`。首字节以`11110`开头，其余三字节以`10`开头，剩余有效位存储码点值的各段。

图：UTF-8 三字节编码示意。首字节以`1110`开头，接下来的两个字节以`10`开头。

下表列出了 UTF-8 各种长度的编码范围和格式：

| UTF-8 字节数 |   码点范围 (U+ …)  |             UTF-8 编码格式（高位）            |
| :-------: | :------------: | :-----------------------------------: |
|    1 字节   |   0000 – 007F  |               `0xxxxxxx`              |
|    2 字节   |   0080 – 07FF  |          `110xxxxx 10xxxxxx`          |
|    3 字节   |   0800 – FFFF  |      `1110xxxx 10xxxxxx 10xxxxxx`     |
|    4 字节   | 10000 – 10FFFF | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` |

根据上述规则，UTF-8 编码能够自识别字节流中字符的边界：每个首字节根据前导位的1的个数确定整个字符占多少字节，其它字节都以 `10` 为前缀标记为续字节。例如，汉字“中”U+4E2D落在范围 0800–FFFF 内，会按 `1110xxxx 10xxxxxx 10xxxxxx` 格式编码（结果`0xE4 0xB8 0xAD`）；而英文字母`A`（U+0041）在 0000–007F 范围，编码为单字节 `0x41`。UTF-8 编码的设计保证了 ASCII 部分的兼容性和自同步性，即任何错误都会被检测到并容易定位。有关更多编码细节，可参考。

## 编程语言中字符与字节的处理（以 Go 为例）

在编程语言中，经常会混淆“字符”和“字节”的概念。以 Go 语言为例：Go 字符串 (`string`) 类型本质上是只读的字节切片，内部以 UTF-8 格式保存文本。这意味着，用 `len(s)` 返回的是**字节数**而非字符数，如果字符串里包含多字节字符，`len` 值会大于字符个数。例如，`len("世界")` 得到 6（每个汉字3字节），但实际字符长度为 2。Go 提供了 `rune` 类型（`int32` 的别名）来表示 Unicode 码点，并且可以将字符串转换为 `[]rune` 来正确处理字符。如下注意点应牢记：

* **长度计算**：对于 `string s`，`len(s)` 返回字节数（UTF-8 编码下每个字符占用的字节总和）。如果想得到字符（码点）个数，应使用 `utf8.RuneCountInString(s)` 或者先转换成 `[]rune` 再取长度。
* **索引和切片**：`s[i]` 返回的是第 `i` 个字节的值（`byte` 类型），不是字符。如果在一个多字节字符的中间位置进行切片（如 `s[:n]`），可能会截断一个字符，导致乱码。正确做法是先用 `[]rune` 或 `for range`（迭代获得每个 `rune`）来确保按字符操作。
* **字符串字面量**：Go 源代码默认以 UTF-8 编码保存。写在代码中的字符串字面量，其实际存储就是 UTF-8 的字节序列。例如，字符 `⌘`（U+2318）在代码中直接写为 `"\u2318"` 或原样写入时，其 UTF-8 编码 `0xE2 0x8C 0x98` 将被存入可执行文件。注意，尽管字符串字面量默认 UTF-8，但 `string` 类型可以存放任意字节序列（只要不违反语言规范），因此务必保证文本正确解码。

总之，在 Go 或其它语言中操作文本时，要区分字节和字符：一个字符可能由多个字节组成，应使用语言提供的 Unicode 支持（如 Go 的 `rune`、Python 的 Unicode 字符串、Java 的 `String` 等）来避免混淆。

## 实例说明“一个字符占多个字节”

为了加深理解，我们举些具体例子（以 UTF-8 编码为主）说明字符与字节数的关系：

* 英文字符`"A"`（U+0041）：UTF-8 编码为单字节 `0x41`，共占1字节。
* 汉字`"中"`（U+4E2D）：UTF-8 编码为三字节 `0xE4 0xB8 0xAD`，共占3字节。
* Emoji表情`"😊"`（U+1F60A）：UTF-8 编码为四字节 `0xF0 0x9F 0x98 0x8A`，共占4字节。

以上例子说明：同一个“字符”（指一个 Unicode 码点），根据编码方式的不同所占字节数也不同。汉字在常见的 UTF-8 中通常是3字节，而 Emoji、稀有字等可能占4字节。在 Go 或其它语言中处理时，必须知道这一点，否则简单地按字节截取往往会出错。以下表格对比了几种字符在不同编码下的存储情况（仅做示例，不含所有编码）：

|  字符  | Unicode 码点 | UTF-8 (hex) | UTF-8 字节数 | UTF-16 (bytes) | UTF-32 (bytes) |
| :--: | :--------: | :---------- | :-------: | :------------: | :------------: |
|  "A" |   U+0041   | 41          |     1     |   2 (0x0041)   | 4 (0x00000041) |
|  "中" |   U+4E2D   | E4 B8 AD    |     3     |   2 (0x4E2D)   | 4 (0x00004E2D) |
| "😊" |   U+1F60A  | F0 9F 98 8A |     4     |  4 (surrogate) | 4 (0x0001F60A) |

如上，ASCII字符 `"A"` 在 UTF-8 和 ASCII 中均占1字节，而汉字 `"中"` 在 UTF-8 中占3字节（UTF-16中占2字节），Emoji `"😊"` 在 UTF-8 中占4字节。不同编码下字符长度不一致需要特别注意。

## 代码示例统计字符数、截取字符串与避免乱码

下面给出一些 Go 代码示例，演示如何正确统计字符串中的字符数、如何截取，以及避免乱码的问题。

```go
package main

import (
    "fmt"
    "unicode/utf8"
)

func main() {
    s := "Go语言Gopher😊"
    fmt.Println(s) 
    // 统计字节数和字符数：
    fmt.Println("len(s) 字节数:", len(s))                       // 计算字节长度（UTF-8 编码）
    fmt.Println("utf8.RuneCountInString(s) 字符数:", utf8.RuneCountInString(s))
    // 使用 []rune 转换后获取字符数：
    runes := []rune(s)
    fmt.Println("len([]rune(s)) 字符数:", len(runes))
    // 演示按字符截取和按字节截取：
    fmt.Println("按字节切片 s[:8]:", s[:8])     // 可能在多字节字符中间截断，导致乱码
    fmt.Println("按字符切片 string([]rune(s)[:5]):", string(runes[:5]))
    // 正确截取前5个字符的方法
    fmt.Println("前5个字符（按rune）：", string(runes[:5]))
}
```
