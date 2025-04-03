---
title: "Linux基础知识"
date: 2024-04-03
weight: 6
description: "Linux操作系统基础与常用命令"
---

Linux 是程序员的必修课程。作为一个合格的程序员，需要理解 Linux 的基础概念和常用命令。以下是一些核心概念以及二三十个常用到的命令：


### 进程管理
- 进程调度（Process Scheduling）：CPU 时间片分配机制
- 进程状态：运行、就绪、阻塞等状态转换
- 进程间通信（IPC）：管道、信号、共享内存等

### 文件系统
- inode：文件元数据存储机制
- 文件权限：读（r）、写（w）、执行（x）权限管理
- 文件系统层次：ext4、XFS、Btrfs 等

### 系统监控
- inotify：文件系统事件监控机制
- systemd：系统和服务管理器
- cgroups：资源限制和隔离

### 网络
- Socket 编程：网络通信基础
- iptables：防火墙规则管理
- 网络协议栈：TCP/IP 实现

### 内存管理
- 虚拟内存：物理内存和交换空间
- 内存分页：页表和地址转换
- OOM（Out of Memory）处理机制

### 常用命令

| 1.dmesg | 2.dd             | 3.tcpdump  | 4.ss           | 5.top                | 6.du                 |
| ------- | ---------------- | ---------- | -------------- | -------------------- | -------------------- |
| 7.iperf | 8.find           | 9.awk      | 10.sed         | 11.grep              | 12.route             |
| 13.ip   | 14.lsof/fuser    | 15.netstat | 16.rpm         | 17.dpkg              | 18.diff              |
| 19.ps   | 20.kill          | 21.unset   | 22.EOF >>      | 23.特殊符号: % $ # & | 24.until             |
| 25.cut  | 26.vmstat/mpstat | 27.free    | 28.curl & wget | 29.iptables          | 30.nmap              |
| 31.jq   | 32.sort          | 33.strace  | 34.uptime      | 35.iostat            | 36.sysctl/modelprobe |
| 37.sar  | 38.brctl         | 39.bridge  |                |                      |                      |