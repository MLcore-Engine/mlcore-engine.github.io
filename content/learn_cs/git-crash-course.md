+++
date = '2025-05-22T09:15:13+08:00'
draft = false
title = 'git-crash-course'
+++

# 目录

1. [忽略文件与刷新缓存](#1-忽略文件与刷新缓存)
   - [添加忽略规则](#添加忽略规则)
   - [强制刷新 Git 缓存](#强制刷新-git-缓存让-gitignore-立即生效)

2. [常用 Git 命令](#2-常用-git-命令)
   - [初始化仓库](#初始化仓库)
   - [克隆远程仓库](#克隆远程仓库)
   - [查看当前状态](#查看当前状态)
   - [添加文件到暂存区](#添加文件到暂存区)
   - [提交更改](#提交更改)
   - [查看提交历史](#查看提交历史)
   - [推送到远程仓库](#推送到远程仓库)
   - [拉取远程更新](#拉取远程更新)

3. [其他实用技巧](#3-其他实用技巧)
   - [查看分支](#查看分支)
   - [创建新分支](#创建新分支)
   - [切换分支](#切换分支)
   - [合并分支](#合并分支)

4. [进阶操作命令](#4-进阶操作命令)
   - [撤销与重置（reset）](#1-撤销与重置reset)
   - [变基（rebase）](#2-变基rebase)
   - [拣选提交（cherry-pick）](#3-拣选提交cherry-pick)
   - [合并分支（merge）](#4-合并分支merge)
   - [git restore](#5-git-restore)
   - [其他命令](#6-其他命令我不太熟)
   - [什么是 HEAD？](#7-什么是-head)

---

# git-crash-course

## 1. 忽略文件与刷新缓存

### 添加忽略规则
在项目根目录创建一个 `.gitignore` 文件，然后添加你不想让 Git 跟踪的文件或目录。例如：
```
docs/test.md
```
 
假设你有一个文件 `docs/test.md`，里面记录了你的私人笔记，你不希望它被 Git 管理。打开 `.gitignore` 文件，写下 `docs/test.md`，保存后，Git 就会忽略这个文件。

### 强制刷新 Git 缓存（让 .gitignore 立即生效）
有时候，你已经跟踪了一些文件，后来才加到 `.gitignore` 里，但 Git 还是会管它们。这时需要"刷新缓存"。

1. **移除已被 Git 跟踪但现在要忽略的文件：**
   ```bash
   git rm --cached docs/test.md
   ``` 
   你之前不小心让 Git 跟踪了 `docs/test.md`，现在加了 `.gitignore`，但它还是在 Git 的管理下。运行 `git rm --cached docs/test.md`，Git 就会停止跟踪这个文件，但你的本地文件不会被删除。就像是跟 Git 说："别管它了，但文件我还要用。"

2. **如果有多项需要刷新，可以批量操作：**
   ```bash
   git rm -r --cached .
   git add .
   ```
   你的项目里有很多文件已经被 Git 跟踪，但 `.gitignore` 更新后需要全部重新整理。运行 `git rm -r --cached .` 会清空 Git 的跟踪列表（文件还在本地），然后 `git add .` 会根据新的 `.gitignore` 规则重新添加文件。就像是给 Git 一个"大扫除"，让它按新规则重新认识项目。

3. **提交更改：**
   ```bash
   git commit -m "refresh git cache and update .gitignore"
   ```
   刷新完缓存后，运行这个命令，把更改记录到 Git 仓库。提交说明可以写得清楚点，比如"刷新缓存并更新忽略规则"，以后看历史就知道这次提交干了啥。

4. **推送到远程仓库：**
   ```bash
   git push
   ```

---

## 2. 常用 Git 命令

### 初始化仓库
```bash
git init
```


### 克隆远程仓库
```bash
git clone <仓库地址>
```

### 查看当前状态
```bash
git status
```
在项目里改了 `index.html` 和 `style.css`，但不确定 Git 现在知道哪些变化。运行 `git status`，Git 会告诉你：`index.html` 已修改但未暂存，`style.css` 已暂存但未提交。就像是 Git 给你一份"待办清单"
这个命令会展示两部分内容： 1. 已经修改的未存入暂存区的 2. 刚创建的还没有被git跟踪的文件(Untracked files)

```bash
(base) gx-site % git status    
位于分支 main
您的分支与上游分支 'origin/main' 一致。

尚未暂存以备提交的变更：
  （使用 "git add <文件>..." 更新要提交的内容）
  （使用 "git restore <文件>..." 丢弃工作区的改动）
  （提交或丢弃子模组中未跟踪或修改的内容）
        修改：     hugo_stats.json
        修改：     public/404.html
        修改：     themes/techdoc (修改的内容)
未跟踪的文件:
  （使用 "git add <文件>..." 以包含要提交的内容）
        test.md
修改尚未加入提交（使用 "git add" 和/或 "git commit -a"）
```

### 添加文件到暂存区
```bash
git add <文件或目录>
```

你改好了 `README.md`，想让 Git 准备提交它。运行 `git add README.md`，这个文件就被放进暂存区。暂存区就像一个购物车，你先把东西放进去，等会儿一起结账（提交）。

### 提交更改
```bash
git commit -m "提交说明"
```

你已经用 `git add` 把 `README.md` 加到暂存区，现在想正式提交。运行 `git commit -m "更新了 README 文件内容"`，Git 就把这些更改存到仓库里，还附上你的说明，方便以后查看。

### 查看提交历史
```bash
git log
```

你想知道项目里都提交过什么。运行 `git log`，Git 会列出所有提交记录。

### 推送到远程仓库
```bash
git push
```

### 拉取远程更新
```bash
git pull
```
---

## 3. 其他实用技巧

### 查看分支
```bash
git branch
```

你在项目里干活，想知道有哪些分支。运行 `git branch`，Git 会列出所有本地分支，比如 `main` 和 `feature/login`

### 创建新分支
```bash
git checkout -b <新分支名>
```

开发一个新功能，比如登录页面。运行 `git checkout -b feature/login`，Git 就会创建一个叫 `feature/login` 的新分支，并自动切换过去。你可以在这个分支上随便折腾，不影响主分支。

### 切换分支
```bash
git checkout <分支名>
```

你在 `feature/login` 分支干活，领导说先检查下主分支。运行 `git checkout main`，Git 就切换到 `main` 分支，项目内容也变成主分支的模样。

### 合并分支
```bash
git merge <分支名>
```

在 `feature/login` 分支做好了登录功能，想把它加到主分支。切换到 `main` 分支（`git checkout main`），然后运行 `git merge feature/login`，Git 就会把 `feature/login` 的更改合并到 `main` 分支。

---

## 4. 进阶操作命令

### 1. 撤销与重置（reset）

- **回退到某个提交（保留修改到暂存区）：**
  ```bash
  git reset --soft <commit>
  ```
  回退到了上一次提交，此时，回退的内容是在stage中， 相当于**执行了** git add 命令， 使用 git log 命令可以看到，已经没有了最新一次的提交， 使用 git diff --cached 可以看到具体的变化，使用 git status 也能看到stage中的内容未被commit。

- **回退到某个提交（保留修改到工作区）：**
  ```bash
  git reset --mixed <commit>
  ```
  
  回退到了文件修改后的状态， 此时**未执行** git add 命令(只是文件已经修改了)， 使用 git status 可以看到有未暂存(stage)的内容， 使用git diff 可以看到具体的区别
  
  ```bash
    (base) xg@xdeMacBook-Pro react-ts % git status
        位于分支 master
        尚未暂存以备提交的变更：
        （使用 "git add <文件>..." 更新要提交的内容）
        （使用 "git restore <文件>..." 丢弃工作区的改动）
                修改：     src/App.tsx

        修改尚未加入提交（使用 "git add" 和/或 "git commit -a"）

    (base) xg@xdeMacBook-Pro react-ts % git add .              
    (base) xg@xdeMacBook-Pro react-ts % git status            
        位于分支 master
        要提交的变更：
        （使用 "git restore --staged <文件>..." 以取消暂存）
                修改：     src/App.tsx
  ```
- 上面两个命令 执行 git diff 区别

  ```bash
    (base) xg@xdeMacBook-Pro react-ts % git reset --mix HEAD~1 
        重置后取消暂存的变更：
        M       src/App.tsx

    (base) xg@xdeMacBook-Pro react-ts % git diff
        diff --git a/src/App.tsx b/src/App.tsx
        index 6e1b5a6..82a66cb 100644
        --- a/src/App.tsx
        +++ b/src/App.tsx
        @@ -26,3 +26,4 @@ function App() {
        export default App;
        // add commit 
        
        +//add third commit

    (base) xg@xdeMacBook-Pro react-ts % git add .              
    (base) xg@xdeMacBook-Pro react-ts % git commit -m'3 commit'
        [master 1ed731d] 3 commit
        1 file changed, 1 insertion(+)

    (base) xg@xdeMacBook-Pro react-ts % git reset --soft HEAD~1
    (base) xg@xdeMacBook-Pro react-ts % git diff               
    (base) xg@xdeMacBook-Pro react-ts % 
    (base) xg@xdeMacBook-Pro react-ts % git diff --cached
        diff --git a/src/App.tsx b/src/App.tsx
        index 6e1b5a6..82a66cb 100644
        --- a/src/App.tsx
        +++ b/src/App.tsx
        @@ -26,3 +26,4 @@ function App() {
        export default App;
        // add commit 
        
        +//add third commit

  ```


- **回退到某个提交（彻底丢弃修改）：**
  ```bash
  git reset --hard <commit>
  ```
  
  这个最简单，改了一堆东西，但觉得没救了，想直接放弃。运行 `git reset --hard HEAD~1`，Git 回退到上个提交，所有改动都被扔掉，像时间倒流一样。


### 2. 变基（rebase）

- **将当前分支变基到目标分支最新提交：**
  ```bash
  git rebase <目标分支>
  ```
  **例子：**  
  你在 `feature/login` 分支开发，`main` 分支有了新提交。你切换到 `feature/login`，运行 `git rebase main`，Git 会把你的提交"搬"到 `main` 的最新提交后面，保持历史整洁。

- **交互式变基（可编辑、合并、重排提交）：**
  ```bash
  git rebase -i <commit>
  ```
  rebase有两个作用， 1. 不用的branch 变基(相对简单) 2. 同一个分支使用 -i 命令 变基， 常用的功能有三个 pick reword edit squash(其中， reword和edit需要注意区别)，下面我从两个方面介绍rebase用法 
  你最近提交了 3 次，想把它们合并成 1 次。运行 `git rebase -i HEAD~3`，Git 打开一个界面，你可以选择合并（squash）、重排或编辑提交，像整理日记本一样。
好的，我来详细讲解 **Git 的 `rebase` 用法** 以及 **应用场景**，用通俗的语言和具体例子帮你理解。

---


#### 基本原理
假设你有一个分支 `feature`，是从 `main` 分支的某个点分出来的。在你开发 `feature` 的过程中，`main` 分支可能有了新的提交。使用 `git rebase`，你可以把 `feature` 的更改应用到 `main` 的最新提交上。
#### 基础用法：`git rebase <branch>`
**举个例子**：
- 初始提交历史：
  ```
  A -- B -- C (main)
       \
        D -- E (feature)
  ```
- 执行 `git checkout feature && git rebase main` 后：
  ```
  A -- B -- C (main)
             \
              D' -- E' (feature)
  ```
- 这里 `D'` 和 `E'` 是 `D` 和 `E` 的"新版本"，被重新应用到 `C` 之上。

---
你在 `feature` 分支开发新功能，同事在 `main` 分支修复了一个 bug。运行 `git rebase main` 后，你的 `feature` 分支会更新到最新的 `main`，包含 bug 修复。

---

#### 交互式变基：`git rebase -i <提交>`
- **作用**：允许你在变基过程中编辑、合并、重排或删除提交。
- **命令**：
  ```bash
  git rebase -i HEAD~3  # 对最近 3 个提交进行交互式变基
  ```
- **操作**：运行后会打开一个编辑器，你可以选择：
  - `pick`：保留提交。
  - `squash`：合并到前一个提交。
  - `edit`：修改提交内容或信息。

**示例**：
你在 `feature` 分支提交了 3 次小更改，想合并成一个提交。运行 `git rebase -i HEAD~3`，将后两个提交改为 `squash`，保存后，3 个提交会合并为 1 个。

---

#### `git rebase` 的应用场景

##### 1. 保持提交历史整洁
- **场景**：你在 `feature` 分支开发时，`main` 分支有新提交。使用 `rebase` 可以让 `feature` 的历史线性，避免多余的合并提交。
- **好处**：提交历史更清晰，易于查看。

**示例**：
- 不用 `rebase`，用 `merge`：
  ```
  A -- B -- C -- M (main)
       \       /
        D -- E (feature)
  ```
- 用 `rebase`：
  ```
  A -- B -- C (main)
             \
              D' -- E' (feature)
  ```

---

##### 2. 整理提交历史
- **场景**：开发过程中提交了很多临时或"WIP"提交，想在合并到 `main` 前清理。
- **好处**：通过交互式变基，可以合并或删除不必要的提交。

**示例**：
初始历史：
```
A -- B -- C -- D (feature)
```
运行 `git rebase -i HEAD~3`，合并 `C` 和 `D` 到 `B`，结果：
```
# 编辑器中这样设置
pick D
squash C
squash B

A -- B' (feature)
```

---

##### 3. 解决冲突
- **场景**：`feature` 分支的更改与 `main` 的新提交有冲突。`rebase` 让你逐步解决每个冲突。
- **好处**：比一次性合并更可控。

**解决步骤**：
1. 运行 `git rebase main`，遇到冲突。
2. 手动解决冲突，运行 `git add <文件>`。
3. 继续变基：`git rebase --continue`。
4. 如果放弃：`git rebase --abort`。

---


##### 1. 不要在公共分支上使用
- **原因**：`rebase` 重写历史，如果多人协作的公共分支被变基，会导致其他人历史不一致。
- **建议**：只在本地私有分支使用。

##### 2. 处理冲突
- **方法**：解决冲突后用 `git rebase --continue`，放弃用 `git rebase --abort`。

##### 3. 与 `git merge` 的区别
- **`rebase`**：历史线性，但重写提交。
- **`merge`**：保留历史，生成合并提交。

---

##### 示例1：基本 `rebase`
- **初始状态**：
  ```
  A -- B -- C (main)
       \
        D -- E (feature)
  ```
- **命令**：
  ```bash
  git checkout feature
  git rebase main
  ```
- **结果**：
  ```
  A -- B -- C (main)
             \
              D' -- E' (feature)
  ```

##### 示例2：交互式变基合并提交
- **初始状态**：
  ```
  A -- B -- C -- D (feature)
  ```
- **命令**：
  ```bash
  git rebase -i HEAD~3
  ```
- **编辑器中**：
  ```
  pick B
  squash C
  squash D
  ```
- **结果**：
  ```
  A -- B'(这是一个新的commit id都不一样) (feature)  # B' 包含 B、C、D 的更改
  ```

##### 示例3:  git rebase -i HEAD～4
- **我现在的需求是把最近一次提交合并到它的上一次提交， 并且要修改倒数第三次的commit信息， 我会分别使用 squash(合并两次提交) reword or edit(修改commit或者文件) 以及 pick**

```bash
    # 当我使用git rebase -i HEAD～4 命令的时候，会进入下面的页面
    pick 6bef283 testrebase
    pick ce44812 update test.ts
    pick 7f8c7b9 second commit   4 commit
    pick 25a3b6b hahahah woxuehuile new fif commit testrebase 1 new six commit

    # 变基 26a2f8c..25a3b6b 到 26a2f8c（4 个提交）
    #
    # 命令:
    # p, pick <提交> = 使用提交
    # r, reword <提交> = 使用提交，但编辑提交说明
    # e, edit <提交> = 使用提交，但停止以便修补提交
    # s, squash <提交> = 使用提交，但挤压到前一个提交



    # 根据上面的需求我要进行修改 
    pick 6bef283 testrebase (HEAD 4)
    reword ce44812 update test.ts (HEAD 2)
    pick 7f8c7b9 second commit   4 commit (HEAD 1)
    squash 25a3b6b hahahah woxuehuile new fif commit testrebase 1 new six commit (这是最新的一次提交,向上依次为父HEAD)

    # 变基 26a2f8c..25a3b6b 到 26a2f8c（4 个提交）
    #
    # 命令:
    # p, pick <提交> = 使用提交
    # r, reword <提交> = 使用提交，但编辑提交说明
    # e, edit <提交> = 使用提交，但停止以便修补提交
    # s, squash <提交> = 使用提交，但挤压到前一个提交

    当我使用 wq保存上面的内容后， 会出现新的页面， 依次为编辑 ce44812 的 commit， 以及合并 7f8c7b9 和 25a3b6b 需要重新写的commit信息
    另外说一下edit和reword区别： 
    1. reword：只修改提交说明
        作用：保留提交的内容（文件更改）不变，仅允许你修改该提交的提交说明（commit message）。
        过程：
        在交互式变基的编辑器中，将 pick 改为 reword。
        保存退出后，Git 会打开一个新的编辑器，让你修改提交说明。
        修改完成后，提交的内容保持不变，只有提交说明更新。

    2. edit：修改提交内容或提交说明
        作用：暂停变基过程，允许你修改提交的文件内容和/或提交说明。
        过程：
        在交互式变基的编辑器中，将 pick 改为 edit。
        保存退出后，Git 会暂停在该提交，让你可以：
        修改文件内容（比如修复代码或添加新文件）。
        用 git add 暂存更改。
        用 git commit --amend 更新提交（可以同时修改提交说明）。
        完成后，运行 git rebase --continue 继续变基。

```

---


### 3. 拣选提交（cherry-pick）


`git cherry-pick` 的作用是**从其他分支或提交历史中选择一个或多个特定的提交**，并将其应用到当前分支。它会复制指定提交的更改（比如文件修改），然后在当前分支创建一个新的提交。

---

#### 1 基本用法：`git cherry-pick <提交哈希>`
- **作用**：将指定提交的更改复制到当前分支，并创建一个新提交。
- **命令**：
  ```bash
  git cherry-pick <提交哈希>
  ```
- **效果**：目标提交的更改会被应用到当前分支，生成一个新提交（哈希值不同，但更改内容相同）。

假设你有两个分支：`main` 和 `feature`。提交历史如下：
```
A -- B -- C (main)
      \
       D -- E (feature)
```
- 提交 `E` 在 `feature` 分支修复了一个 bug（修改了 `bugfix.txt`），你想把这个修复应用到 `main` 分支。
- 找到 `E` 的提交哈希（通过 `git log --oneline`）：
  ```
  e123456 (feature) Fix bug in bugfix.txt
  ```
- 在 `main` 分支运行：
  ```bash
  git checkout main
  git cherry-pick e123456
  ```
- **结果**：
  ```
  A -- B -- C -- E' (main)
      \
       D -- E (feature)
  ```
  - `E'` 是一个新提交，包含 `E` 的更改（`bugfix.txt` 的修复），但哈希值不同。
  - `main` 分支现在有了 bug 修复。

---

#### 2. 摘取多个提交：`git cherry-pick <提交1> <提交2> ...`
- **作用**：按顺序将多个提交应用到当前分支。
- **命令**：
  ```bash
  git cherry-pick <提交1> <提交2>
  ```


---

#### 3. 处理冲突
- **场景**：如果 `cherry-pick` 的提交与当前分支有冲突，Git 会暂停并提示解决。
- **命令**：
  ```bash
  git cherry-pick <提交哈希>
  git add <冲突文件>  # 解决冲突后
  git cherry-pick --continue  # 继续
  git cherry-pick --abort  # 放弃
  ```


---

#### 4. 其他选项
- **`--no-commit`**：应用提交的更改，但不立即创建新提交：
  ```bash
  git cherry-pick --no-commit <提交哈希>
  ```
  - 适合想手动调整更改后再提交。
- **`--edit`**：允许编辑新提交的提交说明：
  ```bash
  git cherry-pick -e <提交哈希>
  ```
- **范围摘取**：摘取连续的提交：
  ```bash
  git cherry-pick <起始提交>..<结束提交>
  ```
  - 示例：`git cherry-pick d789012..e123456` 摘取 `d789012` 到 `e123456` 的所有提交。

---

### 4. 合并分支（merge）


`git merge` 的作用是将一个分支的更改合并到当前分支。它会把目标分支的提交历史引入当前分支，生成一个新的合并提交（merge commit）或直接快进（fast-forward）合并，具体取决于分支的状态。

---


#### 1. 基本用法：`git merge <目标分支>`
- **作用**：将指定的 `<目标分支>` 的更改合并到当前分支。
- **命令**：
  ```bash
  git checkout main
  git merge feature
  ```
- **效果**：将 `feature` 分支的更改合并到 `main` 分支。

- **快进合并（Fast-forward）**：
  - 如果当前分支（比如 `main`）没有新的提交，而目标分支（比如 `feature`）有新的提交，Git 会直接将 `main` 的 HEAD 移动到 `feature` 的最新提交。
  - 历史呈线性，无额外合并提交。
  - 示例历史：
    ```
    A -- B -- C (main)
          \
           D -- E (feature)
    ```
    合并后：
    ```
    A -- B -- C -- D -- E (main, feature)
    ```

- **合并提交（Merge commit）**：
  - 如果当前分支和目标分支都有新提交，Git 会创建一个新的合并提交，结合两者的更改。
  - 示例历史：
    ```
    A -- B -- C (main)
          \
           D -- E (feature)
    ```
    合并后：
    ```
    A -- B -- C -- M (main)
          \     /
           D -- E (feature)
    ```
    其中 `M` 是合并提交。

---

#### 2. 合并时处理冲突
- **场景**：如果两个分支修改了同一文件的同一部分，Git 无法自动合并，会产生冲突。
- **命令**：
  ```bash
  git merge <目标分支>
  git status  # 查看冲突文件
  git add <冲突文件>  # 解决冲突后标记
  git commit  # 完成合并
  ```

##### 处理冲突
1. 你在 `main` 分支修改了 `index.html`：
   ```html
   <h1>Main Title</h1>
   ```
2. 在 `feature` 分支也修改了 `index.html`：
   ```html
   <h1>Feature Title</h1>
   ```
3. 运行：
   ```bash
   git checkout main
   git merge feature
   ```
4. Git 报冲突：
   ```
   CONFLICT (content): Merge conflict in index.html
   Automatic merge failed; fix conflicts and then commit the result.
   ```
5. 打开 `index.html`，Git 会标记冲突：
   ```html
   <<<<<<< HEAD
   <h1>Main Title</h1>
   =======
   <h1>Feature Title</h1>
   >>>>>>> feature
   ```
6. 手动解决冲突，改为：
   ```html
   <h1>Combined Title</h1>
   ```
7. 标记解决并提交：
   ```bash
   git add index.html
   git commit
   ```

---

#### 3. 强制快进合并：`git merge --ff-only`
- **作用**：强制执行快进合并，如果无法快进（即当前分支有新提交），会报错。
- **命令**：
  ```bash
  git merge --ff-only feature
  ```
- **场景**：确保历史线性，适合没有分叉的情况。

---

#### 4. 禁用快进合并：`git merge --no-ff`
- **作用**：即使可以快进，也强制创建合并提交。
- **命令**：
  ```bash
  git merge --no-ff feature
  ```
- **场景**：想保留分支历史，明确记录合并点。

---

#### 5. 取消合并：`git merge --abort`
- **作用**：如果合并遇到冲突且不想继续，取消合并。
- **命令**：
  ```bash
  git merge --abort
  ```
- **场景**：合并冲突太多，或误操作想恢复原状。

---

#### 应用场景

1. **整合功能分支**：
   - 你在 `feature-login` 分支开发登录功能，完成后合并到 `main`：
     ```bash
     git checkout main
     git merge feature-login
     ```

2. **同步分支**：
   - `main` 分支有新更新，想合并到你的开发分支：
     ```bash
     git checkout feature
     git merge main
     ```

3. **保持历史清晰**：
   - 使用 `--no-ff` 保留合并记录：
     ```bash
     git merge --no-ff feature
     ```

4. **避免复杂合并**：
   - 使用 `--ff-only` 确保线性历史。

---


#### 示例1：快进合并
- **初始状态**：
  ```
  A -- B (main)
        \
         C -- D (feature)
  ```
- **命令**：
  ```bash
  git checkout main
  git merge feature
  ```
- **结果**：
  ```
  A -- B -- C -- D (main, feature)
  ```

#### 示例2：合并提交
- **初始状态**：
  ```
  A -- B -- E (main)
        \
         C -- D (feature)
  ```
- **命令**：
  ```bash
  git checkout main
  git merge feature
  ```
- **结果**：
  ```
  A -- B -- E -- M (main)
        \     /
         C -- D (feature)
  ```



#### `git merge` vs. `git rebase`
- **合并历史**：
  - `merge`：保留分支历史，可能产生合并提交。
  - `rebase`：重写历史，线性但改变提交哈希。
- **选择**：
  - 用 `merge` 保留完整历史，适合团队协作。
  - 用 `rebase` 追求简洁历史，适合本地开发。


---


### 5. git restore


---

`git restore` 是 Git 2.23（2019年8月）引入的命令，用于将**工作区（Working Directory）**和/或**暂存区（Staging Area）**中的文件恢复到特定状态，通常是最后一次提交（HEAD）或指定的提交。它是早期 `git checkout` 部分功能的替代品，设计上更直观，专注于撤销更改。

`git restore` 的主要用途包括：
1. **丢弃工作区的改动**（撤销未提交的文件更改）。
2. **取消暂存区的更改**（将已暂存的文件移出暂存区）。
3. **将文件恢复到特定提交的状态**。

---

#### 1. 丢弃工作区的改动：`git restore <文件>...`

- **目的**：将指定的文件恢复到最后一次提交（HEAD）的状态。
- **效果**：工作区中文件的未提交更改会被**永久丢弃**，文件内容恢复为最后一次提交的版本。
- **注意**：只影响工作区，不影响暂存区或提交历史。

假设你的项目有一个文件 `index.html`，最后提交的版本（HEAD）内容是：
```html
<h1>你好，世界！</h1>
```

你在工作区修改了 `index.html`，变成：
```html
<h1>你好，Git！</h1>
<p>测试更改</p>
```

这些更改还未用 `git add` 暂存或提交。现在你决定放弃这些更改，恢复到最后提交的状态。

1. 检查状态，确认更改：
   ```bash
   git status
   ```
   输出：
   ```
   On branch main
   Changes not staged for commit:
     modified:   index.html
   ```

2. 丢弃 `index.html` 的更改：
   ```bash
   git restore index.html
   ```

3. 验证文件：
   - 打开 `index.html`，内容恢复为：
     ```html
     <h1>你好，世界！</h1>
     ```
   - 再次运行 `git status`：
     ```
     On branch main
     nothing to commit, working tree clean
     ```
- `git restore index.html` 将工作区的 `index.html` 恢复到最后提交（HEAD）的状态。
- 更改（`<h1>你好，Git！</h1>` 和 `<p>测试更改</p>`）被丢弃，无法恢复，除非你有备份。

**多个文件**
可以同时恢复多个文件：
```bash
git restore file1.txt file2.txt
```
或丢弃工作区所有更改：
```bash
git restore .
```

---

#### 2. 取消暂存区的更改：`git restore --staged <文件>...`
这个用法用于将已暂存的文件（通过 `git add` 添加到暂存区的文件）移出暂存区。

- **目的**：将暂存区的更改"取消暂存"，移回到工作区。
- **效果**：暂存区恢复到最后提交的状态，但工作区的更改保留。
- **使用场景**：你不小心用 `git add` 暂存了文件，想取消暂存。

```bash
git restore --staged <文件>...
```
假设你修改了 `index.html` 并已暂存：
```html
<h1>你好，Git！</h1>
<p>测试更改</p>
```

1. 暂存更改：
   ```bash
   git add index.html
   git status
   ```
   输出：
   ```
   On branch main
   Changes to be committed:
     modified:   index.html
   ```

2. 取消暂存：
   ```bash
   git restore --staged index.html
   ```

3. 验证状态：
   ```bash
   git status
   ```
   输出：
   ```
   On branch main
   Changes not staged for commit:
     modified:   index.html
   ```

4. 检查文件：
   - 工作区的 `index.html` 仍包含：
     ```html
     <h1>你好，Git！</h1>
     <p>测试更改</p>
     ```
---

#### 3. 恢复文件到特定提交：`git restore --source=<提交> <文件>...`

这个用法可以将文件恢复到特定提交的状态，而不仅仅是最后提交。

- **目的**：将工作区的指定文件恢复到某个提交的状态。
- **效果**：工作区的文件被覆盖为指定提交的版本，未暂存的更改会被丢弃。
- **注意**：除非使用 `--staged`，暂存区不会受影响。

```bash
git restore --source=<提交> <文件>...
```
假设你的提交历史如下：
```
提交 A: index.html = "<h1>版本一</h1>"
提交 B: index.html = "<h1>版本二</h1>"
提交 C: index.html = "<h1>版本三</h1>" (HEAD)
```

工作区的 `index.html` 有未提交更改：
```html
<h1>版本三</h1>
<p>新内容</p>
```

你想将 `index.html` 恢复到提交 B 的状态。

1. 查看提交历史，找到提交 B 的哈希：
   ```bash
   git log --oneline
   ```
   输出：
   ```
   c123456 (HEAD -> main) 版本三
   b789012 版本二
   a456789 版本一
   ```

2. 恢复 `index.html` 到提交 B：
   ```bash
   git restore --source=b789012 index.html
   ```

3. 检查文件：
   - `index.html` 现在是：
     ```html
     <h1>版本二</h1>
     ```

- `git restore --source=b789012 index.html` 将工作区的 `index.html` 恢复到提交 `b789012` 的状态。
- 工作区中的未提交更改（即 <p>新内容</p>）以及提交 C（c123456）中对 index.html 的更改（即 <h1>版本三</h1>）在工作区中会被覆盖并丢弃，而且这些更改无法直接恢复

---

##### 关键注意事项
- **不可逆操作**：  
  使用 `git restore <文件>` 或 `git restore --source=<提交> <文件>` 丢弃的更改无法恢复，除非有备份或更改仍在暂存区/其他提交中。
- **与 `git reset` 的区别**：  
  - `git restore` 针对**文件**，只影响工作区或暂存区，不改变提交历史。  
  - `git reset` 移动分支指针，影响提交历史（参考你之前的提问）。
- **多个文件或目录**：  
  使用 `git restore .` 丢弃工作区所有更改，或 `git restore --staged .` 取消所有暂存。
- **安全检查**：  
  在丢弃更改前，用 `git status` 查看更改，用 `git diff` 查看具体差异：
  ```bash
  git diff
  ```

---

### 6 其他命令(我不太熟)


#### 1. `git diff`
- **作用**: 显示工作区、暂存区或提交之间的差异。
- **用法**:
  ```bash
  git diff                    # 工作区与暂存区的差异
  git diff --cached           # 暂存区与最后提交的差异
  git diff HEAD               # 工作区+暂存区与最后提交的差异
  git diff <commit1> <commit2> # 两个提交之间的差异
  ```
- **场景**: 
  - 检查修改了哪些内容，避免提交错误。
  - 比较不同版本的代码。
- **示例**:
  你修改了 `index.html`，想看看改了什么：
  ```bash
  git diff index.html
  ```
  输出：
  ```
  - <h1>Hello</h1>
  + <h1>Hello, Git!</h1>
  ```

---

#### 2. `git stash`
- **作用**: 临时保存工作区和暂存区的更改，并恢复到干净状态。
- **用法**:
  ```bash
  git stash push -m "描述"    # 保存更改到 stash
  git stash list              # 查看所有 stash
  git stash apply             # 恢复最新 stash（不删除）
  git stash pop               # 恢复最新 stash 并删除
  git stash drop <stash>      # 删除指定 stash
  ```
- **场景**:
  - 在切换分支前，临时保存未完成的工作。
  - 想清理工作区但不丢弃更改。
- **示例**:
  你在 `feature` 分支修改了 `style.css`，但需要切换到 `main` 处理紧急 bug：
  ```bash
  git stash push -m "WIP: 样式调整"
  git checkout main
  ```
  修复完 bug 后，恢复：
  ```bash
  git checkout feature
  git stash pop
  ```
 

---

#### 3. `git fetch`
- **作用**: 从远程仓库获取最新数据，但不自动合并到本地分支。
- **用法**:
  ```bash
  git fetch <远程>            # 获取远程仓库所有更新
  git fetch <远程> <分支>     # 获取特定分支
  ```
- **场景**:
  - 检查远程分支的更新但不想立即合并。
  - 与 `git pull`（获取+合并）相比更安全。
- **示例**:
  你想查看 `origin/main` 的最新状态：
  ```bash
  git fetch origin main
  git log --oneline main..origin/main
  ```
。

---

#### 4. `git revert`
- **作用**: 创建一个新提交，撤销指定提交的更改。
- **用法**:
  ```bash
  git revert <提交哈希>
  ```
- **场景**:
  - 想撤销某次提交但保留历史（尤其在公共分支）。
  - 比 `git reset` 更安全，不会重写历史。
- **示例**:
  提交 `abc123` 引入了 bug，想撤销：
  ```bash
  git revert abc123
  ```
  这会创建一个新提交，抵消 `abc123` 的更改。

---

#### 5. `git blame`
- **作用**: 显示文件的每一行由谁、在何时修改。
- **用法**:
  ```bash
  git blame <文件>
  ```
- **场景**:
  - 排查代码问题，找到修改人。
  - 了解某行代码的提交历史。
- **示例**:
  你想知道 `index.html` 每行是谁改的：
  ```bash
  git blame index.html
  ```
  输出：
  ```
  abc123 (Alice 2025-05-01) <h1>Hello</h1>
  def456 (Bob   2025-05-02) <p>Welcome</p>
  ```

---

#### 6. `git tag`
- **作用**: 为特定提交打标签，通常用于标记版本号。
- **用法**:
  ```bash
  git tag <标签名>            # 创建轻量标签
  git tag -a <标签名> -m "描述" # 创建带注释的标签
  git push origin <标签名>     # 推送标签到远程
  git tag                      # 查看所有标签
  ```
- **场景**:
  - 标记发布版本（例如 `v1.0.0`）。
  - 记录重要的里程碑提交。
- **示例**:
  发布版本 1.0：
  ```bash
  git tag -a v1.0 -m "Release v1.0"
  git push origin v1.0
  ```


---

#### 7. `git clean`
- **作用**: 删除工作区中未跟踪的文件和目录。
- **用法**:
  ```bash
  git clean -f                # 强制删除未跟踪文件
  git clean -fd               # 删除未跟踪文件和目录
  git clean -n                # 预览将删除的内容
  ```
- **场景**:
  - 清理临时文件（如 `.log` 文件或构建产物）。
  - 恢复干净的工作区。
- **示例**:
  项目中有未跟踪的 `temp.log` 文件：
  ```bash
  git clean -n  # 预览
  git clean -f  # 删除
  ```

---

#### 8. `git reflog`
- **作用**: 显示 Git 引用日志，记录 HEAD 和分支的所有操作。
- **用法**:
  ```bash
  git reflog
  ```
- **场景**:
  - 找回被误删的提交或分支。
  - 查看历史操作（比如 reset 或 rebase）。
- **示例**:
  你不小心用 `git reset --hard` 删了提交，想找回：
  ```bash
  git reflog
  ```
  输出：
  ```
  abc123 HEAD@{0}: reset: moving to HEAD~1
  def456 HEAD@{1}: commit: Add feature
  ```
  恢复到 `def456`：
  ```bash
  git reset --hard def456
  ```

---

#### 9. `git remote`
- **作用**: 管理远程仓库的配置。
- **用法**:
  ```bash
  git remote -v               # 查看远程仓库信息
  git remote add <名称> <URL> # 添加远程仓库
  git remote remove <名称>    # 删除远程仓库
  ```
- **场景**:
  - 检查或更新远程仓库地址。
  - 添加新的远程仓库（如协作仓库）。
- **示例**:
  添加一个新的远程仓库：
  ```bash
  git remote add upstream https://github.com/other/repo.git
  git remote -v
  ```

---

#### 10. `git show`
- **作用**: 显示某个提交的详细信息和更改内容。
- **用法**:
  ```bash
  git show <提交哈希>
  ```
- **场景**:
  - 查看某个提交的具体改动。
  - 检查提交的元数据（如作者、时间）。
- **示例**:
  查看提交 `abc123` 的详情：
  ```bash
  git show abc123
  ```
  输出：提交信息、作者和文件差异。

---

#### 综合示例
假设你在开发项目，当前状态：
- 修改了 `index.html` 和 `style.css`。
- 想保存进度但不提交。
- 检查远程更新并合并。

操作：
1. 保存临时更改：
   ```bash
   git stash push -m "WIP: 样式调整"
   ```
2. 获取远程更新：
   ```bash
   git fetch origin
   git merge origin/main
   ```
3. 恢复工作：
   ```bash
   git stash pop
   ```
4. 检查改动：
   ```bash
   git diff
   ```
5. 提交并打标签：
   ```bash
   git add .
   git commit -m "完成样式调整"
   git tag -a v1.1 -m "Release v1.1"
   ```

---


### 7 什么是 `HEAD`？

`HEAD` 是一个指针，**指向你当前所在分支的最新提交**。
```
A ← B ← C ← D  （HEAD 当前指向 D）
```

* `D` 就是你当前的最新提交，也就是 `HEAD` 所在的位置。
* `HEAD` 的值本质上是个引用，指向这个 D 提交。

---

`HEAD~1` 是 "HEAD 的第一个父提交"，即：

```
HEAD~1  = D 的父 = C
HEAD~2  = C 的父 = B
HEAD~3  = B 的父 = A
```

```
A ← B ← C ← D（HEAD）

HEAD     = D
HEAD~1   = C
HEAD~2   = B
HEAD~3   = A
```


