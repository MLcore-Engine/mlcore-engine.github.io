+++
date = '2025-04-03T15:35:22+08:00'
draft = false
title = '使用 Hugo 和 GitHub Pages 创建个人网站'
+++

## 前提条件

本指南假设你已经有 GitHub 账户。

## 步骤1：创建 GitHub 仓库

在 GitHub 创建一个新仓库，仓库名必须是 `用户名.github.io`

例如，如果你的 GitHub 用户名是 `mlcore-engine`，则创建的仓库名应为 `mlcore-engine.github.io`

## 步骤2：安装 Hugo

使用 Homebrew 安装 Hugo：

```bash
brew install hugo
```

## 步骤3：创建 Hugo 站点

创建一个新的 Hugo 站点：

```bash
hugo new site my-site
cd my-site
```

## 步骤4：选择并安装主题

在 [Hugo Themes](https://themes.gohugo.io) 选择一个主题，然后安装：

```bash
# 创建主题目录
mkdir -p themes

# 克隆主题 (示例使用 techdoc 主题)
git clone https://github.com/thingsym/hugo-theme-techdoc themes/techdoc
```

## 步骤5：配置主题

编辑 `hugo.toml` 文件，配置主题：

```toml
theme = "techdoc"
```

## 步骤6：创建内容

创建新文章：

```bash
hugo new content/posts/my-first-post.md
```

## 步骤7：本地预览

启动 Hugo 服务器预览网站（包括草稿内容）：

```bash
hugo server -D
```

## 步骤8：构建站点

编辑完文章后，生成静态网站文件：

```bash
hugo --minify
```

## 步骤9：设置 Git 仓库

初始化 Git 仓库并连接到 GitHub：

```bash
cd ~/my-site
git init
# 注意：使用前需配置 SSH 密钥
git remote add origin git@github.com:你的用户名/你的用户名.github.io.git
```

## 步骤10：配置 GitHub Actions

创建 GitHub Actions 工作流配置文件：

```bash
mkdir -p .github/workflows
touch .github/workflows/hugo.yml
```

将以下内容添加到 `.github/workflows/hugo.yml` 文件：

```yaml
name: Deploy Hugo site to GitHub Pages

on:
  # 当推送到 main 分支时触发部署
  push:
    branches:
      - main
  # 允许手动触发工作流
  workflow_dispatch:

# 设置 GITHUB_TOKEN 权限，以允许部署到 GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write

# 只允许一个并发部署，跳过正在运行中的队列
concurrency:
  group: "pages"
  cancel-in-progress: false

# 默认的运行环境
defaults:
  run:
    shell: bash

jobs:
  # 构建 Hugo 站点的任务
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: true  # 获取 Hugo 主题
          fetch-depth: 0    # 获取所有历史记录和标签

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v2
        with:
          hugo-version: 'latest'
          extended: true    # 启用 Hugo 扩展版本

      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v4

      - name: Build with Hugo
        env:
          # 使用 GitHub Pages URL 作为基本 URL
          HUGO_BASEURL: ${{ steps.pages.outputs.base_url }}/
        run: |
          hugo --minify

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./public

  # 部署到 GitHub Pages 的任务
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4 
```

## 步骤11：提交并推送到 GitHub

提交所有更改并推送到 GitHub：

```bash
git add .
git commit -m "初始化 Hugo 网站"
git push -u origin main
```

**注意**：GitHub Pages 现在使用 `main` 作为默认分支，而不是 `master`。

