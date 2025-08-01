#!/bin/bash

# =============================
#  一键提交本地代码到 GitHub
# =============================

echo "🚀 开始提交代码到 GitHub..."

# 获取当前时间作为默认 commit 信息
DEFAULT_COMMIT_MSG="auto commit $(date '+%Y-%m-%d %H:%M:%S')"

# 提示用户输入提交信息，回车则使用默认
read -p "请输入提交信息（回车使用默认）: " COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="$DEFAULT_COMMIT_MSG"
fi

# 执行 Git 操作
echo "📝 正在添加文件..."
git add .

echo "📦 正在提交: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

if [ $? -ne 0 ]; then
    echo "⚠️  没有需要提交的更改，跳过提交。"
else
    echo "📤 正在拉取远程更新（防止冲突）..."
    git pull origin main --allow-unrelated-histories --no-edit

    if [ $? -ne 0 ]; then
        echo "❌ 拉取失败，请检查网络或远程配置。"
        exit 1
    fi

    echo "🚀 正在推送到远程仓库..."
    git push -u origin main

    if [ $? -eq 0 ]; then
        echo "✅ 推送成功！🎉"
    else
        echo "❌ 推送失败，请检查权限或网络。"
        exit 1
    fi
fi