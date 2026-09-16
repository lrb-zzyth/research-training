#!/bin/bash
# 提交推送脚本 (修正为仓库实际路径)
cd "$(dirname "$0")"
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
git add .
git commit -m "update"
git push -u origin main
