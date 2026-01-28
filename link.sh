cd /data/student/Fengjunming/RoMa

# 找到所有符号链接并处理
for link in $(find src/ -type l); do
    target=$(readlink -f "$link")
    echo "处理: $link -> $target"
    rm "$link"
    cp -r "$target" "$link"
done

# 验证
find src/ -type l -ls  # 应该没有输出,说明没有符号链接了

# 提交
git add src/
git status  # 查看变化
git commit -m "Replace symbolic links with actual files"
git push origin main