#!/bin/bash

# 从标准输入读取内容
while IFS= read -r line; do
    # 提取文件名并获取文件大小
    filename=$(echo $line | awk -F'|' '{print $1}')
    if [[ -f $filename ]]; then
        file_size=$(stat -c %s "$filename")
        echo "Size of $filename: $file_size bytes"
    else
        echo "$filename not found!"
    fi
done

