#!/bin/bash

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate bert-vits2

# 使用pgrep查找与bert-vits2相关的进程
PIDS=$(pgrep -f "bert-vits2/bin/python")

# 如果找到了相关进程，则关闭它们
if [ -n "$PIDS" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Found bert-vits2 related processes: $PIDS"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Killing them..."
    kill -9 $PIDS
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Processes killed."
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - No bert-vits2 related processes found."
fi

#echo "$(date '+%Y-%m-%d %H:%M:%S') - Wait 10s..."
#sleep 10

#echo "$(date '+%Y-%m-%d %H:%M:%S') - Restarting train..."
#python train_ms_i.py -c configs/config.json --cont True >> ./t.log 2>&1

