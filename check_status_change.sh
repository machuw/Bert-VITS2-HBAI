#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate bert-vits2

FILE_PATH="/root/autodl-tmp/models/Bert-VITS2-HBAI/t.log"
TMP_FILE="/tmp/file_checksum.tmp"
RESTART_SCRIPT="/root/autodl-tmp/models/Bert-VITS2-HBAI/restart_hang_process.sh"
LOG_FILE="/root/autodl-tmp/models/Bert-VITS2-HBAI/logs/logfile.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') - ********* Start check running status *********"

# 计算文件的checksum
CURRENT_CHECKSUM=$(md5sum $FILE_PATH | awk '{print $1}')

# 如果临时文件存在，则读取上次的checksum
if [ -f $TMP_FILE ]; then
    LAST_CHECKSUM=$(cat $TMP_FILE)
else
    LAST_CHECKSUM=""
fi

# 打印新旧文件的md5值到日志文件
echo "$(date '+%Y-%m-%d %H:%M:%S') - Old Checksum: $LAST_CHECKSUM"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Current Checksum: $CURRENT_CHECKSUM"

# 比较当前checksum和上次的checksum
if [ "$CURRENT_CHECKSUM" == "$LAST_CHECKSUM" ]; then
    # 如果文件内容没有变化，则执行restart脚本
    echo "$(date '+%Y-%m-%d %H:%M:%S') - File content has not changed. Executing restart script."
    bash $RESTART_SCRIPT
else
    # 更新临时文件的checksum
    echo $CURRENT_CHECKSUM > $TMP_FILE
    echo "$(date '+%Y-%m-%d %H:%M:%S') - File content has changed. Not executing restart script."
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') - ********* End check running status *********"
