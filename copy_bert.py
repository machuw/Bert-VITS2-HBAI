import os
import shutil


def copy_files_content():

    # 遍历base_path下的所有文件
    try:
        # 源目录和目标目录
        SRC_DIR = "./Japanese_raw"
        DEST_DIR = "./Japanese"

        # 遍历源目录中的所有文件
        for dirpath, dirnames, filenames in os.walk(SRC_DIR):
            for filename in filenames:
                if not filename.endswith(".wav"):
                    # 获取完整的源文件路径
                    src_file_path = os.path.join(dirpath, filename)
                    
                    # 获取目标文件路径
                    relative_path = os.path.relpath(dirpath, SRC_DIR)
                    dest_file_dir = os.path.join(DEST_DIR, relative_path)
                    dest_file_path = os.path.join(dest_file_dir, filename)
                    
                    # 创建目标目录（如果不存在）
                    os.makedirs(dest_file_dir, exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(src_file_path, dest_file_path)

        print("Files copied successfully!")
    except Exception as error:
        print("err!", file_path, error)


if __name__ == "__main__":
    copy_files_content()