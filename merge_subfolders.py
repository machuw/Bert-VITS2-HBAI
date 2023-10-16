import os
import shutil

def merge_subfolders(src_folder1, src_folder2, dest_folder):
    # 获取两个源文件夹中的子文件夹名称
    subfolders1 = set(os.listdir(src_folder1))
    subfolders2 = set(os.listdir(src_folder2))
    
    # 找到两个文件夹中都存在的同名子文件夹
    common_subfolders = subfolders1.intersection(subfolders2)
    
    # 遍历每一个同名子文件夹
    for subfolder in common_subfolders:
        src_path1 = os.path.join(src_folder1, subfolder)
        src_path2 = os.path.join(src_folder2, subfolder)
        dest_path = os.path.join(dest_folder, subfolder)
        
        # 如果目标路径不存在，则创建它
        os.makedirs(dest_path, exist_ok=True)
        
        # 合并两个源文件夹中的文件
        for src_path in [src_path1, src_path2]:
            for file_name in os.listdir(src_path):
                full_file_name = os.path.join(src_path, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, dest_path)

# 示例
src_folder1 = 'path_to_source_folder1'
src_folder2 = 'path_to_source_folder2'
dest_folder = 'path_to_destination_folder'

merge_subfolders(src_folder1, src_folder2, dest_folder)
