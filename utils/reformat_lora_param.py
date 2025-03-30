import argparse
from ast import parse
import os
import shutil

def reformat(source_path, target_path):
    # 确保目标路径存在
    os.makedirs(target_path, exist_ok=True)

    # 遍历 source_path 下的所有文件夹
    folders = sorted(os.listdir(source_path))  # 按文件夹名称排序

    for index, folder in enumerate(folders):
        folder_path = os.path.join(source_path, folder)
        if os.path.isdir(folder_path):
            model_file = os.path.join(folder_path, "adapter_model.safetensors")
            if os.path.isfile(model_file):
                # 构建目标文件路径
                target_file = os.path.join(target_path, f"adapter_model_{index}.safetensors")
                # 复制文件并重命名
                shutil.copy(model_file, target_file)
                print(f"Copied: {model_file} -> {target_file}")
            else:
                print(f"Skipped: {folder_path} (model.safetensor not found)")

    print("All files processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("reformat lora weight")
    parser.add_argument("--source_path", required=True, help='souce path')
    parser.add_argument("--target_path", required=True, help='target path')
    args = parser.parse_args()
    source_path = args.source_path
    target_path = args.target_path
    reformat(source_path, target_path)
