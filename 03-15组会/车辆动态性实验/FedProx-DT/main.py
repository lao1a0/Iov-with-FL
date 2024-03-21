# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 13:01
@Author: KI
@File: main.py
@Motto: Hungry And Humble
"""
from args import args_parser
from server import FedProx
import os
def delete_files_in_directory(directory_path):
    try:
        # 获取目录中所有文件的列表
        file_list = os.listdir(directory_path)
        # 遍历每个文件并删除它
        for file_name in file_list:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"已删除文件：{file_path}")
        print("所有文件已成功删除。")
    except Exception as e:
        print(f"删除文件时出错：{e}")

def main():
    args = args_parser()
    fedProx = FedProx(args)
    fedProx.server()


if __name__ == '__main__':
    # 示例用法：删除目录“/path/to/your/directory”中的文件
    delete_files_in_directory("csv")
    delete_files_in_directory("fig")
    delete_files_in_directory("model")
    main()
