import os
import json
'''
清除文件中不存在label中的图片
'''
# 定义标注文件路径和图片目录路径
label_file_path = r'E:\train_data\det\val.txt'
image_dir = r'E:\train_data\det\val'

# 读取标注文件中所有图片文件名
with open(label_file_path, 'r', encoding='utf-8') as f:
    label_data = f.readlines()

# 从标注文件中提取所有图片文件名
label_images = set()
for line in label_data:
    parts = line.strip().split('\t')
    if len(parts) > 0:
        image_path = parts[0]
        label_images.add(os.path.basename(image_path))

# 遍历图片目录中的所有图片文件
for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    # 如果图片文件不在label.txt中，删除该图片文件
    if os.path.isfile(image_path) and image_file not in label_images:
        os.remove(image_path)
        print(f'Deleted: {image_path}')

print('Finished cleaning up the image directory.')
