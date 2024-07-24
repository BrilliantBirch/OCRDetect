import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from paddleocr import PaddleOCR
import shutil

# 初始化PaddleOCR
det_model_dir=r'inference\ch_PPOCRv3_det_719'
rec_model_dir=r''
ocr = PaddleOCR(lang='ch',det_model_dir=det_model_dir,use_angle_cls=False,use_gpu=True,det_limit_side_len=736, det_limit_type='min')

# # 单张图片测试
# image_path = r'E:\train_data\det\test\A6S51_1413.jpg'
# savepath ='test.jpg'
# image = cv2.imread(image_path)
# result = ocr.ocr(image_path, cls=True)
# # 绘制检测到的文本框（红色）
# for line in result:
#     for box in line:
#         points = np.array(box[0], dtype=np.int32)
#         cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

# cv2.imwrite(savepath,image)   


# 标注文件路径
annotation_file_path = r'E:\train_data\det\test.txt'
output_dir = r'E:\OCRSceneImage\Trk500Model_2'
# matched_dir = os.path.join(output_dir, 'test_matched')
# unmatched_dir = os.path.join(output_dir, 'test_unmatched')

# # 确保输出目录存在
# os.makedirs(matched_dir, exist_ok=True)
# os.makedirs(unmatched_dir, exist_ok=True)

def calculate_iou(box1, box2):
    """
    计算两个矩形框的 IoU（交并比）
    :param box1: 第一个矩形框，格式为[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :param box2: 第二个矩形框，格式为[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :return: IoU 值
    """
    def get_rectangle_coords(box):
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    
    x1_min, y1_min, x1_max, y1_max = get_rectangle_coords(box1)
    x2_min, y2_min, x2_max, y2_max = get_rectangle_coords(box2)

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def is_points_match(detected_points, label_points, iou_threshold=0.5):
    """
    比较两个points列表，判断是否匹配
    :param detected_points: OCR 检测到的 points 列表
    :param label_points: 标注的 points 列表
    :param iou_threshold: IoU 阈值，默认值为0.5
    :return: 是否匹配
    """
    iou = calculate_iou(detected_points, label_points)
    return iou >= iou_threshold


#直接读取图片路径
for img in tqdm(os.listdir(r'E:\OCRSceneImage\Trk'),desc='文本框检测中'):
    image = cv2.imread(os.path.join(r'E:\OCRSceneImage\Trk',img))
    # cv2.imshow("1",image)   
    # cv2.waitKey(0) 
    if image is None:
        print(f'加载图片失败: {image}')
        continue
    #ocr检测
    result = ocr.ocr(image, cls=True)
    result2 = ocr.text_detector.predict(image)
    for line in result:
        if line is None:
            continue
        for box in line:
            points = np.array(box[0], dtype=np.int32)
            cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
    output_image_path=os.path.join(output_dir,img)     
    cv2.imwrite(output_image_path, image)     
    # cv2.imshow("2",image)  
    # cv2.waitKey(0)  
    print(f'图片保存：{output_image_path}')
    
    
#读取标注文件
# with open(annotation_file_path, 'r', encoding='utf-8') as f:
#     for line in tqdm(f, desc='文字识别中'):
#         parts = line.strip().split('\t')
#         if len(parts) < 2:
#             continue
#         image_path = parts[0]
#         annotations = json.loads(parts[1])
        
#         # 读取图像
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f'Failed to load image: {image_path}')
#             continue

#         # 运行OCR检测
#         result = ocr.ocr(image_path, cls=True)
#         detected_boxes = [box[0] for line in result for box in line]

#         all_matched = True

#         # 绘制检测到的文本框（红色）
#         for line in result:
#             for box in line:
#                 points = np.array(box[0], dtype=np.int32)
#                 cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

#         # # 绘制标签文本框（绿色），并检查是否一致
#         # for annotation in annotations:
#         #     points = np.array(annotation['points'], dtype=np.int32)
#         #     cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

#         #     if not any(is_points_match(points, dp,0.3) for dp in detected_boxes):
#         #         all_matched = False

#         # 构建输出路径并保存结果图像
#         output_image_filename = os.path.basename(image_path)
#         if all_matched:
#             output_image_path = os.path.join(matched_dir, output_image_filename)
#         else:
#             output_image_path = os.path.join(unmatched_dir, output_image_filename)

#         cv2.imwrite(output_image_path, image)

# # 创建 zip 文件
# # shutil.make_archive(os.path.join(output_dir, 'matched_images'), 'zip', matched_dir)
# # shutil.make_archive(os.path.join(output_dir, 'unmatched_images'), 'zip', unmatched_dir)

# print('文字检测完成.')
