import os
import json
from paddleocr import PaddleOCR
import sys
# from sklearn.metrics import accuracy_score

def print_progress_bar(iteration, total, length=50):
    """
    打印进度条

    :param iteration: 当前的迭代次数
    :param total: 总的迭代次数
    :param length: 进度条的长度（字符数）
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent}% Complete')
    sys.stdout.flush()       

#识别模型地址
rec_model_path = r'E:\OCRDetect\inference\en_PP-OCRv4_rec_latest'
#检测模型地址
det_model_path = 'path_to_your_trained_detection_model'

# 准备PaddleOCR对象
ocr = PaddleOCR(use_angle_cls=True,lang="en",use_gpu=0,rec_model_dir = rec_model_path)

# ICDAR 2015 数据集路径
icdar2015_data_path = r'E:\OCRDetect\train_data\icdar2015\rec'

# 准备ICDAR 2015的测试图像和标注文件
image_folder = os.path.join(icdar2015_data_path, 'train')
label_file = r'E:\OCRDetect\train_data\icdar2015\rec\rec_gt_train.txt'

with open(label_file, 'r', encoding='utf-8-sig') as f:
    annotations = [line.strip().split('\t') for line in f]


# 准备存储结果的列表
total = len(annotations)
ac = 0
count = 0
print_progress_bar(count,total)
with open('latest_result_onTrainSet', 'w', encoding='utf-8') as out_f:
# 遍历所有标注，进行OCR识别并计算准确性
    for annotation in annotations:
        image_file, true_text = annotation
        image_path = os.path.join(image_folder, image_file)
    
    # 读取图像并进行OCR识别
        result = ocr.ocr(image_path, cls=True)
        
    # 获取OCR识别结果文本
        if result and result[0]:
            ocr_text = result[0][0][1][0]
        else:
            ocr_text = ""
    
        if ocr_text ==true_text:
           ac+=1
        out_f.write(f'{image_file}:\t True:{true_text}\t OCR:{ocr_text}\t Result:{ocr_text ==true_text} \n')   
        count+=1
        print_progress_bar(count,total)
    out_f.write('AC:{:.2f}%'.format(ac/total *100))
print()
print('AC:{:.2f}%'.format(ac/total *100))

  
    
# 计算准确性
# accuracy = accuracy_score(true_texts, predicted_texts)
# print(f'OCR Accuracy on ICDAR 2015 dataset: {accuracy:.2%}')
