from paddleocr import PaddleOCR, draw_ocr
import time
import sys
import os
from PIL import Image

def detect():
    total = len(os.listdir(img_path))
    count = 0
    # ac_count=0
    print_progress_bar(count,total)
    for file in os.listdir(img_path):
        file_path = os.path.join(img_path,file)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # begin_time = time.time()
            result = ocr.ocr(file_path,cls=True)
            # end_time = time.time()
            # print("{}识别耗时{:.2f}s".format(file,end_time-begin_time))
            result = result[0]
            image = Image.open(file_path).convert('RGB')
            boxes = [line[0] for line in result] if result is not None else []
            txts = [line[1][0] for line in result] if result is not None else []
            scores = [line[1][1] for line in result] if result is not None else []
            filename = file.split('_')[0]
            ac = any(filename in item for item in txts)
            # ac_count+=1 if ac else 0
            im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
            im_show = Image.fromarray(im_show)
            if not os.path.exists('车号识别result1'):
                os.makedirs('车号识别result1')
            if not os.path.exists('车号识别result2'):
                os.makedirs('车号识别result2')    
            if ac:
             im_show.save(r'车号识别result1\{}.jpg'.format(file))
            else:
             im_show.save(r'车号识别result2\{}.jpg'.format(file))
        count+=1
        print_progress_bar(count,total)
        
    print()
    # print('AC:{:.2f}%'.format(ac_count/total *100))

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
        
if __name__=='__main__':
    # total = 100
    # for i in range(total + 1):
    #     print_progress_bar(i, total)
    #     time.sleep(0.1)  # 模拟一些工作负载
    img_path=r'E:\OCRSceneImage\Trk'
    #识别模型地址
    rec_model_path = r'E:\OCRDetect\inference\en_PP-OCRv4_rec_latest'
    #检测模型地址
    det_model_path = 'path_to_your_trained_detection_model'

    # 准备PaddleOCR对象
    ocr = PaddleOCR(use_angle_cls=True,lang="en",use_gpu=0,rec_model_dir = rec_model_path)
    detect()


