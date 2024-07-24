import paddle
import argparse
"""
Description:从蒸馏模型中提取教师或学生模型
Author:白彬楠
Date:2024/7/22
"""
def extract(args):
    inputmodel = args.inputmodel
    outputmodel = args.outputmodel
    # 加载预训练模型
    all_params = paddle.load(inputmodel)
    # 查看权重参数的keys
    print(all_params.keys())
    # 学生模型的权重提取
    s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
    # 查看学生模型权重参数的keys
    print(s_params.keys())
    # 保存
    paddle.save(s_params, outputmodel)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputmodel',type=str,default=r'pretrain_models\ch_PP-OCRv3_rec_train\best_accuracy.pdparams',help='待提取的模型')
    parser.add_argument('--outputmodel',type=str,default=r'pretrain_models\ch_PP-OCRv3_rec_train\student.pdparams',help='输出模型')
    
    args = parser.parse_args()
    extract(args)
    