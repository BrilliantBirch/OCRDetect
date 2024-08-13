"""
det训练完成后立即训练rec
"""
import subprocess

def run_script(script_name,config):
    try:
        # 运行 Python 脚本
        result = subprocess.run(['python', script_name, '-c', config], capture_output=False, text=True, check=True)
        # 输出结果
        print(f"{script_name} executed successfully.")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        # 处理失败情况
        print(f"{script_name} failed with exit code {e.returncode}.")
        print("Error output:")
        print(e.stderr)
        return False

# 执行第一个脚本
script_name=r'tools/train.py'
det_config=r'configs\det\ch_PP-OCRv3\ch_PP-OCRv3_det_student.yml'
rec_config=r'configs\rec\PP-OCRv3\ch_PP-OCRv3_rec.yml'
# run_script(script_name,det_config)
run_script(script_name,rec_config)


# # 可以根据需要进一步处理
# if not script1_success:
#     print("First script failed. Please check the output for details.")
# script_path = 'tools/train.py'
# det_config_path = r'configs\det\ch_PP-OCRv3\ch_PP-OCRv3_det_student.yml'
# rec_config_path=r'configs\rec\PP-OCRv3\ch_PP-OCRv3_rec.yml'
# # 执行脚本并传递参数
# subprocess.run(['python', script_path, '-c', det_config_path])
# subprocess.run(['python', script_path, '-c', rec_config_path])