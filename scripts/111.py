
import os
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('../data/ISBI2016_ISIC_Part1_Test_GroundTruth.csv')

# 获取 result 文件夹中所有图片文件名
tested_images = [f for f in os.listdir('./results') if os.path.isfile(os.path.join('./results', f))]

# 提取测试完图片文件名中的数字部分
tested_image_numbers = [name.split('_output_ens')[0].lstrip('0') for name in tested_images]

# 提取 CSV 文件中文件名的数字部分
def extract_number(file_name):
    return file_name.split('ISIC_')[-1].split('.')[0].lstrip('0')

df['number'] = df['img'].apply(extract_number)

# 从 DataFrame 中删除已测试的图片对应的行
df = df[~df['number'].isin(tested_image_numbers)]

# 删除临时添加的 number 列
df = df.drop(columns='number')

print(1)
# 将结果保存为新的 CSV 文件
csv_path = '../data/ISBI2016_ISIC_Part1_Test_GroundTruth.csv'
df.to_csv(csv_path, index=False)