''' 대전 csv 불러오고 전처리 '''


import pandas as pd
import numpy as np
from konlpy.tag import Okt, Kkma

def untactTrip(fileName):
    fileName = pd.read_csv(f"C:/Users/COM/여기어떼 Dropbox/데이터/관광지 정보/{fileName}_processed.csv", encoding="CP949")

    address = fileName['address']
    address = list(address)
    gu = []

    for x in range(len(address)):
        gu.append(address[x].split(" ")[1])

    fileName['gu'] = gu

    img = fileName['img_path']
    tag = fileName['tag']

    ''' 태그에서 단어만 추출하기 '''
    nlpy = Okt()
    tags = []
    for i in range(len(address)):
        nouns = nlpy.nouns(tag[i])
        tags.append(nouns)

    for x in range(len(tags)):
        tags[x] = list(set(tags[x]))

    fileName['tags'] = tags

    return fileName

data1 = untactTrip('all')
len(data1)
files = pd.read_csv("../../데이터/관광지 정보/all_processed.csv", encoding="CP949")
files.head()
address = files['address']
address = list(address)
gu = []

for x in range(len(address)):
    gu.append(address[x].split(" ")[1])

files['gu'] = gu
files.head()
img = files['img_path']
tag = files['tag']

nlpy = Okt()
tags = []

for i in tag:
    nouns = nlpy.nouns(i)
    tags.append(nouns)

for x in range(len(tags)):
    tags[x] = list(set(tags[x]))


for x in range(len(tags)):
    tags[x] = list(set(tags[x]))

fileName['tags'] = tags

return fileName











''' 구글 클라우드 이용 이미지 라벨 추출 '''


import io
import os
import pandas as pd
from google.cloud import vision
from tqdm import tqdm

os.getcwd()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./코드/plenary-beanbag-286002-3b62dd5f0f28.json"
def google_labeling(path):
    path_dir = f"C:/Users/COM/Desktop/{path}"
    file_list = os.listdir(path_dir)
    len(file_list)
    score_df = pd.DataFrame()
    label_img = []

    for i in tqdm(range(len(file_list))):
        file_name = os.path.join(path_dir, file_list[i])
        with io.open(file_name, "rb") as image_file:
            content = image_file.read()

        client = vision.ImageAnnotatorClient()
        image = vision.types.Image(content = content)
        response = client.label_detection(image = image)
        labels = response.label_annotations
        print(len(labels))
        print(labels)
        label_img.append(labels)

        # score = 0
        # for label in labels:
        #     if label.description in ['Sea', 'Water']:
        #         score += 20
        #     elif label.description in ['Vacation', 'Sunset', 'Temple', 'Mountain', 'Valley', 'Hill', 'Ocean', 'Coast', 'Land', 'Farm', 'Island', 'Forest', 'Lake']:
        #         score += 5
        #
        # temp = pd.DataFrame({'Image':file_list[i], 'Score':score}, index = [i])
        # score_df = score_df.append(temp)

    return label_img

googleDf = google_labeling('ex_img')

googleDf

titleList = list(data1['title'])
addressList = list(data1['address'])
tagList = list(data1['tags'])
len(tagList)
# for label in tagList:
SCO = []

for x in tagList:
    sco = 0
    for y in x:
        if y in ['힐링','가족', '연인', '휴식', '아이', '경치', '연인', '데이트', '남녀노소', '산책', '사진', '문화', '공원', '역사', '부모님']:
            sco += 10
    SCO.append(sco)
temp2 = pd.DataFrame({'Title':titleList, 'Score':SCO})



''' 여행지 태그와 구글 이미지 라벨을 이용한 점수 분류 '''

temp2['Scores'] = googleDf['Score']
temp2
temp2['Sums'] = temp2.sum(axis=1)
temp2.sort_values(by = ["Sums"], axis=0, ascending=False).head(10)

import io
import os
import pandas as pd
from google.cloud import vision
from tqdm import tqdm
C:\Users\COM\여기어떼 Dropbox\코드\plenary-beanbag-286002-3b62dd5f0f28.json
os.getcwd()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./코드/plenary-beanbag-286002-3b62dd5f0f28.json"
def get_label(path, place_list = [], label_list = [], len_list = []):
    path_dir = f"C:/Users/COM/Desktop/{path}"
    file_list = os.listdir(path_dir)
    for i in tqdm(range(len(file_list))):
        file_name = os.path.join(path_dir, file_list[i])
        with io.open(file_name, "rb") as image_file:
            content = image_file.read()

        client = vision.ImageAnnotatorClient()
        image = vision.types.Image(content = content)
        response = client.label_detection(image = image)
        labels = response.label_annotations

        print(file_list[i])
        place_list.append(file_list[i].replace(".JPG","").replace(".jpg","").replace(".jpeg","").replace(".gif","").replace(".PNG","").replace(".png",""))

        temp = {}

        for label in labels:
            temp[label.description] = round(label.score, 2)

        len_list.append(len(temp))

        label_list.append(temp)

    return place_list, label_list

a = []
b = []
c = []
get_label("gyeonggi_img", a, b)
a
b
total_df1 = pd.DataFrame({'place':a, 'label':b})
total_df1

label_table_raw1 = pd.DataFrame.from_dict(b)
label_table_raw1

total_len1 = len(label_table_raw1.columns)*len(a)

label_table1 = pd.DataFrame(columns = ['place', 'label', 'score'], index = list(range(0, total_len1, 1)))
label_table1

k = 0
for i in tqdm(range(len(a))):
    for j in range(len(label_table_raw1.columns)):
        label_table1.iloc[k+j, 0] = a[i]
        label_table1.iloc[k+j, 1] = label_table_raw1.columns[j]
        label_table1.iloc[k+j, 2] = label_table_raw1.iloc[i, j]
    k = j * (i+1)
table1 = label_table1.dropna(axis = 0)

table1


a = []
b = []
c = []
get_label("gyeonggi_img", a, b)
a
b
total_df1 = pd.DataFrame({'place':a, 'label':b})
total_df1

label_table_raw1 = pd.DataFrame.from_dict(b)
label_table_raw1

total_len1 = len(label_table_raw1.columns)*len(a)

label_table1 = pd.DataFrame(columns = ['place', 'label', 'score'], index = list(range(0, total_len1, 1)))
label_table1

k = 0
for i in tqdm(range(len(a))):
    for j in range(len(label_table_raw1.columns)):
        label_table1.iloc[k+j, 0] = a[i]
        label_table1.iloc[k+j, 1] = label_table_raw1.columns[j]
        label_table1.iloc[k+j, 2] = label_table_raw1.iloc[i, j]
    k = j * (i+1)
table1 = label_table1.dropna(axis = 0)

table1
