import requests
import json    # 파이썬 기본 모듈
import urllib  # 파이썬 기본 모듈
import os
import datetime as dt
import pandas as pd
from pandas import DataFrame


api_key = "" # 경욱 키

q = " 여행"    # 검색어
page = 1       # 접근할 페이지 번호(1~50)
size = 80      # 가져올 데이터 수 (1~80)

datetime = dt.datetime.now().strftime("%y%m%d_%H%M%S")
dirname = "%s_%s" % (q, datetime)
dirname

if not os.path.exists(dirname):
    os.mkdir(dirname)

# 접속 세션 만들기
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36"
session = requests.Session()
session.headers.update( {'User-agent': user_agent, 'referer': None, 'Authorization': 'KakaoAK ' + api_key} )


'''
params = {"page": page, "size": size, "query": q}
query = urllib.parse.urlencode(params)
query
url_tpl = "https://dapi.kakao.com/v2/search/image"
api_url = url_tpl + "?" + query
api_url
r = session.get(api_url)
if r.status_code != 200:
    print("[%d Error] %s" % (r.status_code, r.reason))
    quit()
r.encoding = "utf-8"
image_dict = json.loads(r.text)
image_dict
image_df = DataFrame(image_dict['documents'])
image_df
# 저장되는 이미지 파일의 수를 카운트 하기 위한 변수
count = 0

# 이미지 주소에 대해서만 반복
for image_url in image_df['image_url']:
    # 카운트 증가
    count += 1
    # 파일이 저장될 경로 생성
    path = "%s/단일저장_%04d.jpg" % (dirname, count)
    print( "[%s] >> %s" % (path, image_url) )
    # 예외처리 구문 적용
    try:
        # 이미지 주소를 다운로드를 위해 stream 모드로 가져온다.
        r = session.get(image_url, stream=True)
        # 에러 발생시 저장이 불가능하므로 건너뛰고 반복의 조건식으로 이동
        if r.status_code != 200:
            print("##########> 저장실패 (%d)" % r.status_code)
            continue
        # 추출한 데이터를 저장
        # -> 'w': 텍스트 쓰기 모드, 'wb': 바이너리(이진값) 쓰기 모드
        with open(path, 'wb') as f:
            f.write(r.raw.read())
            print("----------> 저장성공")
    except Exception as ex:
        print("~~~~~~~~~~~> 저장실패")
        print(ex) # 에러 메시지를 강제 출력 --> 에러 원인을 확인하기 위함.
        continue
'''

# 저장되는 이미지 파일의 수를 카운트 하기 위한 변수
count = 0

for i in range(0, 10):  # 0~9까지 --> 반복 범위를 조절하면 최대 다운받을 이미지 수가 제어된다.
    page = i + 1        # 1~10까지가 된다.

    # API에 전달할 파라미터 인코딩
    params = {"page": page, "size": size, "query": q}
    query = urllib.parse.urlencode(params)
    #print(query)

    # 최종 접속 주소 구성
    url_tpl = "https://dapi.kakao.com/v2/search/image"
    api_url = url_tpl + "?" + query
    #print(api_url)

    # API에 접근하여 데이터 가져오기
    r = session.get(api_url)

    if r.status_code != 200:
        print("[%d Error] %s" % (r.status_code, r.reason))
        # 예를 들어 21번째 페이지 접근에 실패했더라도
        # 22페이지부터 이어서 수행되어야 하므로 continue
        continue

    # 가져온 결과를 딕셔너리로 변환
    r.encoding = "utf-8"
    image_dict = json.loads(r.text)
    #print(image_dict)

    # DataFrame 생성
    image_df = DataFrame(image_dict['documents'])
    #print(image_df)

    # 이미지 주소에 대해서만 반복
    for image_url in image_df['image_url']:
        # 카운트 증가
        count += 1

        # 파일이 저장될 경로 생성
        path = "%s/반복저장_%04d.jpg" % (dirname, count)
        print( "[%s] >> %s" % (path, image_url) )

        # 예외처리 구문 적용
        try:
            # 이미지 주소를 다운로드를 위해 stream 모드로 가져온다.
            r = session.get(image_url, stream=True)

            # 에러 발생시 저장이 불가능하므로 건너뛰고 반복의 조건식으로 이동
            if r.status_code != 200:
                print("##########> 저장실패 (%d)" % r.status_code)
                continue

            # 추출한 데이터를 저장
            # -> 'w': 텍스트 쓰기 모드, 'wb': 바이너리(이진값) 쓰기 모드
            with open(path, 'wb') as f:
                f.write(r.raw.read())
                print("----------> 저장성공")
        except Exception as ex:
            print("~~~~~~~~~~~> 저장실패")
            print(ex) # 에러 메시지를 강제 출력 --> 에러 원인을 확인하기 위함.
            continue

print("끝")
