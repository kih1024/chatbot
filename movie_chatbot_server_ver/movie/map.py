import os
import sys
import urllib.request
import urllib.parse
import geocoder
import xml.etree.ElementTree as ET

g = geocoder.ip('me')
# y, x = g.latlng # 현재 자신 컴퓨터 ip의 좌표

KakaoAK = "카카오 key 발급"
dup = "duplicate"
class Map:
    def sameDong(self, s):
        listSame = []
        encQuery1 = urllib.parse.quote(s)  # 검색 키워드
        url = "https://dapi.kakao.com/v2/local/search/address.xml?query=" + encQuery1
        request = urllib.request.Request(url)
        request.add_header("Authorization", KakaoAK)

        tree2 = ET.ElementTree(file=urllib.request.urlopen(request))
        root = tree2.getroot()
        if len(root) > 2:
            for i in range(1, len(root)):
                temp = str(root[i][1].text)
                # temp = str(i)
                listSame.append(str(temp))
            return listSame

    def run(self, s):
        # x = 127.059664
        # y = 37.619658  # 광운대학교의 좌표

        # x = 127.02639262588131
        # y = 37.50165916066829  # cgv 강남의 좌표
        # print("[" + str(x) + ", " + str(y) + "]\n")

        encQuery1 = urllib.parse.quote(s)  # 검색 키워드
        url = "https://dapi.kakao.com/v2/local/search/address.xml?query=" + encQuery1
        request = urllib.request.Request(url)
        request.add_header("Authorization", KakaoAK)

        tree1 = ET.ElementTree(file=urllib.request.urlopen(request))
        root = tree1.getroot()


        # if len(root) > 1:
        #     for i in range(1, len(root)):
        #         listSame.append(str(root[i][1].txt))
        #     return listSame

        if len(root) > 2:
            return dup

        encY = urllib.parse.quote(str(root[1][3].text))  # x좌표
        encX = urllib.parse.quote(str(root[1][4].text))  # y좌표

        encQuery = urllib.parse.quote("영화관")  # 검색 키워드
        encCategory = urllib.parse.quote("CT1")  # 문화시설만 찾기
        # encX = urllib.parse.quote(str(x))  # x좌표
        # encY = urllib.parse.quote(str(y))  # y좌표
        encRadius = urllib.parse.quote("3000")  # ip기반 위치 추적한 곳에서 반경 3000미터
        encSort = urllib.parse.quote("distance")
        url = "https://dapi.kakao.com/v2/local/search/keyword.xml?query=" + encQuery + "&category_group_code=" + encCategory + "&x=" + encX + "&y=" + encY + "&radius=" + encRadius + "&sort=" + encSort
        request = urllib.request.Request(url)
        request.add_header("Authorization", KakaoAK)

        tree = ET.ElementTree(file=urllib.request.urlopen(request))
        root = tree.getroot()
        r = '출력 결과'
        # print(root[2].text)
        for i in range(1, len(root)):
            if "CGV" in root[i][9].text or "롯데시네마" in root[i][9].text or "메가박스" in root[i][9].text:
                ad = str(root[i][0].text)
                name = str(root[i][2].text)
                distance = str(root[i][3].text)
                res = "주소 : " + ad + "\n" + "이름 : " + name + " --" + distance + "m"
                r = r + '\n\n' + res
                # list2.append(res)
                # print(root[i][0].text)  # 카카오맵 주소
                # # print(root[i][1].text) #카테고리 그룹 이름
                # print(root[i][2].text)  # 상호명
                # print(root[i][3].text + "m")  # 중심점에서의 거리
                # print("\n")

        return r



