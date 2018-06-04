import xml.etree.ElementTree as ET
import urllib.request
from datetime import date, timedelta
import sys


def actor_name_search():
    # 영화진흥위원회 - 영화목록 url
    url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchWeeklyBoxOfficeList.xml?key="
    key = "영화 진흥원 발급키"
    url = url + key
    retva ='주간 박스오피스 \n'
    ret = []
    # 입력받은 line을 utf-8로 변형
    # utfLine = str(keyword.encode('utf-8'))[2:-1].replace('\\x', '%')
    ret = []
    # 사용자 입력 + 영화목록 url
    yesterday = date.today() - timedelta(7)
    now = yesterday.strftime("%Y%m%d")
    newLine = url+"&targetDt=" + now + "&weekGb=0"
    # print(newLine)
    # xml 탐색 부분.
    tree = ET.ElementTree(file=urllib.request.urlopen(newLine))
    root = tree.getroot()  # root 노드

    for i in range(0, len(root[3])):
        ret.append(root[3][i][5].text)
        # if i < len(root[2]) - 1:
        #     retva = retva + str(i+1) + '. ' + root[2][i][5].text + '\n'
        # else:
        #     retva = retva + str(i+1) + '. ' + root[2][i][5].text + '.'
    # ret.append(retva)
    return ret

