import urllib.request
import xml.etree.ElementTree as ET  # 트리 형식으로 찾으려고.
import re

def movie_rating_search(keyword):

    client_id = "4KmDmXTR6GT9s71Sl0z0"
    client_secret = "KygI6wAaja"
    encText = str(keyword.encode('utf-8'))[2:-1].replace('\\x', '%')

    url = "https://openapi.naver.com/v1/search/movie.xml?query=" + encText  # xml

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)

    tree = ET.ElementTree(file=urllib.request.urlopen(request))  # tree
    root = tree.getroot()  # root 노드

    for i in range(0, int(root[0][4].text)):
        # print(root[0][7+i][0], root[0][7+i][0].text)
        # print(root[0][7+i][7], root[0][7+i][7].text)
        str_root_text = re.sub('<.+?>', '', root[0][7+i][0].text, 0)
        print("영화 ", str_root_text, "의 평점 ", root[0][7+i][7].text)