import xml.etree.ElementTree as ET
import urllib.request

def movie_release_date_search(keyword):
    # KMDB- 영화목록 url
    url = "http://api.koreafilm.or.kr/openapi-data2/wisenut/search_api/search_xml.jsp?collection=kmdb_new&ServiceKey="
    key = "키 발급해야한"
    url = url + key

    # 입력받은 line을 utf-8로 변형
    utfLine = str(keyword.encode('utf-8'))[2:-1].replace('\\x', '%')

    newLine = url + "&title=" + utfLine + "&detail=Y"
    print(newLine)
    # xml 탐색 부분.
    tree = ET.ElementTree(file=urllib.request.urlopen(newLine))
    root = tree.getroot()  # root 노드

    for i in range(len(root[2])):
        titleName = str(root[2][i][3].text)
        if '<!HS>' in titleName:
            titleName = titleName.replace("<!HS>"," ")
            titleName = titleName.replace("<!HE>"," ")
        # print(titleName.strip()) #영화제목
        print("감독 " + root[2][i][8][0][0].text.strip() + "의 영화 " + titleName.strip() + "의 개봉일은 " +
              root[2][i][22].text.strip() + "입니다.")  # 감독명