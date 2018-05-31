import xml.etree.ElementTree as ET
import urllib.request

def movie_name_search(keyword):
    # KMDB- 영화목록 url
    url = "http://api.koreafilm.or.kr/openapi-data2/wisenut/search_api/search_xml.jsp?collection=kmdb_new&ServiceKey="
    key = "키 발급 해야 한"
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
        print(titleName.strip()) #영화제목
        print("감독 : " + root[2][i][8][0][0].text.strip()) #감독명
        print("배우 : ")
        for j in range(len(root[2][i][9])):
            if j == 5:
                break;
            print(root[2][i][9][j][0].text.strip()) #영화배우명
        print("외 다수")
        print(root[2][i][10].text.strip()) #제작국가
        print(root[2][i][12].text.strip()) #줄거리
        print(root[2][i][13].text.strip() + "분") #상영시간
        print(root[2][i][14].text.strip()) #관람등급
        print(root[2][i][15].text.strip()) #장르
        print(root[2][i][22].text.strip()) #개봉일
        # rand = randint(0, len(root[2][i][25].text.split('|'))) #포스터 이상한것도 껴져 있길래 그냥 랜덤으로 줌
        print(root[2][i][25].text.strip().split('|')[0] + "\n") #포스터_주소