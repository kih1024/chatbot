from random import *
import xml.etree.ElementTree as ET
import urllib.request

def movie_name_search(keyword):
    # print("영화 제목을 검색해주세요 : ")
    # title = input()
    # encTitle = urllib.parse.quote(title)
    # KMDB- 영화목록 url
    url = "http://api.koreafilm.or.kr/openapi-data2/wisenut/search_api/search_xml.jsp?collection=kmdb_new&ServiceKey="
    key = "키 발"
    url = url + key
    # keyword = keyword.replace(" ", "")
    keywords = keyword.split()

    # 입력받은 line을 utf-8로 변형
    utfLine = ""
    for i in keywords:
        utfLine += str(i.encode('utf-8'))[2:-1].replace('\\x', '%')
        utfLine += '%20'
    utfLine = utfLine[0:-3]

    ret = []
    retva = ''

    # 입력받은 line을 utf-8로 변형
    # utfLine = str(keyword.encode('utf-8'))[2:-1].replace('\\x', '%')
    #[2:-1].replace('\\x', '%')

    # request = urllib.request.Request(url)

    # tree = ET.ElementTree(file=urllib.request.urlopen(request))
    # root = tree.getroot()
    newLine = url + "&title=" + utfLine + "&detail=Y&sort=prodYear,1"
    # xml 탐색 부분.
    tree = ET.ElementTree(file=urllib.request.urlopen(newLine))
    root = tree.getroot()  # root 노드


    newLine = root[2].attrib.get('TotalCount')

    titleName = str(root[2][0][3].text)
    if '<!HS>' in titleName:
        titleName = titleName.replace("<!HS>", " ")
        titleName = titleName.replace("<!HE>", " ")
    # print(titleName.strip()) #영화제목
    retva = retva + titleName.strip() + '\n\n'
    # print("감독 : " + root[2][i][8][0][0].text.strip()) #감독명
    retva = retva + "감독 : " + root[2][0][8][0][0].text.strip() + '\n'
    # print("배우 : ")
    retva = retva + "배우 : "
    for j in range(len(root[2][0][9])):
        if j == 5:
            break
        # print(root[2][i][9][j][0].text.strip()) #영화배우명
        retva = retva + root[2][0][9][j][0].text.strip() + ','
    # print("외 다수")

    retva = retva + "외 다수" + '\n'
    # print(root[2][i][10].text.strip()) #제작국가
    retva = retva + '제작 국가 : ' + root[2][0][10].text.strip() + '\n\n'
    # print(root[2][i][12].text.strip()) #줄거리
    retva = retva + '줄거리 : ' + root[2][0][12].text.strip() + '\n\n'
    # print(root[2][i][13].text.strip() + "분") #상영시간
    retva = retva + '상영시간 : ' + root[2][0][13].text.strip() + "분" + '\n'
    # print(root[2][i][14].text.strip()) #관람등급
    retva = retva + '관람등급 : ' + root[2][0][14].text.strip() + '\n'
    # print(root[2][i][15].text.strip()) #장르
    retva = retva + '장르 : ' + root[2][0][15].text.strip() + '\n'
    #
    # rand = randint(0, len(root[2][num][25].text.split('|')))  # 포스터 이상한것도 껴져 있길래 그냥 랜덤으로 줌
    # pic = root[2][num][25].text.strip().split('|')[rand]
    pic = '.'
    pic = root[2][0][25].text.strip().split('|')[0]
    if pic == '':
        pic = 'http://www.musool.kr/skin/board/utf_musool_staff/img/no_image.gif'

    # for i in range(len(root[2][num][25])):
    #     pic = '..'
    #     if root[2][num][25].text.strip().split('|')[i] == None :
    #         pic = '...'
    #         pic = root[2][num][25].text.strip().split('|')[i]
    #         break

    # print(root[2][0][25].text.strip().split('|')[rand] + "\n")  # 포스터_주소
    # print(root[2][i][22].text.strip()) #개봉일
    ret.append(retva)
    ret.append(pic)
    return ret
