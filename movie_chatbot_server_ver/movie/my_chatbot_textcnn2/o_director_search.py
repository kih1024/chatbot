import xml.etree.ElementTree as ET
import urllib.request

# KMDB- 영화목록 url
url = "http://api.koreafilm.or.kr/openapi-data2/wisenut/search_api/search_xml.jsp?collection=kmdb_new&ServiceKey="
key = "F191DA714E3E2D5EA9EE47ECF7D1EB0BB922AF96D2A4D5FB5166874A957B2B"
url = url + key

utfLine = "우디앨런"
utfLine = str(utfLine.encode('utf-8'))[2:-1].replace('\\x', '%')

# 사용자 입력 + 영화목록 url
# 제작년도 기준으로 정렬
newLine = url+"&director="+utfLine+"&startCount=0&listCount=100&sort=prodYear,1"
# &director=%EC%9A%B0%EB%94%94%EC%95%A8%EB%9F%B0&startCount=0&listCount=100&sort=prodYear,1
print(newLine)
# xml 탐색 부분.
tree = ET.ElementTree(file=urllib.request.urlopen(newLine))
root = tree.getroot()  # root 노드

for i in range(0, int(root[2].attrib.get('TotalCount'))):  # 모든 제목 출력
    print(str(i+1)+".", root[2][i][3].text)
