from bs4 import BeautifulSoup
import urllib.request as req
from urllib.request import Request, urlopen
# import requests


def movie_attendance_search(keyword):
    url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query="

    str_keyword = str(keyword.encode('utf-8'))[2:-1].replace('\\x', '%')
    plus = "+"
    total_attendance = "누적관객수"
    str_total_attendance = str(total_attendance.encode('utf-8'))[2:-1].replace('\\x', '%')
    url = url + str_keyword + plus + str_total_attendance
    hdr = {'User-Agent': 'Mozilla/5.0'}
    requ = Request(url, headers=hdr)

    res = req.urlopen(requ)
    soup = BeautifulSoup(res, 'html.parser')
    people = soup.select_one("div > a > em.v").string

    print("영화 " + keyword + "의 누적관객수 정보입니다. " + people)
