import sys
import re

def isHangul(text):
    encText = text

    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', encText))
    return hanCount > 0

def openfile(link):
    openTest = open(link, 'r')
    saveTest = open("./dataset/input-good.txt", 'w')

    line = 0  # 번호
    line2 = 0  # 시간
    line3 = 0  # 대사
    
    print(isHangul("안녕하세요"))
    
    for line in openTest:
        if isHangul(line) == True:
            line = line.replace("<br>", "")
            line = line.replace("- ", "")
            line = line.replace("-", "")
            line = line.replace("<", "")
            line = line.replace(">", "")
            line = line.replace("<i>", "")
            line = line.replace("</i>", "")
            print(line)
            saveTest.writelines(line)
    openTest.close()
    saveTest.close()

if __name__ == '__main__':
    openfile("./dataset/input.txt")