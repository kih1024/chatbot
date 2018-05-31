class dataset():
    def __init__(self):
        self.vocab_list = []  # 단어 리스트
        print("초기화 되었습니다.")

def main():
    openTest = open("./dataset/sample10.txt", 'r')
    saveTest = open("./dataset/sample10-good.txt", 'w')
    # openTest = open("C:\Users\ojh86\Desktop\ChatBotdsasdadsadasasdsdasda\dataset\input2.txt", 'r', encoding='utf-8-sig')

    number = 1
    line = 0 # 번호
    line2 = 0 # 시간
    line3 = 0 # 대사
    for line in openTest:
        if line == str(number) + "\n":
            line2 = openTest.readline()
            line3 = openTest.readline()
            line3 = line3.replace("- ", "")
            # line3 = line3.replace("\n", "")
            saveTest.writelines(line3)
            line3 = openTest.readline()
            if(line3 != "\n"):
                line3 = line3.replace("- ", "")
                # line3 = line3.replace("\n", "")
                saveTest.writelines(line3)
                line3 = openTest.readline()
                if (line3 != "\n"):
                    line3 = line3.replace("- ", "")
                    # line3 = line3.replace("\n", "")
                    saveTest.writelines(line3)
                    line3 = openTest.readline()
        number+=1
    openTest.close()
    saveTest.close()
    # openTest = open("./dataset/input2.txt", 'w')


if __name__ == "__main__":
    main()