

class Test():
    def __init__(self):
        self.vocab_list = []  # 단어 리스트
        print("초기화 되었습니다.")

    # def good(a):
    #     listt = [a]
    #     print(listt)

def main():
    test = Test()
    test2 = Test()

    i = 0
    test.vocab_list.append([1, 12])
    test.vocab_list.append([2, 44])
    test.vocab_list.append([6, 77])
    test.vocab_list.append(222)
    test.vocab_list.append(385)
    test.vocab_list.append(400)
    test.vocab_list.append(517)
    test.vocab_list.append(892)

    temp = test.vocab_list[1:1 + 6]

    print("출력 : ", test.vocab_list)
    print("출력2 : ", temp)
    print("length 출력 : ", len(test.vocab_list))
    print("length 출력[0] : ", len(test.vocab_list[0]))
    print("length 출력[1] : ", len(test.vocab_list[1]))

    if test.vocab_list[2][0] == 6:
        print("NICE")

    testt = []
    testt.append(test.vocab_list[0])

    abc = []
    abc.append("힐로라")
    abc.append("간디")
    abc.append("나방")
    abc.append("호랑이간")
    abcd = ""

    # for i in range(0, len(abc)):
    #     if abc[i] in "호랑이간다라":
    #         print()


if __name__ == "__main__":
    main()