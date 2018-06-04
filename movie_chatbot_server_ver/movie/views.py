#from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import xml.etree.ElementTree as ET
import urllib.request
from movie.models import User_state
# from movie.models import Picture
from django.core.exceptions import ObjectDoesNotExist
from django.template.loader import get_template
# from django.template import Context
# from meal.ChatBot_Final.config import FLAGS# 이거 임포트 까진 됨
# from meal.ChatBot_Final.chat import ChatBot
from movie.my_chatbot_textcnn2 import my_predict
from movie.map import Map
from movie import dailyBoxOffice
from movie import o_movie_name_search2
import o_actor_search  # 배우 검색용 - KMDB
import o_movie_name_search  # 영화 제목 검색용 - KMDB
import o_movie_release_date_search  # 영화 개봉년도 검색용 - KMDB
import o_movie_runtime_search  # 영화 상영시간 검색용 - KMDB
import o_movie_rating_search  # 영화 평점 검색 - Naver API
import o_movie_attendance_search
import reservation

# from movie.mctccm.my_predict import tokenizer
from tensorflow.contrib.learn.python.learn.preprocessing import text

select = 0

# chatbot = myPredict()
# chatbot = ChatBot(FLAGS.voc_path, FLAGS.vec_path, FLAGS.train_dir)

def searchP(request):
        pictureN = request
        ret = []
        ret1 = []
        client_id = "4KmDmXTR6GT9s71Sl0z0"
        client_secret = "KygI6wAaja"

        # encText = str(pictureN.encode('utf-8'))[2:-1].replace('\\x', '%')
        encQuery1 = urllib.parse.quote(pictureN)  # 검색 키워드
        """여기는 블로그 검색하는 부분. 참고용."""
        # url = "https://openapi.naver.com/v1/search/blog?query=" + encText # json 결과  # 원본
        # url = "https://openapi.naver.com/v1/search/blog?query=" + encText + "&display=11" # json 결과
        # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과  # 원본

        url = "https://openapi.naver.com/v1/search/movie.xml?query=" + encQuery1  # xml
        # url = "https://openapi.naver.com/v1/search/movie.json?query=" + encText  # json

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        # response = urllib.request.urlopen(request)
        # rescode = response.getcode()

        tree = ET.ElementTree(file=urllib.request.urlopen(request))  # tree
        root = tree.getroot()  # root 노드

        # print("root[0][0] :", root[0][7][2].text)  # 요놈이 image
        if len(root[0]) - 7 == 1:
                # ret1 = root[0][7][2].text
                ret1.append(root[0][7][2].text)
                return ret1
        else:
                for i in range(0, len(root[0]) - 7):
                        spl = root[0][7 + i][0].text
                        newstr = spl.replace("<b>", "")
                        newstr = newstr.replace("</b>", "")
                        newstr = newstr.replace("amp;", "")

                        ret.append(newstr)
                        # print("root[0][i][2] :", root[0][7 + i][2].text)
                return ret


        ###이미지 부분
        # responses = requests.get(root[0][7][2].text)
        # img = Image.open(BytesIO(responses.content))
        ### 이미지 부분

        # print("root[1] :", root[0][7])
        # print("root[2] :", len(root[0][7]))

        # if rescode == 200:
        #         response_body = response.read()
        #         # print(type(response_body))  # response_body는 bytes 클래스
        #
        #         # print(response_body.decode('utf-8'))  # utf-8 옵션을 줘서 한글로 decode
        #         print(response_body)
        #
        # else:
        #         print("Error Code:" + rescode)




def search(request):
# 영화진흥위원회 - 영화목록 url
        actorN = request
        url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/people/searchPeopleList.xml?key="
        key = "2b0f287ca6e1727194e49bf726da146d"
        url = url + key
        ret = []

        # 입력받은 line을 utf-8로 변형
        utfLine = str(actorN.encode('utf-8'))[2:-1].replace('\\x', '%')

        # 사용자 입력 + 영화목록 url
        newLine = url + "&peopleNm=" + utfLine
        # print(newLine)
        # xml 탐색 부분.
        tree = ET.ElementTree(file=urllib.request.urlopen(newLine))
        root = tree.getroot()  # root 노드


        for i in range(0, len(root[1])):
                if root[1][i][3].text == "배우":
                        # print(i + 1, actorN, "의 출연작품 :", root[1][i][4].text)
                        ret.append(root[1][i][4].text)
                        # print(root[1][i][3])
        return ret

def nowLocation(request):
    t = get_template('marker.html')
    html = t.render()
    return HttpResponse(html)

def nowStatic(request):
    t = get_template('pic1.html')
    html = t.render()
    return HttpResponse(html)

def keyboard(request):
        return JsonResponse({
                'type': 'buttons',
                'buttons': ['영화 챗봇', '영화관 검색', '주간 박스 오피스']
        })


@csrf_exempt
def message(request):
        global select
        message = ((request.body).decode('utf-8'))
        return_json_str = json.loads(message)
        return_str = return_json_str['content']
        return_key = return_json_str['user_key']
        try:
                fb = User_state.objects.get(user_key=return_key)  # 디비에 유저 id가 있는지 확인 여기까진 됨
        except ObjectDoesNotExist:
                fb = User_state(user_key=return_key, user_state=0)
                fb.save()

#        try:
#                pb = Picture.objects.get(pic = 'kakao1.jpg')
#        except ObjectDoesNotExist:
#                pb = User_state(pic = 'kakao1.jpg')
#                pb.save()

        if return_str == '영화관 검색':
                fb.user_state = 1
                fb.save()
                # print('사진 검색')

                return JsonResponse({
                        'message': {
                                'text': '찾을 방법을 골라주세요'
                        },
                        'keyboard': {
                                'type': 'buttons',
                                'buttons': ['현재 위치', '동 검색']
                        }
                })


        elif return_str == '메뉴':
                fb.user_state = 0
                fb.save()

                return JsonResponse({
                        'message': {
                                'text': '원하시는 기능을 고르세요.'
                        },
                        'keyboard': {
                                'type': 'buttons',
                                'buttons': ['영화 챗봇', '영화관 검색', '주간 박스 오피스']
                        }
                })
        elif return_str == '주간 박스 오피스':
                fb.user_state = 2
                fb.save()
                mName = dailyBoxOffice.actor_name_search()
                return JsonResponse({
                        'message': {
                                'text': '주간 박스 오피스는 다음과 같습니다.   \n맨위 부터 순위별로 나열\n예매 할 영화를 고르세요.'
                        },
                        'keyboard': {
                                'type': 'buttons',
                                'buttons': mName
                        }
                })

        elif return_str == '영화 챗봇':
                fb.user_state = 3
                fb.sameName = 0
                fb.save()
                return JsonResponse({
                        'message': {
                                'text': '영화 정보 제공을 위한 챗봇입니다',
                                'photo': {
                                        "url": 'https://postfiles.pstatic.net/MjAxODA1MjZfMjg2/MDAxNTI3MzMwNTcwNzcx.y7gf5_2Zc84LZXp01xk24IDmUcjCuTz131GTGwXzCqog.KlfFEFYpI1z9fRhK12rUnmIYxPUxEIEeJ4Q0tV7VqWsg.PNG.rladlsgh654/kakao1.png?type=w773',
                                        "width": 720,
                                        "height": 630
                                }
                        },
                        'keyboard': {
                                'type': 'text'
                        }
                })
        elif return_str == '동 검색':
                fb.user_state = 4
                fb.save()

                return JsonResponse({
                        'message': {
                                'text': '주소를 입력해주세요!',
                                'photo': {
                                        "url": 'https://postfiles.pstatic.net/MjAxODA1MjZfMTI3/MDAxNTI3MzMwNTcwNzcx.qY6yI3zWLm3W3BBtOk7c8x2cOHiukhbs8NRKA_I1y8Yg.iFxCj3TDBobL20xIqKqdGYnVjdX6pO-fBxDzAyRilAMg.PNG.rladlsgh654/kakao2.png?type=w773',
                                        "width": 720,
                                        "height": 630
                                }
                        },
                        'keyboard': {
                                'type': 'text'
                        }

                })
        elif return_str == '현재 위치':
                fb.user_state = 5
                fb.save()
                # t = get_template('marker.html')
                # html = t.render()

                # return JsonResponse(html)
                return JsonResponse({
                        'message': {
                                'text': '현재 위치 확인 완료' + '\n\n' +'처음 메뉴 돌아갈시 -> 메뉴' + '\n' + '동으로 검색시 -> 동 검색',
                                'message_button': {
                                    'label': '클릭!',
                                    'url': 'http://13.124.200.65/nowLocation'
                                }
                        },
                        'keyboard': {
                                'type': 'text'
                        }

                })

        elif return_str == '이전으로 가기':
                if fb.user_state == 2:
                        mName = dailyBoxOffice.actor_name_search()
                        return JsonResponse({
                                'message': {
                                        'text': '주간 박스 오피스는 다음과 같습니다.   \n맨위 부터 순위별로 나열\n예매 할 영화를 고르세요.'
                                },
                                'keyboard': {
                                        'type': 'buttons',
                                        'buttons': mName
                                }
                        })
        elif return_str == '해당 영화 예매하기':
                rets = fb.nameTem
                ret = reservation.theater_search(rets)
                fb.nameTem = 'empty'
                fb.save()
                return JsonResponse({
                        'message': {
                                'text': '해당영화 예매 하러가기!',
                                'message_button': {
                                    'label': '클릭!',
                                    'url': ret
                                }
                        },
                        'keyboard': {
                                'type': 'buttons',  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                'buttons': ['메뉴', '이전으로 가기']
                        }

                })

        elif return_str == return_str:
                if fb.user_state == 2:
                        mName = o_movie_name_search2.movie_name_search(return_str)
                        rets = return_str.replace(' ', '')
                        fb.nameTem = rets
                        fb.save()
                        return JsonResponse(
                                {
                                        'message': {
                                                'text': mName[0],
                                                'photo': {
                                                        "url": mName[1],
                                                        "width": 720,
                                                        "height": 630
                                                }
                                        },
                                        'keyboard': {
                                                'type': 'buttons',  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                'buttons': ['해당 영화 예매하기', '메뉴', '이전으로 가기']
                                        }

                                }
                        )


                elif fb.user_state == 3:
                        # mName = search(return_str)
                        # global chatbot
                        mn1 = return_str
                        mName1 = my_predict.predict_unseen_data(mn1, fb)

                        if mName1 == 'error!!':
                                fb.user_state = 3
                                fb.save()
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': '해당 결과가 없습니다. 다시 입력해주세요',
                                                        'photo': {
                                                                "url": 'https://postfiles.pstatic.net/MjAxODA1MjZfMjAw/MDAxNTI3MzMwNTcwNzcy.v_7NPyf0I68qlD2Am5UixH_d5WfDQPvUZ_ryvarZ6Qog.U9VkHiVHgvoYQ5TM_B8MUDViXk5LafOImbkJM1gjOLIg.PNG.rladlsgh654/kakao3.png?type=w773',
                                                                "width": 720,
                                                                "height": 630
                                                        }
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }

                                        }
                                )

                        if mName1[0] == '':
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': '',
                                                        'photo' : {
                                                                 "url" : 'https://postfiles.pstatic.net/MjAxODA1MjZfMTY4/MDAxNTI3MzMxNTAyODQw.bMqja_nz13oyc4otREccoIKV87y0eiSF5VpSx3IV87cg.STq8ibZ6neCh4KFY9TJcnNbZF_otd9YnMi6IOp8C98gg.PNG.rladlsgh654/helppage.png?type=w773',
                                                        "width": 720,
                                                        "height": 630
                                                        }
                           
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }

                                        }
                                )

                        if fb.user_state == 10:
                                if len(mName1) > 1 and mName1[1].find('http') == -1:
                                        fb.sameName = 1
                                        fb.save()
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': '어떤 영화인지 골라주세요.'
                                                        },
                                                        'keyboard': {
                                                                'type': 'buttons',  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                                'buttons': mName1
                                                        }

                                                }
                                        )
                                else:
                                        fb.user_state = 3
                                        fb.sameName = 0
                                        fb.save()
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': mName1[0],
                                                                'photo': {
                                                                        "url": mName1[1],
                                                                        "width": 720,
                                                                        "height": 630
                                                                }
                                                        },
                                                        'keyboard': {
                                                                'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                        }

                                                }
                                        )
                        elif fb.user_state == 11:
                                if len(mName1) > 1 and mName1[0].find(" -- key : ") != -1:
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': '동명이인이 있습니다. 골라주세요.'
                                                        },
                                                        'keyboard': {
                                                                'type': 'buttons',  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                                'buttons': mName1
                                                        }

                                                }
                                        )

                                elif len(mName1) > 1:
                                        fb.sameName = 1
                                        fb.save()
                                        # number = len(mName1) - 1
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': '해당 배우의 출연영화 리스트 입니다.'
                                                        },
                                                        'keyboard': {
                                                                'type': 'buttons',  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                                'buttons': mName1
                                                        }

                                                }
                                        )
                                else:
                                        fb.user_state = 3
                                        fb.sameName = 0
                                        fb.save()
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': mName1[0]
                                                        },
                                                        'keyboard': {
                                                                'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                        }

                                                }
                                        )
                        elif fb.user_state == 12:
                                fb.user_state = 3
                                fb.sameName = 0
                                fb.save()
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': mName1[0]
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }

                                        }
                                )
                        elif fb.user_state == 13:
                                fb.user_state = 3
                                fb.sameName = 0
                                fb.save()
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': mName1[0]
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }

                                        }
                                )
                        elif fb.user_state == 14:
                                fb.user_state = 3
                                fb.sameName = 0
                                fb.save()
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': mName1[0]
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }

                                        }
                                )
                        elif fb.user_state == 15:
                                fb.user_state = 3
                                fb.sameName = 0
                                fb.save()
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': mName1[0]
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }

                                        }
                                )
                        elif fb.user_state == 16:
                                fb.user_state = 3
                                fb.sameName = 0
                                fb.save()
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': mName1
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }

                                        }
                                )
                        elif fb.user_state == 17:
                                #fb.user_state = 3
                                #fb.save()
                                menu1 = mName1 + ' - ' + '영화'
                                menu2 = mName1 + ' - ' + '배우'
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': '해당 키워드가 어떤 것 인가요?'
                                                },
                                                'keyboard': {
                                                        'type': 'buttons',  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                        'buttons': [menu1, menu2]
                                                }

                                        }
                                )
                        elif fb.user_state == 18:
                                fb.user_state = 3
                                fb.sameName = 0
                                fb.save()
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': 'HELP',
                                                        'photo': {
                                                                "url": 'https://postfiles.pstatic.net/MjAxODA1MjZfMjIy/MDAxNTI3MzMxNTAyODU4.hX4GlxU1kYNWBb-4wPEOb1wyoMf8QyQpWIUXz0JnV54g.X2QrK6_i3jeLEM9rUsayV-vSs0Kbk08Nmx6QsKFp2bEg.PNG.rladlsgh654/HELP.png?type=w773',
                                                                "width": 720,
                                                                "height": 630
                                                        }
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }

                                        }
                                )
                        elif fb.user_state == 19:
                                fb.user_state = 3
                                fb.sameName = 0
                                fb.save()
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': '다시 검색해주세요!',
                                                        'photo': {
                                                                "url": 'https://postfiles.pstatic.net/MjAxODA1MjhfMTEx/MDAxNTI3NDk5MzQ1MzI0.0LKRc9dvU_3vFC_nNwD-u7IPwQxCIQZkRg3FdTfXmXIg.Dxem6ZKBA-_C-wUdFe-O__RjH5YkzT0-e7mei5cjm_Yg.PNG.rladlsgh654/KakaoTalk_20180528_181657365.png?type=w773',
                                                                "width": 720,
                                                                "height": 630
                                                        }
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }

                                        }
                                )
                        elif fb.user_state == 20:
                                fb.user_state = 3
                                fb.sameName = 0
                                fb.save()
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': '해당 키워드가 없어요!',
                                                        'photo': {
                                                                "url": 'https://postfiles.pstatic.net/MjAxODA1MjZfMjAw/MDAxNTI3MzMwNTcwNzcy.v_7NPyf0I68qlD2Am5UixH_d5WfDQPvUZ_ryvarZ6Qog.U9VkHiVHgvoYQ5TM_B8MUDViXk5LafOImbkJM1gjOLIg.PNG.rladlsgh654/kakao3.png?type=w773',
                                                                "width": 720,
                                                                "height": 630
                                                        }
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }

                                        }
                                )


                elif fb.user_state == 4:
                        # spl = return_str.split()
                        map = Map()
                        res = map.run(return_str)
                        # res = map.run(127.059664, 37.619658)
                        # '\n'.join(res)
                        if res == 'duplicate':
                                list3 = map.sameDong(return_str)
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': '중복되는 위치가 다음과 같습니다. 어떤걸 선택하시겠습니까?'
                                                },
                                                'keyboard': {
                                                        'type': 'buttons',  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                        'buttons': list3
                                                }
                                        }
                                )
                        else:
                                res = map.run(return_str)
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': res + '\n\n' +'처음 메뉴 돌아갈시 -> 메뉴' + '\n' + '자기위치로 검색시 -> 현재 위치'
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }
                                        }
                                )
                elif fb.user_state == 10:
                        # mName = search(return_str)
                        # global chatbot
                        mn1 = return_str.replace(' -- 감독 : ', '|').replace(' -- Seq : ', '|')
                        tmp = mn1.split("|")
                        res = tmp[0].replace(' ', '') #영화 이름
                        res2 = tmp[1]
                        res3 = tmp[2]
                        mName2 = o_movie_name_search.movie_name_search(res, res2, res3 ,fb)
                        fb.user_state = 3
                        fb.sameName = 0
                        fb.save()

                        return JsonResponse(
                                {
                                        'message': {
                                                'text': mName2[0],
                                                'photo': {
                                                        "url": mName2[1],
                                                        "width": 720,
                                                        "height": 630
                                                }
                                        },
                                        'keyboard': {
                                                'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                        }

                                }
                        )
                elif fb.user_state == 11:
                        # mName = search(return_str)
                        # global chatbot
                        if " -- key : " in return_str:
                                mName1 = o_actor_search.actor_name_search(return_str)
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': '해당 배우의 출연영화는 다음과 같습니다!' + str(len(mName1))
                                                },
                                                'keyboard': {
                                                        'type': 'buttons',  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                        'buttons': mName1
                                                }

                                        }
                                )
                        mn1 = return_str.replace(' -- 감독 : ', '|').replace(' -- Seq : ', '|')
                        tmp = mn1.split("|")
                        res = tmp[0].replace(' ', '') #영화 이름
                        res2 = tmp[1]
                        res3 = tmp[2]
                        mName1 = o_movie_name_search.movie_name_search(res, res2, res3 ,fb)
                        # mName1 = o_movie_name_search.movie_name_search(mn1, fb)
                        fb.user_state = 3
                        fb.sameName = 0
                        fb.save()

                        return JsonResponse(
                                {
                                        'message': {
                                                'text': mName1[0],
                                                'photo': {
                                                        "url": mName1[1],
                                                        "width": 720,
                                                        "height": 630
                                                }
                                        },
                                        'keyboard': {
                                                'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                        }

                                }
                        )
                elif fb.user_state == 12:
                        # mName = search(return_str)
                        # global chatbot
                        mn1 = return_str
                        mName1 = o_movie_release_date_search.movie_release_date_search(mn1)
                        fb.user_state = 3
                        fb.save()

                        return JsonResponse(
                                {
                                        'message': {
                                                'text': mName1[0]
                                        },
                                        'keyboard': {
                                                'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                        }

                                }
                        )
                elif fb.user_state == 13:
                        # mName = search(return_str)
                        # global chatbot
                        mn1 = return_str
                        mName1 = o_movie_runtime_search.movie_runtime_search(mn1)
                        fb.user_state = 3
                        fb.save()

                        return JsonResponse(
                                {
                                        'message': {
                                                'text': mName1[0]
                                        },
                                        'keyboard': {
                                                'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                        }

                                }
                        )
                elif fb.user_state == 14:
                        # mName = search(return_str)
                        # global chatbot
                        mn1 = return_str
                        mName1 = o_movie_rating_search.movie_rating_search(mn1)
                        fb.user_state = 3
                        fb.save()

                        return JsonResponse(
                                {
                                        'message': {
                                                'text': mName1[0]
                                        },
                                        'keyboard': {
                                                'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                        }

                                }
                        )
                elif fb.user_state == 15:
                        # mName = search(return_str)
                        # global chatbot
                        mn1 = return_str
                        mName1 = o_movie_attendance_search.movie_attendance_search(mn1)
                        fb.user_state = 3
                        fb.save()

                        return JsonResponse(
                                {
                                        'message': {
                                                'text': mName1[0]
                                        },
                                        'keyboard': {
                                                'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                        }

                                }
                        )
                elif fb.user_state == 17:
                        # mn1 = return_str.replace(' - ', '|').replace('#', '')
                        mn1 = return_str.replace(' - ', '|')
                        mn1 = mn1.split('|')

                        if mn1[1] == '영화':
                                mName1 = o_movie_name_search.movie_name_search(mn1[0], 'empty', 'empty' ,fb)

                                if mName1 == "error!!":
                                        fb.user_state = 3
                                        fb.save()
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': '해당 키워드의 정보가 없습니다. 다시 입력하세요',
                                                                'photo': {
                                                                        "url": 'https://postfiles.pstatic.net/MjAxODA1MjZfMjAw/MDAxNTI3MzMwNTcwNzcy.v_7NPyf0I68qlD2Am5UixH_d5WfDQPvUZ_ryvarZ6Qog.U9VkHiVHgvoYQ5TM_B8MUDViXk5LafOImbkJM1gjOLIg.PNG.rladlsgh654/kakao3.png?type=w773',
                                                                        "width": 720,
                                                                        "height": 630
                                                                }
                                                        },
                                                        'keyboard': {
                                                                'type': 'text'
                                                        }

                                                }
                                        )

                                elif len(mName1) > 1 and mName1[1].find('http') == -1:
                                        fb.user_state = 10
                                        fb.save()
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': '어떤 영화인지 골라주세요!!'
                                                        },
                                                        'keyboard': {
                                                                'type': 'buttons',  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                                'buttons': mName1
                                                        }

                                                }
                                        )
                                elif len(mName1) == 2:
                                        fb.user_state = 3
                                        fb.save()
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': mName1[0],
                                                                'photo': {
                                                                        "url": mName1[1],
                                                                        "width": 720,
                                                                        "height": 630
                                                                }
                                                        },
                                                        'keyboard': {
                                                                'type': 'text'
                                                        }

                                                }
                                        )
                                else:
                                        fb.user_state = 3
                                        fb.save()
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': '해당 키워드의 정보가 없습니다. 다시 입력하세요',

                                                        },
                                                        'keyboard': {
                                                                'type': 'text'
                                                        }

                                                }
                                        )
                        elif mn1[1] == '배우':
                                fb.user_state = 11
                                fb.save()
                                mName1 = o_actor_search.actor_name_search(mn1[0])

                                if mName1 == "error!!":
                                        fb.user_state = 3
                                        fb.save()
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': '해당 키워드의 정보가 없습니다. 다시 입력하세요',
                                                                'photo': {
                                                                        "url": 'https://postfiles.pstatic.net/MjAxODA1MjZfMjAw/MDAxNTI3MzMwNTcwNzcy.v_7NPyf0I68qlD2Am5UixH_d5WfDQPvUZ_ryvarZ6Qog.U9VkHiVHgvoYQ5TM_B8MUDViXk5LafOImbkJM1gjOLIg.PNG.rladlsgh654/kakao3.png?type=w773',
                                                                        "width": 720,
                                                                        "height": 630
                                                                }
                                                        },
                                                        'keyboard': {
                                                                'type': 'text'
                                                        }

                                                }
                                        )
                                elif len(mName1) > 0 and mName1[0].find(' -- key : ') != -1:
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': '동명이인이 있습니다. 골라주세요'
                                                        },
                                                        'keyboard': {
                                                                'type': 'buttons',  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                                'buttons': mName1
                                                        }

                                                }
                                        )
                                elif len(mName1) > 0:
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': '해당 배우의 출연영화는 다음과 같습니다.'
                                                        },
                                                        'keyboard': {
                                                                'type': 'buttons',  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                                'buttons': mName1
                                                        }

                                                }
                                        )
                                else:
                                        fb.user_state = 3
                                        fb.save()
                                        return JsonResponse(
                                                {
                                                        'message': {
                                                                'text': '해당 키워드의 정보가 없습니다. 다시 입력하세요'
                                                        },
                                                        'keyboard': {
                                                                'type': 'text'
                                                        }

                                                }
                                        )

                        else:
                                fb.user_state = 3
                                fb.save()
                                return JsonResponse(
                                        {
                                                'message': {
                                                        'text': '다시 입력하세요'
                                                },
                                                'keyboard': {
                                                        'type': 'text'  # 텍스트로 입력받기 위하여 키보드 타입을 text로 설정
                                                }

                                        }
                                )













# Create your views here.
