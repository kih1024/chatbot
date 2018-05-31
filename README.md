### 프로젝트 : 영화 정보 제공 Chatbot

#### 사용 라이브러리 :
 - Tensorflow
 - Numpy
 - Django
 - Sklearn
 - Konlpy
 - Mecab
 - Genlsim
 - Pandas
 - BeautifulSoup
 
#### 정보 제공:
 - KMDB Open API
 - 영화진흥위원회 Open API
 - 네이버 영화
 - Yes24 영화 예매

### 데이터: 
 - 입력: 클라이언트의 예상 질문

    - Ex) "영화배우 #하정우 에 대해서 검색해줘"
    
 - 출력: 사용자의 의도

     - Ex) 배우 검색

### 새로운 모델 트레이닝:

 - 명령: python3 train.py training_data.file parameters.json new
 - Ex) ```python3 train.py ./data/movie_dataset.csv.zip ./parameters.json new```
 
 새로운 폴더와 함께 모델 checkpoint, 단어장 pickle파일, Word2Vec, FastText 벡터 파일이 만들어질 것이다.

### 존재하는 모델 트레이닝:

 - 명령: python3 train.py training_data.file parameters.json model_name.folder(directory)
 - Ex) ```python3 train.py ./data/movie_dataset.csv.zip ./parameters.json ./trained_model_1479757124/```
 
 이미 이전에 만들어진 모델을 더 교육 시킬 수 있다. 갱신되는 checkpoint파일이 새로 만들어질 것이다.

### 모델 예측:

 - 명령: python3 my_predict.py ./trained_model_directory/
 - Ex) ```python3 my_predict.py ./trained_model_1479757124/```

트레이닝한 모델을 통해서 클라이언트의 입력에 따른 출력 결과 및 영화 정보를 제공해준다.

 
### 주의 사항:
 - 서버-클라이언트 환경에서 쓸 경우에는 my_predict.py에서 51, 52, 67, 87줄에서 사용자 환경에 맞게 다시 경로 설정 해야함
 - 서버 버전의 경우 장고와 카카오톡 플러스친구 api 에 맞게 작성 되었다. 서버 돌릴시 카카오톡 플러스친구와 같이 돌려야 한다.
 
### 카카오톡 플러스친구 'MovieChatBot'을 통해서 영화 정보 제공 Chatbot을 이용할 수 있습니다.