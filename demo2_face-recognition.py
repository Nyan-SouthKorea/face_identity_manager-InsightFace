# basic
import os
import time
from copy import deepcopy
import threading

# pip install
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# custom
from utils import imread_kr, imwrite_kr

class Stream_FR:
    def __init__(self, threshold=0.7):
        # 변수 초기화
        self.threshold = threshold
        self.results = []
        self.crop_face_pitch = 160

        # 폰트 초기화
        self.color_known = (255,0,0) # PIL 이미지 rgb 기준
        self.color_unknown = (0,0,0)
        self.font = ImageFont.truetype('NanumGothicBold.ttf', 24)

        # 필요 폴더 생성
        for folder in ['gallary', 'snapshot']:
            os.makedirs(folder, exist_ok=True)
                
        # 안면 인식 엔진 초기화
        print('안면 인식 엔진 초기화 중... ', end='')
        modelpack = 'buffalo_l' # buffalo_l, buffalo_s(표준) // antelopev2(최신 고성능)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        det_size = {'320':(320,320), '480':(480,480), '640':(640,640), '800':(800,800)} # 저사양/실시간서비스/보안/오프라인분석
        ctx_id = 0 # GPU:0 사용, CPU만 사용할 땐 -1
        self.app = FaceAnalysis(name=modelpack, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size['640'])
              
        # 갤러리 로드
        self.gallary = self._load_gallary()

        # 웹캠 실시간 안면 인식 쓰레드 구동
        self.img = None
        threading.Thread(target=self._face_recognition, daemon=True).start()

        # 웹캠 라이브 스트림 시작
        self._live_stream()

    def _load_gallary(self):
        '''gallary 폴더의 얼굴들을 로드하는 함수'''
        
        # 갤러리 dic 초기화
        gallary = {}

        # 폴더 규칙: 한글 이름, 영어 이름(", "로 구분)
        folder_list = os.listdir('gallary')
        # 폴더별로 얼굴 임베딩 추출
        for folder in tqdm(folder_list, desc='갤러리 로드'):
            kr_name, en_name = folder.split(', ')
            # 폴더 내 이미지별로 추출
            embeddings = []
            for img_name in os.listdir(f'gallary/{folder}'):
                # 이미지 정상적으로 읽고 얼굴 임베딩 추출
                img = imread_kr(f'gallary/{folder}/{img_name}')
                faces = self.app.get(img)
                if len(faces) == 0:
                    print(f'경고: gallary/{folder}/{img_name}에서 얼굴을 찾지 못했습니다')
                    continue
                # 첫 번째 하나의 얼굴만 사용(무조건 이미지 한 장 안에 하나의 얼굴만 있다고 가정)
                face = faces[0]
                embedding = face.embedding # 512차원 얼굴 임베딩 추출
                x1, y1, x2, y2 = face.bbox.astype(int)
                # crop_face는 나중에 활용 예정
                crop_face = img[y1:y2, x1:x2]
                crop_face = cv2.resize(crop_face, (self.crop_face_pitch, self.crop_face_pitch))                
                embeddings.append(embedding)
            # 갤러리에 저장
            gallary[en_name] = {'kr_name':kr_name, 'embeddings':embeddings}
        return gallary
        
    def _face_recognition(self):
        '''
        안면 인식 기능을 멀티 쓰레드로 수행하는 함수
        '''
        # self.img가 None이 아닐 때까지 대기
        while True:
            if self.img is not None:
                break
            time.sleep(0.01)
        
        # 안면 인식 메인 루프
        while True:
            # 안면 인식 수행
            faces = self.app.get(self.img)

            # 웹캠으로 입력된 얼굴들을 갤러리와 비교
            results = []
            for face in faces:
                webcam_embedding = face.embedding
                # 갤러리별로 비교(갤러리 내 1명의 사람에 대해 유사도 1,2등을 평균내어 대표 유사도로 사용)
                sim_dict = {}
                for en_name, info in self.gallary.items():
                    sims = []
                    for gallary_emb in info['embeddings']:
                        sim = self.get_similarity(webcam_embedding, gallary_emb)
                        sims.append(sim)
                    sims = sorted(sims, reverse=True)
                    # 상위 2개 유사도 평균 내기
                    avg_sim = np.mean(sims[:2])
                    sim_dict[en_name] = avg_sim
                # 최고 유사도를 가진 갤러리 인물 선택
                best_en_name = max(sim_dict, key=sim_dict.get)
                best_sim = sim_dict[best_en_name]
                # 임계치 이상일 때만 인식 성공 처리
                if best_sim >= self.threshold:
                    kr_name = self.gallary[best_en_name]['kr_name']
                    results.append({'en_name':best_en_name, 'kr_name':kr_name, 'similarity':best_sim, 'bbox':face.bbox.astype(int), 'det_score':float(face.det_score)})
                else:
                    results.append({'en_name':'Unknown', 'kr_name':'미등록', 'similarity':best_sim, 'bbox':face.bbox.astype(int), 'det_score':float(face.det_score)})
            
            # main 로직에서 업데이트 가능하도록 results 복사
            self.results = deepcopy(results)


    def _draw_fr(self, mode='kr'):
        '''
        안면 인식 결과를 이미지에 그려주는 함수
        self.img에다가 self.results를 그려서 반환한다.
        모드에 따라서 한국 이름, 영어 이름 등을 다르게 표시할 수 있다.
        '''
        # 그리는 도중에 초기화 되지 않도록 복사
        pil_img = Image.fromarray(cv2.cvtColor(deepcopy(self.img), cv2.COLOR_BGR2RGB))
        results = deepcopy(self.results)

        # pil 그리기 준비
        draw = ImageDraw.Draw(pil_img)

        # bbox 및 정보 그리기
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            det_score = result['det_score']
            en_name, kr_name = result['en_name'], result['kr_name']
            similarity = result['similarity']

            # 색상 지정 및 crop_face 좌측 상단에 붙이기
            if en_name == 'Unknown':
                draw_color = self.color_unknown
            else:
                draw_color = self.color_known
  
            # bbox 그리기
            draw.rectangle([(x1, y1), (x2, y2)], outline=draw_color, width=2) # rgb 컬러 기준

            # 텍스트 그리기(박스 왼쪽 아래)
            if mode == 'en':
                text = f'{en_name}, {round(similarity, 2)}'
            else:
                text = f'{kr_name}, {round(similarity, 2)}'
            draw.text((x1+5, y2), text, font=self.font, fill=draw_color)
        
        # pil -> cv2 변환
        draw_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return draw_img

    def _live_stream(self):
        '''
        main 로직으로써, 카메라로 입력된 비디오를 실시간으로 스트리밍 해준다.
        안면 인식 기능은 멀티쓰레드로 구동되어 정보를 전달 받아 표시한다.
        '''
        # 웹캠 초기화
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('웹캠을 열 수 없습니다')
            return
        
        # 웹캠 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 이미지 받아오기
        while True:
            ret, img = cap.read()
            if ret == False:
                print('프레임을 읽지 못했습니다')
                break

            # 이미지 좌우 반전
            self.img = cv2.flip(img, 1)

            # 이미지 그리기
            draw_img = self._draw_fr()
            
            # 이미지 출력(esc 누르면 종료)
            cv2.imshow('Face Recognition', draw_img)
            key = cv2.waitKey(1)

            # esc 버튼을 누르면 종료
            if key == 27:
                break
            # s 버튼 누르면 이미지 저장
            elif key == ord('s'):
                imwrite_kr(f'snapshot/snapshot_{int(time.time())}.jpg', img)
                print('스냅샷 저장 완료')
    
    def get_similarity(self, emb1, emb2):
        """임베딩 2개를 받아 0 ~ 1 범위의 유사도 점수를 반환"""
        # L2 normalize
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        # cosine similarity: [-1, 1]
        cos_sim = float(np.dot(emb1, emb2))
        # [-1, 1] -> [0, 1]
        sim_0_1 = (cos_sim + 1.0) / 2.0
        # 수치 안정성 (아주 미세한 오차 방지)
        return max(0.0, min(1.0, sim_0_1))

if __name__ == '__main__':
    Stream_FR()