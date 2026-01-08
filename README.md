# Stream_FR: 실시간 웹캠 안면 인식(InsightFace) 데모

본 프로젝트는 **웹캠(실시간 스트림)** 입력에서 얼굴을 검출하고(InsightFace `FaceAnalysis`), **갤러리(gallary)에 등록된 인물과 유사도 기반으로 매칭**하여 화면에 **bbox + 이름(한글/영문) + 유사도 점수**를 표시하는 데모 코드입니다.  
또한 임계치(threshold) 미만으로 판단되는 얼굴은 **미등록(Unknown)** 으로 처리하며, 해당 프레임을 `unknown/` 폴더에 자동 저장합니다.

---

## 핵심 기능 요약

- **실시간 웹캠 스트리밍** (`cv2.VideoCapture(0)`)
- **InsightFace 기반 얼굴 검출 + 임베딩 추출** (`FaceAnalysis`)
- **갤러리 인물별 다중 이미지 임베딩을 활용한 매칭**
  - 한 사람(폴더) 안에 여러 장의 등록 이미지를 넣을 수 있음
  - 유사도 상위 2개를 평균내어 대표 유사도로 사용(노이즈 완화 목적)
- **임계치 기반 인식 성공/실패(Unknown) 분기**
- **bbox + (한글/영문) 이름 + 유사도 점수** 오버레이 출력(PIL)
- **미등록 얼굴/프레임 자동 저장** (`unknown/unknown_#.jpg`)
- **멀티스레드 추론**
  - 메인 스레드: 카메라 프레임 수집/표시
  - 백그라운드 스레드: 얼굴 인식 수행 및 결과 업데이트

---

## 폴더 구조(중요)

아래 구조를 반드시 지켜주세요. 코드에서 폴더명을 **`gallary`** 로 사용합니다(일반적으로 gallery 철자와 다름).

```
project/
 ├─ main.py                  # 사용자 코드 파일명은 자유
 ├─ NanumGothicBold.ttf      # 한글 렌더링용 폰트(필수)
 ├─ gallary/
 │   ├─ Ryan/                # 폴더명 = 영문 이름(en_name)
 │   │   ├─ 라이언           # "확장자 없는 파일명" 1개 = 한글 이름(kr_name)
 │   │   ├─ 1.jpg
 │   │   ├─ 2.jpg
 │   │   └─ ...
 │   ├─ DongHoon/
 │   │   ├─ 동훈
 │   │   ├─ 1.jpg
 │   │   └─ ...
 └─ unknown/
     ├─ unknown_0.jpg
     ├─ unknown_1.jpg
     └─ ...
```

### 갤러리 규칙 상세

- `gallary/<영문이름>/` 폴더를 사람(클래스) 단위로 만듭니다.
- 해당 폴더 안에는:
  1. **등록 얼굴 이미지들**: `*.jpg` 파일들 (여러 장 가능)
  2. **한글 이름 파일**: 확장자가 없는 **파일 1개**를 만들고, **파일명 자체**를 한글 이름으로 사용합니다.  
     예) `gallary/Ryan/라이언` (확장자 없음)

> 왜 이런 방식인가요?  
> 코드에서 `if '.jpg' not in img_name:`인 경우를 한글 이름(kr_name)으로 간주하고, 그 파일명(예: "라이언")을 화면에 표시합니다.

---

## 동작 원리(코드 기준 설명)

### 1) 초기화(`__init__`)
- 임계치 `threshold` 설정(기본 0.7)
- `gallary/`, `unknown/` 폴더 자동 생성
- `unknown/` 폴더에 이미 저장된 이미지 수를 세어 `unknown_cnt`로 이어 저장
- InsightFace 엔진 초기화:
  - `modelpack = 'buffalo_l'`
  - `providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]`
  - `ctx_id = 0` (GPU 0번 사용, CPU만 사용할 때 -1)
  - `det_size = 640x640` 사용
- 갤러리 로드 `_load_gallary()`
- 백그라운드 얼굴 인식 스레드 시작 `_face_recognition()`
- 메인 루프(웹캠 스트림) 시작 `_live_stream()`

### 2) 갤러리 로드(`_load_gallary`)
- `gallary/` 아래 폴더(사람)들을 순회
- 사람 폴더 내 `*.jpg` 파일마다 얼굴 검출 후 첫 얼굴의 임베딩(512-d)을 추출하여 리스트로 저장
- 한글 이름은 “확장자 없는 파일명”에서 읽어 `kr_name`으로 저장
- 최종적으로 다음 형태의 딕셔너리를 생성:

```python
gallary = {
  "Ryan": {
    "kr_name": "라이언",
    "embeddings": [emb1, emb2, ...]
  },
  ...
}
```

> 주의: 이미지에서 얼굴을 못 찾으면 경고를 출력하고 해당 이미지는 건너뜁니다.

### 3) 실시간 얼굴 인식(백그라운드 스레드)(`_face_recognition`)
- 메인 스레드가 `self.img`를 넣어줄 때까지 대기
- 매 프레임마다 `self.app.get(self.img)`로 얼굴 검출+임베딩 추출
- 각 얼굴에 대해:
  - 갤러리의 각 인물별 모든 임베딩과 cosine similarity(0~1로 정규화) 계산
  - 유사도 상위 2개 평균을 인물의 대표 점수로 사용
  - 대표 점수가 `threshold` 이상이면 등록 인물로 인식
  - 아니면 Unknown 처리 + 프레임을 `unknown/`에 저장
- 결과는 `self.results`에 `deepcopy`로 저장하여 메인 스레드가 안전하게 읽을 수 있도록 함

### 4) 시각화(`_draw_fr`)
- `self.img`에 `self.results`를 그립니다.
- PIL을 사용해 한글 텍스트 렌더링:
  - 등록(known): 빨간색 `(255,0,0)`
  - 미등록(unknown): 검정색 `(0,0,0)`
- `mode='kr'`(기본)일 때 한글 이름을 표시, `mode='en'`일 때 영문 이름 표시

### 5) 웹캠 스트리밍(`_live_stream`)
- `cv2.VideoCapture(0)`으로 웹캠 열기
- 해상도 1280x720 설정
- 프레임을 좌우 반전(`cv2.flip(img, 1)`) 후 `self.img`에 저장
- `_draw_fr()` 결과를 실시간 표시
- **ESC(27)** 키를 누르면 종료

### 6) 유사도 계산(`get_similarity`)
- 두 임베딩을 L2 normalize 후 cosine similarity 계산
- `[-1, 1]` 범위를 `[0, 1]`로 변환해 직관적인 점수 제공
- 수치 안정성 위해 0~1 범위로 clamp

---

## 설치 및 실행 가이드

### 1) 권장 실행 환경
- OS: Ubuntu / Windows 모두 가능 (GPU 가속은 환경에 따라 다름)
- Python: 3.8 이상 권장
- GPU: NVIDIA GPU + CUDA 사용 시 실시간 성능 향상  
  (CPU도 가능하나 프레임 처리 속도가 낮을 수 있음)

### 2) 필수 패키지 설치

```bash
pip install numpy opencv-python insightface tqdm pillow
```

> 주의: InsightFace는 내부적으로 onnxruntime를 사용합니다.  
> GPU 사용을 원하면 CUDA가 포함된 onnxruntime/환경 구성이 필요할 수 있습니다(환경별 상이).

### 3) 한글 폰트 준비(필수)
코드에서 다음 폰트를 로드합니다.

```python
ImageFont.truetype('NanumGothicBold.ttf', 24)
```

따라서 프로젝트 루트에 **`NanumGothicBold.ttf`** 파일을 두어야 합니다.  
(없는 경우 실행 시 폰트 로드 에러가 발생합니다.)

### 4) 갤러리 등록 방법

1. `gallary/` 폴더 아래에 인물 폴더를 만듭니다. (폴더명 = 영문이름)
2. 인물 폴더 안에 얼굴 이미지 `*.jpg`를 넣습니다.
3. 같은 폴더에 **확장자 없는 파일 1개**를 만들고, **파일명=한글 이름**으로 설정합니다.

예시:

- `gallary/Ryan/1.jpg`
- `gallary/Ryan/2.jpg`
- `gallary/Ryan/라이언`  (확장자 없음)

### 5) 실행

```bash
python main.py
```

- 실행 직후 “갤러리 로드”가 진행됩니다.
- 창이 뜨면 웹캠 화면에 bbox/이름/유사도가 표시됩니다.
- **ESC**를 누르면 종료됩니다.

---

## 파라미터 튜닝 가이드

### threshold (기본 0.7)
- 값이 높을수록:
  - 오인식(False Accept) 감소
  - 미등록(Unknown) 증가(보수적)
- 값이 낮을수록:
  - 등록 인물로 더 잘 “잡히는” 대신
  - 오인식 위험 증가

추천:
- 실시간 데모/가벼운 인증: `0.65 ~ 0.75`
- 보안/인증 강화: `0.75 ~ 0.85` (환경에 따라 재조정 필요)

사용 예:
```python
if __name__ == '__main__':
    Stream_FR(threshold=0.75)
```

---

## 성능/안정성 관련 주의사항 (중요)

1) **Unknown 저장 로직**
- 현재 구현은 Unknown 판정 시 `self.img` 전체 프레임을 저장합니다.
- 같은 프레임에서 얼굴이 여러 개이고 모두 Unknown이면, 동일 프레임이 여러 번 저장될 수 있습니다.
- 장시간 실행 시 `unknown/` 폴더 용량이 빠르게 증가할 수 있으니 관리가 필요합니다.

2) **스레드 동기화**
- `self.img`, `self.results`를 스레드 간 공유합니다.
- `deepcopy`로 그리기/결과 업데이트를 안전하게 처리하고 있으나,
  더 엄격한 동기화가 필요하면 `threading.Lock` 도입을 고려하세요.

3) **갤러리 폴더 내 한글 이름 파일 누락**
- 현재 코드상 `kr_name`이 지정되지 않은 상태로 폴더 루프가 끝나면 예외가 발생할 수 있습니다.
- 반드시 인물 폴더마다 확장자 없는 “한글 이름 파일”을 하나 두는 것을 권장합니다.

4) **얼굴이 여러 명 있는 장면**
- `faces = self.app.get(self.img)`로 다중 얼굴 처리합니다.
- 각 얼굴에 대해 갤러리 전체와 비교하므로, 얼굴 수가 많아지면 연산량이 증가합니다.

---

## 커스터마이징 포인트

- **모델팩 변경**
  ```python
  modelpack = 'buffalo_l'  # buffalo_l, buffalo_s, antelopev2 등
  ```
- **det_size 변경**: 작은 값일수록 빠르지만 작은 얼굴 검출 성능이 떨어질 수 있음
  ```python
  self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))
  ```
- **표시 언어 변경**
  - `_draw_fr(mode='en')`로 호출하면 영문 이름 표기

---

## 트러블슈팅

### Q1. “웹캠을 열 수 없습니다”
- 다른 프로그램(Zoom/Teams/카메라 앱)이 웹캠을 점유 중인지 확인
- `cv2.VideoCapture(0)`의 인덱스(0/1/2)가 장치와 맞는지 확인

### Q2. 폰트 에러(OSError: cannot open resource)
- 프로젝트 루트에 `NanumGothicBold.ttf`가 있는지 확인
- 파일명 대소문자/경로가 정확한지 확인

### Q3. 갤러리 로드 중 “얼굴을 찾지 못했습니다”
- 등록 이미지에서 얼굴이 너무 작거나, 흐리거나, 측면/가림이 심한 경우
- 가능하면 **정면에 가깝고 선명한 사진**을 여러 장 등록하세요.

---

## 라이선스/주의
- 본 README는 제공된 코드 동작을 기준으로 작성되었습니다.
- InsightFace 및 관련 모델팩/가중치의 사용 조건은 각 프로젝트/모델의 라이선스를 따릅니다.

---

## 실행 예시(가장 간단)
```python
if __name__ == '__main__':
    Stream_FR(threshold=0.7)
```
