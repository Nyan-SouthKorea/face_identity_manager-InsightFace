import cv2
import numpy as np

def imread_kr(img_path):
    '''윈도우 아나콘다 환경에서 한국어 경로를 입력하면 이미지로 읽어서 cv2 로 반환'''
    try:
        return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        return cv2.imread(img_path)

def imwrite_kr(write_path, img):
    '''윈도우 아나콘다 환경에서 한국어 경로를 입력하면 다양한 이미지 형식으로 출력 가능'''
    try:
        ext = '.' + write_path.split('.')[-1]  # ".jpg", ".png" 형태로 만들어줌
        success, encoded_img = cv2.imencode(ext, img)
        if success:
            encoded_img.tofile(write_path)
    except:
        cv2.imwrite(write_path, img)

def smart_resize(img, max_size=1280, mode='long'):
    '''
    최대 변의 길이를 맞추면서 비율을 유지하여 이미지 리사이즈
    img: cv2 이미지
    max_size: 최대 크기
    mode: long / short로 나뉨. 긴 변을 기준으로 할 것인가, 짧은 변을 기준으로 할 것인가 정하기

    return: resize된 cv2 이미지 반환
    '''
    h, w, c = img.shape

    # 리사이즈 진행
    # (long 모드)
    if mode == 'long':
        if w > h:
            img = cv2.resize(img, (max_size, int(h/w*max_size)))
        else:
            img = cv2.resize(img, (int(w/h*max_size), max_size))
        
    # (short 모드)
    elif mode == 'short':
        if w > h:
            img = cv2.resize(img, (int(w/h*max_size), max_size))
        else:
            img = cv2.resize(img, (max_size, int(h/w*max_size)))

    # (에러 처리)
    else:
        print('mode가 잘 못 설정되었습니다. long / short 설정 필요. 아무런 변화 없이 이미지가 반환됩니다.')

    return img