# InsightFace GPU Environment Setup (Windows 11 + Anaconda)

본 문서는 Windows 11 환경에서 NVIDIA GPU를 사용하여  
InsightFace를 **CUDA 기반으로 안정적으로 실행**하기 위한 설정 가이드이다.

실제 시행착오를 기반으로 작성되었으며,
- NumPy 2.x 충돌
- onnxruntime-gpu 버전 미스매치
- OpenCV / NumPy ABI 문제
를 모두 회피하는 **검증된 조합**만 사용한다.

---

## 0. 사전 조건 (필수)

### 하드웨어 / OS
- Windows 11
- NVIDIA GPU (RTX / GTX 계열)
- 최신 NVIDIA Driver 설치 완료

```bat
nvidia-smi
```

정상적으로 GPU 정보가 출력되어야 한다.

---

## 1. Conda 가상환경 생성

```bat
conda create -n insightface python=3.10 -y
conda activate insightface
```

> Python 3.10 기준으로 테스트됨  

---

## 2. CUDA 런타임 (Conda 방식)

```bat
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9 -y
```

---

## 3. PyTorch CUDA (선택이지만 권장)

```bat
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

---

## 4. NumPy 버전 고정 (매우 중요)

```bat
pip install "numpy<2" --force-reinstall
```

확인:
```bat
python -c "import numpy as np; print(np.__version__)"
```

---

## 5. ONNX Runtime GPU 설치 (핵심)

```bat
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.16.3
```

확인:
```bat
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

---

## 6. OpenCV 확인

```bat
python -c "import cv2, numpy as np; print('cv2', cv2.__version__, 'numpy', np.__version__)"
```

---

## 7. InsightFace 설치

```bat
pip install insightface opencv-python onnx
```

---

## 8. InsightFace GPU 초기화 테스트

```bat
python -c "from insightface.app import FaceAnalysis; app=FaceAnalysis(providers=['CUDAExecutionProvider','CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640,640)); print('OK')"
```

---

## 9. GPU 실제 사용 여부 확인

```bat
nvidia-smi -l 1
```

---

## 10. 검증된 최종 조합 요약

| 항목 | 버전 |
|---|---|
| Python | 3.10 |
| CUDA Runtime | 11.8 |
| cuDNN | 8.9 |
| NumPy | 1.26.4 |
| onnxruntime-gpu | 1.16.3 |
| OpenCV | 4.12.0 |
| InsightFace | 최신 |
