import cv2
import numpy as np

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop


class InsightFaceEngine:
    def __init__(self):
        # ===== 여기서 그냥 값으로 고정 =====
        
        # 모델 리스트: 
        # buffalo_l, buffalo_s(표준)
        # antelopev2(최신 고성능)
        self.model_name = "antelopev2"
        self.det_size = (640, 640)
        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.ctx_id = 0

        self.app = FaceAnalysis(
            name=self.model_name,
            providers=self.providers
        )
        self.app.prepare(
            ctx_id=self.ctx_id,
            det_size=self.det_size
        )

    def detect_faces(self, bgr_img):
        """
        입력: BGR 이미지 (OpenCV)
        출력: 얼굴 리스트
          - bbox: (x1, y1, x2, y2)
          - score: confidence
          - kps: (5,2) keypoints
        """
        faces = self.app.get(bgr_img)

        results = []
        for f in faces:
            results.append({
                "bbox": f.bbox,
                "score": f.det_score,
                "kps": f.kps,
                "face": f
            })
        return results

    def align_face(self, bgr_img, kps):
        """
        입력:
          - 원본 BGR 이미지
          - keypoints (5,2)
        출력:
          - 정렬된 얼굴 (112x112 BGR)
        """
        aligned = norm_crop(bgr_img, kps, image_size=112)
        return aligned

    def get_embedding(self, aligned_bgr):
        """
        입력:
          - 정렬된 얼굴 이미지 (112x112 BGR)
        출력:
          - 임베딩 벡터 (512,)
        """
        faces = self.app.get(aligned_bgr)

        if len(faces) == 0:
            raise RuntimeError("정렬된 얼굴에서 얼굴을 찾지 못했습니다.")

        return faces[0].embedding

    def similarity(self, emb1, emb2):
        """
        cosine similarity
        """
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        return float(np.dot(emb1, emb2))

    def detect_and_align_all(self, bgr_img):
        """
        이미지 한 장에서:
          - 모든 얼굴 검출
          - 전부 정렬해서 리스트로 반환
        """
        detections = self.detect_faces(bgr_img)

        aligned_faces = []
        for d in detections:
            if d["kps"] is None:
                continue

            aligned = self.align_face(bgr_img, d["kps"])
            aligned_faces.append({
                "bbox": d["bbox"],
                "score": d["score"],
                "aligned": aligned,
                "face": d["face"]
            })

        return aligned_faces
