import time
import cv2

from model_insightface import InsightFaceEngine
from insightface.app import FaceAnalysis

def main():
    # ===== InsightFace 초기화 (GPU 우선) =====
    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))  # det_size는 속도/정확도 트레이드오프

    # ===== 웹캠 열기 =====
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows면 CAP_DSHOW가 보통 안정적
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. VideoCapture(0) 인덱스를 바꿔보세요.")
        return

    # (선택) 웹캠 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다.")
            break

        # ===== 얼굴 검출 =====
        faces = app.get(frame)

        # ===== bbox 그리기 =====
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            score = float(f.det_score)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{score:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ===== FPS 표시 =====
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)  # 부드럽게 표시

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("InsightFace Realtime (BBox only)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
