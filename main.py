import cv2
import math
from ultralytics import YOLO

# 전역 변수: 현재 추적 중인 사람의 ID
selected_person_id = None 

def auto_body_following_digital_zoom():
    global selected_person_id

    ZOOM_MARGIN = 1.2 
    SMOOTHING_ALPHA = 1
    
    # 💡 [디지털 줌 배율 변수] 기본값은 1.0 (확대 없음)
    digital_zoom_level = 1.0 

    print("YOLO 모델 로드 중...")
    model = YOLO('yolov8n.pt') 
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    target_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = target_w / target_h

    prev_cx, prev_cy = target_w // 2, target_h // 2
    prev_crop_w, prev_crop_h = target_w, target_h 

    window_name = "Auto Follow Cam (Digital Zoom)"
    cv2.namedWindow(window_name)

    print("\n[🤖 디지털 줌 + 자동 전신 팔로잉 카메라 시작]")
    print("=== [조작 가이드] ===")
    print(" [+] / [=] 키 : 디지털 줌 인 (Zoom IN)")
    print(" [-] / [_] 키 : 디지털 줌 아웃 (Zoom OUT)")
    print(" [q] 키       : 프로그램 종료")
    print("======================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()
        
        # 1. YOLO 추적
        results = model.track(source=frame, classes=[0], conf=0.5, persist=True, verbose=False)
        boxes = results[0].boxes
        
        current_tracks = []
        current_track_ids = []
        target_person_box = None

        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
            coords = boxes.xyxy.cpu().numpy()

            for track_id, coord in zip(track_ids, coords):
                x1, y1, x2, y2 = map(int, coord)
                current_tracks.append((track_id, (x1, y1, x2, y2)))
                current_track_ids.append(track_id)

        # 타겟 지정 로직
        if current_track_ids:
            if selected_person_id not in current_track_ids:
                selected_person_id = current_track_ids[0] 
                print(f"✅ 사람 감지! ID {selected_person_id}번 팔로잉 시작")
            
            for track_id, box in current_tracks:
                if track_id == selected_person_id:
                    target_person_box = box
                    break
        else:
            if selected_person_id is not None:
                print("❌ 화면에 사람이 없어 대기 상태로 복귀합니다.")
            selected_person_id = None

        # --- 2. 목표 계산 ---
        target_cx, target_cy = prev_cx, prev_cy
        # base_crop: AI가 계산한 기본 줌(박스에 맞춘 크기)
        base_crop_w, base_crop_h = prev_crop_w, prev_crop_h

        if selected_person_id is not None and target_person_box is not None:
            tx1, ty1, tx2, ty2 = target_person_box
            tw, th = tx2 - tx1, ty2 - ty1 
            
            target_cx = tx1 + tw // 2
            target_cy = ty1 + th // 2

            if tw / th > aspect_ratio:
                base_crop_w = int(tw * ZOOM_MARGIN)
                base_crop_h = int(base_crop_w / aspect_ratio)
            else:
                base_crop_h = int(th * ZOOM_MARGIN)
                base_crop_w = int(base_crop_h * aspect_ratio)
        else:
            base_crop_w, base_crop_h = target_w, target_h

        # 💡 [핵심] 사용자가 설정한 디지털 줌 배율을 반영합니다!
        # 화면을 'digital_zoom_level' 만큼 더 작게 잘라내면 줌인 효과가 발생합니다.
        target_crop_w = int(base_crop_w / digital_zoom_level)
        target_crop_h = int(base_crop_h / digital_zoom_level)

        # 너무 과도하게 확대되어 에러가 나지 않도록 최소 크기 제한 (100px)
        if target_crop_w < 100:
            target_crop_w = 100
            target_crop_h = int(100 / aspect_ratio)

        # --- 3. 부드러운 이동 (EMA) ---
        real_cx = int(SMOOTHING_ALPHA * target_cx + (1 - SMOOTHING_ALPHA) * prev_cx)
        real_cy = int(SMOOTHING_ALPHA * target_cy + (1 - SMOOTHING_ALPHA) * prev_cy)
        
        real_crop_w = int(SMOOTHING_ALPHA * target_crop_w + (1 - SMOOTHING_ALPHA) * prev_crop_w)
        real_crop_h = int(SMOOTHING_ALPHA * target_crop_h + (1 - SMOOTHING_ALPHA) * prev_crop_h)

        prev_cx, prev_cy = real_cx, real_cy
        prev_crop_w, prev_crop_h = real_crop_w, real_crop_h

        x1 = real_cx - real_crop_w // 2
        y1 = real_cy - real_crop_h // 2
        x2 = real_cx + real_crop_w // 2
        y2 = real_cy + real_crop_h // 2

        if x1 < 0: x1, x2 = 0, real_crop_w
        if y1 < 0: y1, y2 = 0, real_crop_h
        if x2 > target_w: x1, x2 = target_w - real_crop_w, target_w
        if y2 > target_h: y1, y2 = target_h - real_crop_h, target_h
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(target_w, x2), min(target_h, y2)

        cropped_img = original_frame[y1:y2, x1:x2]

        if cropped_img.size > 0:
            final_output = cv2.resize(cropped_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            crop_w = x2 - x1
            crop_h = y2 - y1

            # 박스 다시 그리기 (비율에 맞게)
            for track_id, box in current_tracks:
                bx1, by1, bx2, by2 = box
                
                new_bx1 = int((bx1 - x1) * (target_w / crop_w))
                new_by1 = int((by1 - y1) * (target_h / crop_h))
                new_bx2 = int((bx2 - x1) * (target_w / crop_w))
                new_by2 = int((by2 - y1) * (target_h / crop_h))

                if track_id == selected_person_id:
                    cv2.rectangle(final_output, (new_bx1, new_by1), (new_bx2, new_by2), (0, 0, 255), 3)
                    cv2.putText(final_output, f"TARGET [ID {track_id}]", (new_bx1, new_by1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.rectangle(final_output, (new_bx1, new_by1), (new_bx2, new_by2), (255, 0, 0), 2)
                    cv2.putText(final_output, f"ID {track_id}", (new_bx1, new_by1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

            # 💡 [디지털 줌 UI 표시]
            cv2.putText(final_output, f"D-ZOOM: {digital_zoom_level:.1f}X", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if selected_person_id is None:
                cv2.putText(final_output, "AUTO MODE: WAITING...", (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            cv2.imshow(window_name, final_output)
        else:
            cv2.imshow(window_name, original_frame)

        # 💡 [키보드 입력 제어: 디지털 줌 조작]
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('=') or key == ord('+'): # + 키 (또는 Shift 없는 =)
            digital_zoom_level += 0.2
            if digital_zoom_level > 20.0: # 최대 5배율 제한
                digital_zoom_level = 20.0
            print(f"🔎 줌 인: {digital_zoom_level:.1f}X")
        elif key == ord('-') or key == ord('_'): # - 키
            digital_zoom_level -= 0.2
            if digital_zoom_level < 1.0: # 기본 1배율 이하로 내려가지 않게 방지
                digital_zoom_level = 1.0
            print(f"🔍 줌 아웃: {digital_zoom_level:.1f}X")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    auto_body_following_digital_zoom()
