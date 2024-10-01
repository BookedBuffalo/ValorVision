import cv2
import numpy as np
import os
import time

def sanitize_filename(name):
    import re
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(' ', '_')
    return name

def load_and_preprocess_map_layouts(map_layout_dir):
    map_layouts = {}
    for filename in os.listdir(map_layout_dir):
        if filename.endswith('_layout.png'):
            map_name = filename.replace('_layout.png', '')
            safe_map_name = sanitize_filename(map_name)
            layout_path = os.path.join(map_layout_dir, filename)
            layout_image = cv2.imread(layout_path)
            if layout_image is not None:
                layout_mask = create_color_mask(layout_image)
                layout_shape = extract_shape(layout_mask)
                map_layouts[map_name] = layout_shape
            else:
                print(f"Failed to load image: {layout_path}")
    return map_layouts

def create_color_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_color = np.array([12, 0, 104])
    upper_color = np.array([86, 255, 132])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

def extract_shape(mask):
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    shape = np.zeros_like(mask)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]
        cv2.drawContours(shape, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return shape

def select_mini_map_roi(cap):
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video frame.")
        cap.release()
        exit()

    roi = cv2.selectROI('Select Mini-map ROI', frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow('Select Mini-map ROI')
    x, y, w, h = map(int, roi)
    print(f"Selected ROI - x: {x}, y: {y}, w: {w}, h: {h}")
    return x, y, w, h

def preprocess_mini_map(mini_map):
    mask = create_color_mask(mini_map)
    shape = extract_shape(mask)
    return shape

def compare_with_map_layouts(mini_map_shape, map_layouts):
    similarity_scores = {}
    for map_name, layout_shape in map_layouts.items():
        if mini_map_shape is not None and layout_shape is not None:
            if mini_map_shape.shape != layout_shape.shape:
                layout_shape_resized = cv2.resize(layout_shape, (mini_map_shape.shape[1], mini_map_shape.shape[0]))
            else:
                layout_shape_resized = layout_shape

            score = cv2.matchShapes(mini_map_shape, layout_shape_resized, cv2.CONTOURS_MATCH_I1, 0.0)
            similarity_scores[map_name] = score
        else:
            similarity_scores[map_name] = float('inf')
    return similarity_scores

def main():
    map_layout_dir = 'map_layouts'
    video_path = 'videos/valorant_gameplay.mp4'
    map_check_interval = 5  

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    x, y, w, h = select_mini_map_roi(cap)

    map_layouts = load_and_preprocess_map_layouts(map_layout_dir)
    if not map_layouts:
        print("No map layouts found. Please ensure the 'map_layouts' directory contains the map layout images.")
        exit()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 
    frame_duration = int(1000 / fps)

    start_time = 0
    current_time = 0 
    best_match = None

    cv2.namedWindow('Gameplay Video', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cropped Mini-map', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Best Match Layout', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mini-map Shape', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read the video frame.")
            break
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = current_frame / fps

        mini_map = frame[y:y+h, x:x+w]

        cv2.imshow('Gameplay Video', frame)
        cv2.imshow('Cropped Mini-map', mini_map)

        if current_time - start_time >= map_check_interval:
            mini_map_shape = preprocess_mini_map(mini_map)
            cv2.imshow('Mini-map Shape', mini_map_shape)
            similarity_scores = compare_with_map_layouts(mini_map_shape, map_layouts)

            best_match = min(similarity_scores, key=similarity_scores.get)
            print(f"\nTime elapsed: {current_time:.2f} seconds")
            print("Similarity scores (lower is better):")
            for map_name, score in similarity_scores.items():
                print(f"{map_name}: {score}")
            print(f"Updated best match: {best_match}")

            best_match_layout = map_layouts[best_match]
            cv2.imshow('Best Match Layout', best_match_layout)

            start_time = current_time

        if best_match:
            cv2.putText(frame, f"Best Match: {best_match}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        key = cv2.waitKey(frame_duration)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()