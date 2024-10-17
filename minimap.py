import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim  # Requires scikit-image library

def sanitize_filename(name):
    import re
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(' ', '_')
    return name

def load_and_preprocess_map_layouts(map_layout_dir, hsv_lower_layout, hsv_upper_layout):
    map_layouts = {}
    for filename in os.listdir(map_layout_dir):
        if filename.endswith('_layout.png'):
            map_name = filename.replace('_layout.png', '')
            layout_path = os.path.join(map_layout_dir, filename)
            layout_image = cv2.imread(layout_path)
            if layout_image is not None:
                # Apply HSV mask to the map layout image
                layout_mask = create_color_mask(layout_image, hsv_lower_layout, hsv_upper_layout)
                map_layouts[map_name] = layout_mask
            else:
                print(f"Failed to load image: {layout_path}")
    return map_layouts

def create_color_mask(image, hsv_lower, hsv_upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    return mask

def adjust_hsv_values(image, window_name='Adjust HSV'):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def nothing(x):
        pass

    cv2.namedWindow(window_name)
    # Create trackbars for color change
    cv2.createTrackbar('Hue Min', window_name, 0, 179, nothing)
    cv2.createTrackbar('Hue Max', window_name, 179, 179, nothing)
    cv2.createTrackbar('Sat Min', window_name, 0, 255, nothing)
    cv2.createTrackbar('Sat Max', window_name, 255, 255, nothing)
    cv2.createTrackbar('Val Min', window_name, 0, 255, nothing)
    cv2.createTrackbar('Val Max', window_name, 255, 255, nothing)

    while True:
        h_min = cv2.getTrackbarPos('Hue Min', window_name)
        h_max = cv2.getTrackbarPos('Hue Max', window_name)
        s_min = cv2.getTrackbarPos('Sat Min', window_name)
        s_max = cv2.getTrackbarPos('Sat Max', window_name)
        v_min = cv2.getTrackbarPos('Val Min', window_name)
        v_max = cv2.getTrackbarPos('Val Max', window_name)

        hsv_lower = np.array([h_min, s_min, v_min])
        hsv_upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Display the mask separately
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((image, result, mask_bgr))

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print(f"Saved HSV values: Lower-{hsv_lower}, Upper-{hsv_upper}")
            cv2.destroyWindow(window_name)
            return hsv_lower, hsv_upper
        elif key == ord('q'):
            print("Exited HSV adjustment without saving.")
            cv2.destroyWindow(window_name)
            exit()

def preprocess_mini_map(mini_map, hsv_lower, hsv_upper):
    mask = create_color_mask(mini_map, hsv_lower, hsv_upper)
    cv2.imshow('Mini-map Mask', mask)  # Display the mask
    return mask

def compare_with_map_layouts(mini_map_mask, map_layouts):
    similarity_scores = {}
    for map_name, layout_mask in map_layouts.items():
        # Resize layout_mask to match mini_map_mask size
        layout_mask_resized = cv2.resize(layout_mask, (mini_map_mask.shape[1], mini_map_mask.shape[0]))

        # Compute Structural Similarity Index (SSIM)
        score, _ = ssim(mini_map_mask, layout_mask_resized, full=True)
        # Since SSIM ranges from -1 to 1, and higher is better, we invert it to 'lower is better'
        similarity_scores[map_name] = 1 - score
    return similarity_scores

def main():
    map_layout_dir = 'map_layouts'
    video_path = 'videos/valorant_gameplay.mp4'
    map_check_interval = 1  # in seconds

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # Read the first frame to select ROI and adjust HSV
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        cap.release()
        exit()

    # Select the mini-map ROI on the first frame
    x, y, w, h = select_mini_map_roi(first_frame)
    mini_map_sample = first_frame[y:y+h, x:x+w]

    # Adjust HSV values for the mini-map in the video
    hsv_lower_mini, hsv_upper_mini = adjust_hsv_values(mini_map_sample, window_name='Adjust HSV for Mini-map')

    # Load a sample map layout image for HSV adjustment
    sample_layout_path = None
    for filename in os.listdir(map_layout_dir):
        if filename.endswith('_layout.png'):
            sample_layout_path = os.path.join(map_layout_dir, filename)
            break

    if sample_layout_path is None:
        print("No map layout images found in the directory.")
        cap.release()
        exit()

    sample_layout_image = cv2.imread(sample_layout_path)
    if sample_layout_image is None:
        print(f"Failed to load sample map layout image: {sample_layout_path}")
        cap.release()
        exit()

    # Adjust HSV values for the map layout images
    hsv_lower_layout, hsv_upper_layout = adjust_hsv_values(sample_layout_image, window_name='Adjust HSV for Map Layouts')

    # Load and preprocess map layouts with the adjusted HSV values
    map_layouts = load_and_preprocess_map_layouts(map_layout_dir, hsv_lower_layout, hsv_upper_layout)
    if not map_layouts:
        print("No map layouts found or failed to preprocess layouts.")
        cap.release()
        exit()

    # Reset video to start from the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default FPS
    frame_duration = int(1000 / fps)

    start_time = 0
    current_time = 0
    best_match = None

    cv2.namedWindow('Gameplay Video', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cropped Mini-map', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Best Match Layout Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mini-map Mask', cv2.WINDOW_NORMAL)  # Display the mini-map mask

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
            # Preprocess the mini-map (apply HSV masking)
            mini_map_mask = preprocess_mini_map(mini_map, hsv_lower_mini, hsv_upper_mini)
            if mini_map_mask is not None:
                similarity_scores = compare_with_map_layouts(mini_map_mask, map_layouts)

                best_match = min(similarity_scores, key=similarity_scores.get)
                print(f"\nTime elapsed: {current_time:.2f} seconds")
                print("Similarity scores (lower is better):")
                for map_name, score in similarity_scores.items():
                    print(f"{map_name}: {score}")
                print(f"Updated best match: {best_match}")

                best_match_layout_mask = map_layouts[best_match]
                # Resize to match display size
                best_match_layout_mask_resized = cv2.resize(best_match_layout_mask, (mini_map_mask.shape[1], mini_map_mask.shape[0]))
                cv2.imshow('Best Match Layout Mask', best_match_layout_mask_resized)

                start_time = current_time
            else:
                print("Failed to preprocess mini-map mask.")

        if best_match:
            cv2.putText(frame, f"Best Match: {best_match}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        key = cv2.waitKey(frame_duration)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def select_mini_map_roi(frame):
    roi = cv2.selectROI('Select Mini-map ROI', frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow('Select Mini-map ROI')
    x, y, w, h = map(int, roi)
    print(f"Selected ROI - x: {x}, y: {y}, w: {w}, h: {h}")
    return x, y, w, h

if __name__ == "__main__":
    main()