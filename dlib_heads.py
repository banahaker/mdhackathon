import cv2
import dlib

def count_faces_dlib(image_path, max_width=800, max_height=600):
    # 載入 Dlib 的人臉檢測器
    detector = dlib.get_frontal_face_detector()

    # 讀取圖像
    img = cv2.imread(image_path)
    if img is None:
        print("Could not open or find the image.")
        return 0

    # 將圖像轉換為灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 檢測人臉
    faces = detector(gray)

    # 繪製矩形在檢測到的人臉上
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 縮放圖像以適應顯示視窗
    resized_img = resize_image(img, max_width, max_height)

    # 顯示圖像與檢測結果
    cv2.imshow('Detected Faces', resized_img)
    
    # 等待按鍵輸入以關閉視窗
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 回傳檢測到的人臉數量
    return len(faces)

def resize_image(image, max_width, max_height):
    # 取得圖像的尺寸
    height, width = image.shape[:2]

    # 計算縮放比例
    scale = min(max_width / width, max_height / height)
    
    # 進行縮放
    if scale < 1:  # 只有當圖像尺寸超過限制時才進行縮放
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image

    return resized_image


# 讀取命令行參數
import sys

if len(sys.argv) > 1:
    image_path = sys.argv[1]

# 範例使用
num_faces = count_faces_dlib(image_path)
print(f'Number of faces detected: {num_faces}')