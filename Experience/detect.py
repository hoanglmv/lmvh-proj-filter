import dlib
import cv2

# Đọc ảnh bằng OpenCV
image = cv2.imread('D:\lmvh\lmvh-proj-filter\src\data\helen/trainset/2203538277_1.jpg')

# Chuyển ảnh sang grayscale để tăng độ chính xác trong việc phát hiệns
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Khởi tạo face detector của dlib
detector = dlib.get_frontal_face_detector()

# Phát hiện các khuôn mặt trong ảnh
faces = detector(gray)

# Duyệt qua các khuôn mặt đã phát hiện
for i, face in enumerate(faces):
    x1 = face.left()   # Tọa độ x của góc trái trên
    y1 = face.top()    # Tọa độ y của góc trái trên
    x2 = face.right()  # Tọa độ x của góc phải dưới
    y2 = face.bottom() # Tọa độ y của góc phải dưới
    
    # Vẽ hình chữ nhật bao quanh khuôn mặt
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print(f"Khuôn mặt {i+1}: Tọa độ bounding box: Left: {x1}, Top: {y1}, Right: {x2}, Bottom: {y2}")

# Hiển thị ảnh kết quả
cv2.imshow('Image with faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
