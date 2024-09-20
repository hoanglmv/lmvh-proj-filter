import cv2
import matplotlib.pyplot as plt
import torch 
# Hàm load file .pts
def load_pts(pts_file):
    landmarks = []
    with open(pts_file) as f:
        lines = f.readlines()
        for line in lines[3:]:  # Bỏ qua các dòng không chứa tọa độ
            if line.strip() == "}":
                break
            x, y = line.strip().split()
            landmarks.append((float(x), float(y)))
    return landmarks

# Hàm hiển thị ảnh và các điểm đặc trưng
def show_image_with_landmarks(image_path, pts_path):
    # Load hình ảnh sử dụng OpenCV
    image = cv2.imread(image_path)
    print(image.size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB để hiển thị đúng màu
    
    # Load các tọa độ điểm đặc trưng từ file .pts
    landmarks = load_pts(pts_path)
    
    # Hiển thị hình ảnh và các điểm đặc trưng
    plt.imshow(image)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Chuyển đổi từ numpy array sang PyTorch tensor
    img_tensor = torch.from_numpy(img_rgb)

    # Chuyển đổi dtype từ uint8 (mặc định của OpenCV) sang float32 để phù hợp với PyTorch
    img_tensor = img_tensor.float()

    # Hoán đổi các trục từ (height, width, channels) sang (channels, height, width)
    img_tensor = img_tensor.permute(2, 0, 1)

    # Nếu cần chuẩn hóa (đưa giá trị về khoảng [0, 1]):
    img_tensor = img_tensor / 255.0
    # Vẽ các điểm đặc trưng lên hình (màu xanh)
    print(img_tensor.shape)  # Ví dụ: torch.Size([3, 256, 256])
    for (x, y) in landmarks:
        plt.scatter(x, y, c='green', s=20)
    
    plt.axis('off')  # Tắt trục
    plt.show()

# Đường dẫn tới hình ảnh và file .pts
image_path = 'src\data\dataresized/232194_1.jpg'
pts_path = 'src\data\dataresized/232194_1.pts'

# Hiển thị hình ảnh với các điểm đặc trưng
show_image_with_landmarks(image_path, pts_path)
