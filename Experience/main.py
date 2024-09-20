import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

# Hàm load tọa độ từ file .pts
def load_pts(pts_file):
    landmarks = []
    with open(pts_file) as f:
        lines = f.readlines()
        for line in lines[3:]:  # Bỏ qua 3 dòng đầu
            if line.strip() == "}":
                break
            x, y = line.strip().split()  # Load tọa độ từng điểm
            landmarks.append((float(x), float(y)))
    landmarks = np.array(landmarks)
    if landmarks.shape[0] != 68:
        raise ValueError(f"Invalid number of landmarks in {pts_file}: {landmarks.shape[0]}")
    return landmarks

# Hàm preprocess ảnh và tọa độ
def preprocess_image(image, landmarks, image_size=(256, 256)):
    h, w, _ = image.shape
    image = cv2.resize(image, image_size)
    
    landmarks[:, 0] = landmarks[:, 0] * (image_size[0] / w)  # Theo chiều rộng
    landmarks[:, 1] = landmarks[:, 1] * (image_size[1] / h)  # Theo chiều cao

    return image, landmarks
class HelenDataset(Dataset):
    def __init__(self, image_dir, transform=None, image_size=(256, 256)):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and 'mirror' not in f]
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pts_name = image_name.replace('.jpg', '.pts')
        pts_path = os.path.join(self.image_dir, pts_name)

        if not os.path.exists(pts_path):
            raise FileNotFoundError(f"Corresponding .pts file not found for {image_name}")

        landmarks = load_pts(pts_path)

        image, landmarks = preprocess_image(image, landmarks, self.image_size)

        if self.transform:
            image = self.transform(image)

        landmarks = torch.tensor(landmarks, dtype=torch.float32)

        # In ra biến index + 1
        print("Done")

        return image, landmarks
from torch.utils.data import DataLoader
from torchvision import transforms

# Biến đổi ảnh (có thể thêm augmentations nếu cần)
transform = transforms.Compose([
    transforms.ToTensor(),  # Chuyển ảnh về dạng Tensor
])

# Khởi tạo Dataset và DataLoader
image_dir = 'D:/Tự học/lmvh-proj-filter/src/data/ibug_300W_large_face_landmark_dataset/helen/trainset'  # Đường dẫn đến folder train
dataset = HelenDataset(image_dir, transform=transform, image_size=(256, 256))
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
