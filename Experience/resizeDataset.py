import os
import cv2

# Đường dẫn tới folder chứa ảnh và keypoint
destination_folder = 'src\data\dataset_no_mirror'  # Thay bằng folder đích của bạn
output_folder = 'src\data\dataresized'  # Đường dẫn tới folder lưu ảnh và tọa độ đã được resize

# Tạo folder mới nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Kích thước đích (256x256)
target_size = (256, 256)

# Hàm để đọc tọa độ keypoints từ file .pts
def read_keypoints(pts_file):
    keypoints = []
    with open(pts_file, 'r') as f:
        lines = f.readlines()[3:-1]  # Bỏ qua 3 dòng đầu và dòng cuối
        for line in lines:
            x, y = map(float, line.strip().split())
            keypoints.append((x, y))
    return keypoints

# Hàm để ghi lại tọa độ keypoints sau khi resize
def write_keypoints(pts_file, keypoints):
    with open(pts_file, 'w') as f:
        f.write("version: 1\n")
        f.write("n_points: {}\n".format(len(keypoints)))
        f.write("{\n")
        for (x, y) in keypoints:
            f.write(f"{x:.6f} {y:.6f}\n")
        f.write("}\n")

# Duyệt qua tất cả file .jpg trong folder đích
for filename in os.listdir(destination_folder):
    if filename.endswith('.jpg'):
        # Đọc ảnh gốc
        img_file = os.path.join(destination_folder, filename)
        img = cv2.imread(img_file)

        # Lấy kích thước gốc của ảnh
        original_size = img.shape[1], img.shape[0]  # (width, height)

        # Resize ảnh về 256x256
        resized_img = cv2.resize(img, target_size)

        # Lưu ảnh đã resize
        output_img_file = os.path.join(output_folder, filename)
        cv2.imwrite(output_img_file, resized_img)

        # Tìm file .pts tương ứng
        base_filename = os.path.splitext(filename)[0]
        pts_file = os.path.join(destination_folder, base_filename + '.pts')

        if os.path.exists(pts_file):
            # Đọc keypoints
            keypoints = read_keypoints(pts_file)

            # Scale lại keypoints theo tỉ lệ mới
            scale_x = target_size[0] / original_size[0]
            scale_y = target_size[1] / original_size[1]
            resized_keypoints = [(x * scale_x, y * scale_y) for (x, y) in keypoints]

            # Lưu lại keypoints đã resize
            output_pts_file = os.path.join(output_folder, base_filename + '.pts')
            write_keypoints(output_pts_file, resized_keypoints)

            print(f"Đã resize {filename} và cập nhật keypoints.")

print("Hoàn thành resize và cập nhật keypoints!")
