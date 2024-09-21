import os
import shutil

# Đường dẫn folder gốc và folder đích
source_folder = 'D:\lmvh\lmvh-proj-filter\src\data\helen\\trainset'  # Đường dẫn tới folder 'train' của bạn
destination_folder = 'D:\lmvh\lmvh-proj-filter\src\data\helen_no_mirror'  # Đường dẫn tới folder đích

# Tạo folder đích nếu chưa tồn tại
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Duyệt qua các file trong folder gốc
for filename in os.listdir(source_folder):
    # Nếu file có định dạng .jpg và không phải là ảnh mirror
    if filename.endswith('.jpg') and not 'mirror' in filename:
        # Lấy tên gốc của file mà không có phần mở rộng
        base_filename = os.path.splitext(filename)[0]

        # Đường dẫn tới file ảnh và file .pts tương ứng
        jpg_file = os.path.join(source_folder, base_filename + '.jpg')
        pts_file = os.path.join(source_folder, base_filename + '.pts')

        # Kiểm tra nếu cả file ảnh và file .pts đều tồn tại
        if os.path.exists(jpg_file) and os.path.exists(pts_file):
            # Di chuyển file ảnh và file .pts sang folder đích
            shutil.copy(jpg_file, destination_folder)
            shutil.copy(pts_file, destination_folder)
            print(f"Đã di chuyển {jpg_file} và {pts_file} tới {destination_folder}")

print("Hoàn thành!")
