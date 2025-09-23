import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# --- CẤU HÌNH ---
folder_path = r'C:\Users\TenNguoiDung\Downloads\KetQuaHuanLuyen'
# Tìm tất cả các file ảnh metrics.png trong thư mục
image_files = sorted(glob.glob("fold_*_metrics.png"))

if not image_files:
    raise FileNotFoundError("Không tìm thấy file ảnh nào có tên theo dạng 'fold_*_metrics.png'.")

# --- VẼ BIỂU ĐỒ TỔNG HỢP ---
# Tạo một lưới 3x2 để chứa 5 ảnh (1 ô sẽ bị trống)
fig, axes = plt.subplots(3, 2, figsize=(20, 24))

# Chuyển mảng axes 2D thành 1D để dễ lặp
axes = axes.flatten()

print(f"Đã tìm thấy {len(image_files)} file ảnh. Bắt đầu ghép ảnh...")

# Lặp qua từng file ảnh và vị trí trên lưới
for i, file_path in enumerate(image_files):
    # Đọc file ảnh
    img = mpimg.imread(file_path)
    
    # Hiển thị ảnh lên ô tương ứng
    axes[i].imshow(img)
    
    # Đặt tiêu đề cho từng ảnh con (ví dụ: "Fold 1")
    # Tách tên từ đường dẫn file
    title = file_path.split('_')[0].capitalize() + " " + file_path.split('_')[1]
    axes[i].set_title(title.replace("-", " "), fontsize=16)
    
    # Ẩn các trục tọa độ không cần thiết
    axes[i].axis('off')

# Ẩn ô cuối cùng nếu bị trống
if len(image_files) < len(axes):
    for i in range(len(image_files), len(axes)):
        axes[i].axis('off')

# Đặt tiêu đề chung cho toàn bộ ảnh lớn
fig.suptitle('Training Process for 5-Fold Cross-Validation', fontsize=24, y=0.95)

# Lưu ảnh tổng hợp lại
output_filename = 'training_metrics_combined.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Hoàn tất! Ảnh tổng hợp đã được lưu với tên: {output_filename}")
plt.show()