import torch
from ultralytics import YOLO

# Tải mô hình YOLO
model = YOLO("best1.pt")  # Đảm bảo rằng bạn đã tải mô hình trước

# Kiểm tra xem có GPU không, nếu có thì sử dụng GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiển thị thông báo trong terminal
if torch.cuda.is_available():
    print("GPU is available. Using GPU for inference.")
else:
    print("GPU is not available. Using CPU for inference.")

# Chuyển mô hình sang thiết bị đã xác định (GPU hoặc CPU)
model = model.to(device)

# Tiếp tục với các phần còn lại của mã của bạn...
