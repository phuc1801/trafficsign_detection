import customtkinter as ctk
from tkinter import filedialog
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO
import threading

# Tải mô hình YOLO
model = YOLO("best1.pt")  # Thay bằng đường dẫn tới file best.pt

# Ánh xạ mã nhãn sang tên hiển thị
label_mapping = {
    "DP.135": "Biển hết tấc cả các lệnh cấm",
    "P.102": "Biển cấm đi ngược chiều",
    "P.103a": "Cấm cấm ô tô",
    "P.103b": "Cấm ô tô rẽ phải",
    "P.103c": "Cấm oto rẽ trái",
    "P.104": "Cấm mô tô",
    "P.106a": "Cấm xe tải",
    "P.106b": "Cấm xe tải trên 2T",
    "P.107a": "Cấm ôtô khách ôtô tải",
    # Các nhãn khác...
}

# Hàm xử lý video
def process_video(video_path, video_label, result_window):
    cap = cv2.VideoCapture(video_path)  # Đọc video
    if not cap.isOpened():
        print("Không thể mở video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Dự đoán từ mô hình YOLO
        results = model.predict(source=frame, conf=0.5)

        # Chuyển đổi khung hình OpenCV thành ảnh PIL
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arialbd.ttf", size=20)
        except:
            font = ImageFont.load_default()

        # Vẽ các kết quả nhận diện lên ảnh
        for box in results[0].boxes:
            cls_index = int(box.cls)
            cls_id = model.names[cls_index]
            label = label_mapping.get(cls_id, "Không xác định")
            confidence = box.conf[0]
            confidence_text = f"{confidence:.2f}"
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            text = f"{label} ({confidence_text})"
            draw.text((x1, y1 - 20), text, fill="green", font=font)

        # Chuyển ảnh PIL sang ImageTk
        img_tk = ImageTk.PhotoImage(img)

        # Kiểm tra xem cửa sổ kết quả vẫn còn mở không
        if result_window.winfo_exists():
            # Cập nhật video_label với ảnh mới
            video_label.configure(image=img_tk)
            video_label.image = img_tk  # Giữ tham chiếu đến ảnh
            result_window.after(10, result_window.update)  # Cập nhật cửa sổ giao diện

    cap.release()

# Hàm chọn video
def select_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if video_path:
        result_window = ctk.CTkToplevel()
        result_window.title("Kết quả nhận diện từ video")
        result_window.geometry("800x600")

        # Sử dụng Label của tkinter thay vì CTkLabel để tránh vấn đề chữ mặc định
        video_label = ctk.CTkLabel(result_window)
        video_label.configure(text="")  # Đảm bảo không có chữ mặc định
        video_label.pack(padx=10, pady=10)

        # Chạy video trong một luồng con
        threading.Thread(target=process_video, args=(video_path, video_label, result_window), daemon=True).start()

# Giao diện chính
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Phát hiện biển báo giao thông")
app.geometry("500x600")

# Tiêu đề
title_label = ctk.CTkLabel(app, text="Phát hiện biển báo giao thông", font=("Arial", 24, "bold"))
title_label.pack(pady=20)

# Nút chọn video
select_video_button = ctk.CTkButton(app, text="Chọn video", command=select_video, font=("Arial", 16))
select_video_button.pack(pady=10)

# Nút thoát
exit_button = ctk.CTkButton(app, text="Thoát", command=app.destroy, font=("Arial", 16), fg_color="red")
exit_button.pack(pady=20)

# Chạy ứng dụng
app.mainloop()
