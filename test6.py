import threading
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO

# Tải mô hình YOLO
model = YOLO("best1.pt")  # Thay bằng đường dẫn tới file best.pt
# Không sử dụng CUDA, chạy trên CPU
model.cpu()  # Sử dụng CPU

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
    "P.112": "Cấm người đi bộ",
    "P.115": "Hạn chế trọng lượng xe",
    "P.117": "Hạn chế chiều cao",
    "P.123a": "Cấm rẽ trái",
    "P.123b": "Cấm rẽ phải",
    "P.124a": "Cấm Quay xe",
    "P.124b": "Cấm oto quay đầu",
    "P.124c": "Cấm xe ngựa",
    "P.125": "Cấm vượt",
    "P.128": "Tốc độ cho phép",
    "P.130": "Cấm dừng và đỗ xe",
    "P.131a": "Cấm đỗ xe",
    "P.137": "Cấm rẽ trái và phải",
    "P.245a": "Chướng ngại vật",
    "R.301a": "Đường đi thẳng",
    "R.301c": "Chỉ dẫn rẽ trái phải theo",
    "R.301d": "Chỉ dẫn được rẽ phải",
    "R.301e": "Chỉ dẫn được rẽ trái",
    "R.302a": "Hướng đi vòng sang phải",
    "R.302b": "Hướng phải đi vòng sang trái",
    "R.303": "Biển Giao nhau theo vòng xuyến",
    "R.407a": "Biển chỗ đỗ xe",
    "R.409": "Chỗ quay xe",
    "R.425": "Bện viện",
    "R.434": "Bễn xe buýt",
    "S.509a": "Biển cảnh báo nguy hiểm",
    "W.201a": "Đường cong trái",
    "W.201b": "Đường cong phải",
    "W.202a": "Giao nhau đường cong trái",
    "W.202b": "Giao nhau đường cong phải",
    "W.203b": "Đường trơn trượt",
    "W.203c": "Đường dốc",
    "W.205a": "Ngã ba",
    "W.205b": "Ngã tư",
    "W.205d": "Đường giao nhau chữ Y",
    "W.207a": "Cầu hẹp",
    "W.207c": "Đường giao tàu hỏa",
    "W.208": "Chỗ ngoặt nguy hiểm",
    "W.209": "Giao đèn có tín hiệu",
    "W.210": "Giao nhau có đường sát",
    "W.219": "Đường không bằng phẳng",
    "W.221b": "Khu vực trường học",
    "W.224": "Khu vực đông người",
    "W.225": "Cấm vượt",
    "W.227": "Tốc độ cho phép",
    "W.233": "Nguy hiểm khác",
    "W.235": "Đường đôi",
    "W.245a": "Chướng ngoại vật",
}

# Đường dẫn font hỗ trợ tiếng Việt
FONT_PATH = "arial.ttf"  # Thay bằng đường dẫn font

# Hàm nhận diện video
def process_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video")
        return

    # Tạo cửa sổ hiển thị kết quả
    result_window = ctk.CTkToplevel()
    result_window.title("Kết quả nhận diện video")
    result_window.geometry("800x600")

    video_label = ctk.CTkLabel(result_window, text="")
    video_label.pack(padx=10, pady=10)

    # Hàm xử lý video trong một luồng riêng biệt
    def process_frame():
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break

            # Giảm độ phân giải của video
            frame = cv2.resize(frame, (640, 480))

            # Chỉ xử lý mỗi 3 khung hình
            if frame_count % 3 == 0:
                results = model.predict(source=frame, conf=0.5)

                # Chuyển đổi khung hình BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Chuyển đổi khung hình thành PIL để vẽ bounding box
                frame_pil = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(frame_pil)
                try:
                    font = ImageFont.truetype(FONT_PATH, size=20)  # Font hỗ trợ tiếng Việt
                except:
                    font = ImageFont.load_default()

                for box in results[0].boxes:
                    cls_index = int(box.cls)
                    cls_id = model.names[cls_index]  # Lấy mã lớp từ mô hình
                    label = label_mapping.get(cls_id, "Không xác định")  # Lấy tên biển báo từ label_mapping
                    confidence = box.conf[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Vẽ bounding box
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

                    # Vẽ nhãn và độ chính xác
                    text = f"{label} ({confidence:.2f})"
                    draw.text((x1, y1 - 20), text, fill="green", font=font)

                # Chuyển đổi khung hình về BGR để sử dụng trong OpenCV
                frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                # Chuyển đổi từ BGR sang Tkinter Image
                frame_tk = ImageTk.PhotoImage(Image.fromarray(frame_bgr))
                video_label.configure(image=frame_tk)
                video_label.image = frame_tk

            frame_count += 1
            result_window.after(10, process_frame)

    # Khởi chạy luồng để xử lý video
    video_thread = threading.Thread(target=process_frame)
    video_thread.start()

    # Nút đóng cửa sổ
    close_button = ctk.CTkButton(result_window, text="Đóng", command=result_window.destroy)
    close_button.pack(pady=20)

# Giao diện chính
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Phát hiện biển báo giao thông")
app.geometry("500x600")

# Tiêu đề
title_label = ctk.CTkLabel(app, text="Phát hiện biển báo giao thông", font=("Arial", 24, "bold"))
title_label.pack(pady=20)

# Nút chọn ảnh
select_button = ctk.CTkButton(app, text="Nhận diện ảnh", command=lambda: print("Hàm nhận diện ảnh"), font=("Arial", 16))
select_button.pack(pady=10)

# Nút chọn video
video_button = ctk.CTkButton(app, text="Nhận diện video", command=process_video, font=("Arial", 16))
video_button.pack(pady=10)

# Nút thoát
exit_button = ctk.CTkButton(app, text="Thoát", command=app.destroy, font=("Arial", 16), fg_color="red")
exit_button.pack(pady=20)

# Chạy ứng dụng
app.mainloop()
