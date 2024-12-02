import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO
import cv2
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
    "P.107a": "Cấm ôtô khách ôtô khách",
    "P.112": "Cấm người đi bộ",
    "P.115": "Hạn chế trọng lượng xe",
    "P.117": "Hạn chế chiều cao",
    "P.123a": "Cấm rẽ trái",
    "P.123b": "Cấm rẽ phải",
    "P.124a": "Cấm Quay xe",
    "P.124b": "Cấm oto quay đầu",
    "P.124c": "Cấm rẽ và quay đầu",
    "P.125": "Cấm vượt",
    "P.128": "Cấm sử dụng còi",
    "P.130": "Cấm dừng và đỗ xe", # 131
    "P.131a": "Cấm đỗ dừng và đỗ",   #130
    "P.137": "Cấm dừng", #sai
    "P.245a": "Biển báo đi chậm",
    "R.301a": "Chỉ được đi thẳng",
    "R.301c": "Chỉ được rẽ trái",
    "R.301d": "Chỉ dẫn được rẽ phải",
    "R.301e": "Chỉ dẫn được rẽ trái",
    "R.302a": "Hướng đi vòng sang phải",
    "R.302b": "Hướng phải đi vòng sang trái",
    "R.303": "Biển Giao nhau theo vòng xuyến",
    "R.407a": "Biển đường một chiều",
    "R.409": "Chỗ quay xe",
    "R.425": "Bện viện",
    "R.434": "Bễn xe buýt",
    "S.509a": "Biển chú ý chiều cao",
    "W.201a": "Ngoặt nguy hiểm vòng bên trái",
    "W.201b": "Ngoặt nguy hiểm vòng bên phải",
    "W.202a": "Giao nhau đường cong trái",
    "W.202b": "Giao nhau đường cong phải",
    "W.203b": "Đường hẹp trái",
    "W.203c": "Đường hẹp phảp",
    "W.205a": "Đường giao nhau",
    "W.205b": "Đường giao nhau 205",
    "W.205d": "Đường giao nhau chữ T",
    "W.207a": "Giao nhau với đường không ưu tiên",
    "W.207c": "Giao nhau với đường không ưu tiên trái",
    "W.208": "Giao nhau với đường ưu tiên",
    "W.209": "Giao đèn có tín hiệu",
    "W.210": "Giao nhau có đường sát",
    "W.219": "Đường dốc",
    "W.221b": "Đường không bằng phẳng",
    "W.224": "Biển người đi bộ",
    "W.225": "Trẻ em qua đường",
    "W.227": "Báo hiệu công trường",
    "W.233": "Cảnh báo nguy hiểm",
    "W.235": "Đường đôi",
    "W.245a": "Đi chậm",
}

# Đường dẫn font hỗ trợ tiếng Việt
FONT_PATH = "arial.ttf"  # Thay bằng đường dẫn font
# Hàm xử lý video

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

# Hàm chọn và xử lý ảnh
def select_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not filepath:
        return
    results = model.predict(source=filepath, conf=0.5)
    show_result_with_labels(filepath, results)

# Hàm hiển thị kết quả trong form mới
def show_result_with_labels(image_path, results):
    # Tạo cửa sổ mới
    result_window = ctk.CTkToplevel()
    result_window.title("Kết quả phân tích")
    result_window.geometry("800x600")

    # Mở ảnh và xử lý
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arialbd.ttf", size=20)
    except:
        font = ImageFont.load_default()

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

    # Chuyển ảnh sang định dạng Tkinter
    img_tk = ImageTk.PhotoImage(img)

    # Hiển thị ảnh trên form mới
    label_image = ctk.CTkLabel(result_window, image=img_tk, text="")
    label_image.image = img_tk  # Giữ tham chiếu tới ảnh
    label_image.pack(padx=10, pady=10)

    # Nút đóng form
    close_button = ctk.CTkButton(result_window, text="Đóng", command=result_window.destroy)
    close_button.pack(pady=20)

# Hàm xử lý video từ camera
def process_camera():
    cap = cv2.VideoCapture(0)  # Mở camera (0 là camera mặc định)
    if not cap.isOpened():
        print("Không thể mở camera.")
        return

    # Tạo cửa sổ kết quả
    result_window = ctk.CTkToplevel()
    result_window.title("Kết quả nhận diện từ camera")
    result_window.geometry("800x600")

    # Label để hiển thị video
    video_label = ctk.CTkLabel(result_window)
    video_label.pack(padx=10, pady=10)

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            return

        # Dự đoán từ mô hình YOLO
        results = model.predict(source=frame, conf=0.5)

        # Chuyển đổi khung hình OpenCV thành ảnh PIL
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(FONT_PATH, size=20)
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

            # Tiếp tục cập nhật frame sau 10ms
            result_window.after(10, update_frame)

    update_frame()  # Bắt đầu cập nhật video


# Thêm thư viện cần thiết
from tkinter import ttk

# Hàm xử lý lựa chọn từ combobox
def handle_combobox_selection(event):
    selected_option = combobox.get()
    if selected_option == "Chọn ảnh":
        select_image()
    elif selected_option == "Nhận diện video":
        select_video()
    elif selected_option == "Nhận diện camera":
        process_camera()

# Giao diện chính
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Phát hiện biển báo giao thông")
app.geometry("500x600")

# Tiêu đề
title_label = ctk.CTkLabel(app, text="Phát hiện biển báo giao thông", font=("Arial", 24, "bold"))
title_label.pack(pady=20)

# Combobox chọn loại nhận diện
options = ["Chọn ảnh", "Nhận diện video", "Nhận diện camera"]
combobox = ttk.Combobox(app, values=options, state="readonly", font=("Arial", 14))
combobox.pack(pady=20)
combobox.bind("<<ComboboxSelected>>", handle_combobox_selection)
combobox.set("Chọn tùy chọn")  # Thiết lập giá trị mặc định

# Nút thoát
exit_button = ctk.CTkButton(app, text="Thoát", command=app.destroy, font=("Arial", 16), fg_color="red")
exit_button.pack(pady=20)

# Chạy ứng dụng
app.mainloop()


