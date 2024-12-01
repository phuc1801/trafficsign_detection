import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# Tải mô hình YOLO
model = YOLO("best1.pt")  # Đảm bảo đường dẫn đúng tới file best.pt

# Hàm xử lý khi chọn ảnh
def select_image_or_video():
    file_path = filedialog.askopenfilename(filetypes=[("Image or Video Files", "*.jpg;*.jpeg;*.png;*.mp4;*.avi")])
    if not file_path:
        return

    # Kiểm tra file là ảnh hay video
    if file_path.lower().endswith(('jpg', 'jpeg', 'png')):
        # Nếu là ảnh, xử lý như ảnh
        process_image(file_path)
    elif file_path.lower().endswith(('mp4', 'avi')):
        # Nếu là video, xử lý video
        process_video(file_path)

# Hàm xử lý ảnh
def process_image(filepath):
    # Hiển thị ảnh gốc
    try:
        img = Image.open(filepath)
        img.thumbnail((400, 400))  # Resize ảnh cho giao diện
        img_tk = ImageTk.PhotoImage(img)
        label_img_original.configure(image=img_tk)
        label_img_original.image = img_tk
    except Exception as e:
        print(f"Lỗi khi hiển thị ảnh gốc: {e}")
        return

    # Dự đoán với YOLO
    results = model.predict(source=filepath, save=True, conf=0.5)
    save_dir = Path(results[0].save_dir)  # Thư mục lưu kết quả
    output_path = save_dir / filepath.split("/")[-1]  # Đường dẫn tới ảnh đã dự đoán

    # Kiểm tra xem có phát hiện đối tượng không
    if not results[0].boxes:  # Nếu không có bounding box nào
        print("Không phát hiện đối tượng nào trong ảnh.")
        return

    # Hiển thị ảnh kết quả trong form mới
    show_result_in_new_form(output_path)

# Hàm xử lý video
def process_video(filepath):
    # Đọc video
    cap = cv2.VideoCapture(filepath)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec để lưu video
    output_video = cv2.VideoWriter("output_video.mp4", fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Dự đoán với YOLO trên từng frame
        results = model.predict(source=frame, conf=0.5)

        # Lấy ảnh đã được gắn bounding box
        output_frame = results[0].plot()  # Plot bounding box lên frame

        # Ghi lại frame đã xử lý vào video
        output_video.write(output_frame)

        # Hiển thị video trực tiếp (tùy chọn)
        cv2.imshow("Video", output_frame)

        # Thoát video khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

    print("Video đã được lưu tại output_video.mp4")

# Hàm hiển thị ảnh kết quả trong form mới
def show_result_in_new_form(image_path):
    try:
        # Tạo cửa sổ mới để hiển thị ảnh kết quả
        result_window = tk.Toplevel(root)
        result_window.title("Kết quả phân tích")

        img_result = Image.open(image_path)
        img_result.thumbnail((600, 600))  # Resize ảnh để phù hợp với cửa sổ mới
        img_result_tk = ImageTk.PhotoImage(img_result)

        label_img_result = tk.Label(result_window, image=img_result_tk)
        label_img_result.image = img_result_tk
        label_img_result.pack(padx=20, pady=20)
    except Exception as e:
        print(f"Lỗi khi hiển thị ảnh kết quả: {e}")

# Giao diện chính
root = tk.Tk()
root.title("Traffic Sign Detection")
root.geometry("500x600")

# Nút chọn ảnh hoặc video
btn_select = tk.Button(root, text="Chọn ảnh hoặc video", command=select_image_or_video)
btn_select.pack(pady=20)

# Nhãn và khu vực hiển thị ảnh gốc
label_img_original = tk.Label(root, text="Ảnh gốc")
label_img_original.pack()

# Chạy ứng dụng
root.mainloop()
