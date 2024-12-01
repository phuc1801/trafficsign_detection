import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# Tải mô hình YOLO
model = YOLO("best1.pt")  # Đảm bảo đường dẫn đúng tới file best.pt

# Ánh xạ nhãn YOLO sang tên biển báo tiếng Việt
label_mapping = {
    "DP.135": "Biển dừng",
    "P.102": "Cấm đi ngược chiều",
    "P.103a": "Cấm rẽ trái",
    "P.103b": "Cấm rẽ phải",
    "P.103c": "Cấm quay đầu",
    "P.104": "Cấm ô tô",
    "P.106a": "Cấm xe máy",
    "P.106b": "Cấm xe tải",
    "P.107a": "Cấm xe đạp",
    "P.112": "Hạn chế tốc độ",
    "P.115": "Cấm dừng, đỗ",
    "P.117": "Cấm bóp còi",
    "P.128": "Hạn chế chiều cao",
    # Thêm các nhãn khác nếu cần...
}

# Hàm xử lý khi chọn ảnh hoặc video
def select_image_or_video():
    file_path = filedialog.askopenfilename(filetypes=[("Image or Video Files", "*.jpg;*.jpeg;*.png;*.mp4;*.avi;*.mov")])
    if not file_path:
        return

    # Phân loại tệp
    if file_path.lower().endswith(('jpg', 'jpeg', 'png')):
        process_image(file_path)
    elif file_path.lower().endswith(('mp4', 'avi', 'mov')):
        process_video(file_path)

# Hàm xử lý ảnh
def process_image(filepath):
    try:
        # Mở ảnh và hiển thị
        img = Image.open(filepath).convert("RGB")
        img.thumbnail((400, 400))  # Resize ảnh cho giao diện
        img_tk = ImageTk.PhotoImage(img)
        label_img_original.configure(image=img_tk)
        label_img_original.image = img_tk

        # Dự đoán với YOLO
        results = model.predict(source=filepath, save=True, conf=0.5)
        save_dir = Path(results[0].save_dir)  # Đường dẫn lưu kết quả
        output_path = save_dir / os.path.basename(filepath)

        # Hiển thị kết quả
        show_result_with_labels(output_path, results)

    except Exception as e:
        print(f"Lỗi khi xử lý ảnh: {e}")

# Hàm xử lý video
def process_video(filepath):
    try:
        # Đọc video từ file
        cap = cv2.VideoCapture(filepath)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter("output_video.mp4", fourcc, 20.0, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Dự đoán từng frame với YOLO
            results = model.predict(source=frame, conf=0.5)

            # Vẽ bounding box và tên biển báo trên frame
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_index = int(box.cls)
                    cls_id = model.names[cls_index]
                    label_name = label_mapping.get(cls_id, "Không xác định")
                    confidence = box.conf[0]

                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                    # Hiển thị tên biển báo và độ tin cậy (độ chính xác)
                    text = f"{label_name} ({confidence:.2f})"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Ghi lại frame vào video đầu ra
            output_video.write(frame)
            cv2.imshow("Video", frame)

            # Nhấn 'q' để dừng video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
        print("Video đã lưu tại: output_video.mp4")

    except Exception as e:
        print(f"Lỗi khi xử lý video: {e}")

# Hàm hiển thị ảnh kết quả và ánh xạ nhãn sang tên biển báo tiếng Việt
def show_result_with_labels(image_path, results):
    try:
        result_window = tk.Toplevel(root)
        result_window.title("Kết quả phân tích")

        img_result = Image.open(image_path).convert("RGB")
        img_result.thumbnail((600, 600))  # Resize ảnh
        img_result_tk = ImageTk.PhotoImage(img_result)

        label_img_result = tk.Label(result_window, image=img_result_tk)
        label_img_result.image = img_result_tk
        label_img_result.pack(padx=20, pady=20)

        # Ánh xạ nhãn YOLO sang tên biển báo
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_index = int(box.cls)  # Lấy chỉ số lớp (class index)
                cls_id = model.names[cls_index]  # Tên lớp từ mô hình YOLO
                label_name = label_mapping.get(cls_id, "Không xác định")  # Ánh xạ nhãn sang tên biển báo

                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                text = f"{label_name} ({box.conf[0]:.2f})"
                cv2.putText(img_result, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        img_result_tk = ImageTk.PhotoImage(img_result)
        label_img_result.configure(image=img_result_tk)
        label_img_result.image = img_result_tk

    except Exception as e:
        print(f"Lỗi hiển thị kết quả: {e}")

# Giao diện chính
root = tk.Tk()
root.title("Traffic Sign Detection")
root.geometry("500x600")

# Nút chọn ảnh hoặc video
btn_select = tk.Button(root, text="Chọn ảnh hoặc video", command=select_image_or_video)
btn_select.pack(pady=20)

# Khu vực hiển thị ảnh gốc
label_img_original = tk.Label(root, text="Ảnh gốc")
label_img_original.pack()

# Chạy ứng dụng
root.mainloop()
