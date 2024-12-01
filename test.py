import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO
import os

# Tải mô hình YOLO
model = YOLO("best1.pt")  # Đường dẫn tới file best.pt

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

# Hàm chọn và xử lý ảnh
def select_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not filepath:
        return

    # Dự đoán với YOLO
    results = model.predict(source=filepath, conf=0.5)
    show_result_with_labels(filepath, results)

# Hàm hiển thị ảnh với nhãn tiếng Việt và độ chính xác
def show_result_with_labels(image_path, results):
    # Mở ảnh bằng Pillow
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Sử dụng font hỗ trợ tiếng Việt (Arial Bold, size lớn hơn)
    try:
        font = ImageFont.truetype("arialbd.ttf", size=20)  # Dùng Arial Bold với cỡ chữ 50
    except:
        font = ImageFont.load_default()

    # Vẽ bounding box, nhãn và độ chính xác lên ảnh
    for box in results[0].boxes:
        # Lấy chỉ số lớp (class index) và ánh xạ sang mã nhãn
        cls_index = int(box.cls)
        cls_id = model.names[cls_index]  # Tên lớp từ mô hình
        label = label_mapping.get(cls_id, "Không xác định")  # Tên tiếng Việt

        # Lấy độ chính xác (confidence score)
        confidence = box.conf[0]  # Confidence score
        confidence_text = f"{confidence:.2f}"  # Định dạng số thập phân (2 chữ số)

        # Lấy tọa độ bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Vẽ bounding box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        # Vẽ nhãn và độ chính xác (màu tím, chữ lớn hơn)
        text = f"{label} ({confidence_text})"
        draw.text((x1, y1 - 20), text, fill="green", font=font)  # Chữ lớn hơn và cách xa hơn

    # Hiển thị ảnh kết quả trong form mới
    result_window = tk.Toplevel(root)
    result_window.title("Kết quả phân tích")

    img_tk = ImageTk.PhotoImage(img)
    label_result = tk.Label(result_window, image=img_tk)
    label_result.image = img_tk
    label_result.pack(padx=10, pady=10)

    # Lưu ảnh kết quả nếu cần
    img.save("result_with_labels.jpg")

# Giao diện chính
root = tk.Tk()
root.title("Traffic Sign Detection")
root.geometry("500x600")

# Nút chọn ảnh
btn_select = tk.Button(root, text="Chọn ảnh", command=select_image)
btn_select.pack(pady=20)

# Chạy ứng dụng
root.mainloop()
