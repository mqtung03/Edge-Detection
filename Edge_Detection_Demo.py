import cv2
import tkinter as tk
from tkinter import filedialog

# Khởi tạo tkinter
root = tk.Tk()
root.withdraw()  # Ẩn cửa sổ gốc

# Mở hộp thoại chọn tệp và lấy đường dẫn tệp
file_path = filedialog.askopenfilename(
    title="Chọn một ảnh",
    filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
)

if file_path:
    # Đọc ảnh từ đường dẫn được chọn
    img = cv2.imread(file_path)
    
    # Hiển thị ảnh gốc
    cv2.imshow('Original', img)
    cv2.waitKey(0)

    # Chuyển thành ảnh xám
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Làm mờ ảnh để dễ lấy cạnh
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)  # Sobel theo x
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)  # Sobel theo y
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)  # Sobel theo cả x và y

    # Hiển thị ảnh đã qua thuật toán Sobel
    cv2.imshow('Sobel X', sobelx)
    cv2.waitKey(0)
    cv2.imshow('Sobel Y', sobely)
    cv2.waitKey(0)
    cv2.imshow('Sobel X Y', sobelxy)
    cv2.waitKey(0)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    # Hiển thị  Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
 
    cv2.destroyAllWindows()
else:
    print("Không có ảnh nào được chọn.")
