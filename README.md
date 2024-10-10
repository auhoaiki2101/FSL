## Đề Tài: Tìm Hiểu Về Few-Shot Learning Và Ứng Dụng
### Giảng viên hướng dẫn:
    ThS.Quách Đình Hoàng
### Sinh viên thực hiện:
    Nguyễn Anh Đắc		 mssv: 19133020
    Nguyễn Thanh Tân Kỷ	 mssv: 19133031
## Hướng dẫn cài đặt
### 1. Chuẩn bị
+ Phiên bản python sử dụng: _3.7.9_, trình soạn thảo: _VsCode_
### 2. Cài đặt thư viện
+ Để tránh ảnh hưởng đến các thư viện python trên máy hiện tại, chúng ta nên tạo một môi trường riêng để chạy ứng dụng này.
+ Tạo môi trường python riêng: 
	`py -m venv fsl`
    + Chúng có thể chỉ định 1 phiên bản python cụ thể nếu trên máy chúng ta có nhiều phiên bản: 
	`py -3.7 -m venv fsl`
+ Kích hoạt môi trường vừa tạo: 
	`.\fsl\Scripts\activate`
+ Sau khi cài đặt môi trường python thì tiến hành cài các thư viện cần file **_requirements.txt_**: 
	`pip install -r requirements.txt`
+ Nếu gặp lỗi liên quan đến torch lỗi vui lòng cài torch theo câu lệnh sau (đây là phiên bản torch phù hợp với python _3.7.x_): 
	`pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
### 3. Chạy app, đảm bảo bạn đang ở môi trường python vừa cài đặt của mình:
	`py .\web_app.py`

File **_[Few-shot-Learning_Model.ipynb](https://colab.research.google.com/drive/14nOQcWWx_Wo_uFD9RkMO7pl7NBc2Bx7y?usp=sharing)_** sẽ khởi tạo mô hình, sau đó đào tạo trên tập _Omniglot_ và lưu lại mô hình sau khi đào tạo vào drive. Chúng ta nên chạy file này ở trên nền tảng **_Colab_**. Sau khi lưu xong việc của chúng ta là sử dụng nó cho việc phân loại các kí tự viết tay.
