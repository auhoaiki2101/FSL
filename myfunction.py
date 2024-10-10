import torch
import os
from torchvision import transforms
from PIL import Image

def get_information_from_data_support (folder_path,image_size,device):
    # folder_path : "./static/data_support"
    # List để lưu trữ các tensor của từng ảnh
    image_tensors = []
    lst_lable_str=[]
    # Duyệt qua từng thư mục trong folder_path
    for folder_name in os.listdir(folder_path):
        folder_dir = os.path.join(folder_path, folder_name)
        index_lable = folder_dir.rfind('\\')
        # get label = name folder
        lable_name = folder_dir[index_lable+1:]
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize([
                    int(image_size * 1.15), int(image_size * 1.15)
                ]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        # Kiểm tra folder_dir có phải là thư mục
        if os.path.isdir(folder_dir):
            # List để lưu trữ các tensor của từng ảnh trong thư mục hiện tại
            folder_tensors = []
            
            # Duyệt qua từng ảnh trong thư mục hiện tại
            for image_name in os.listdir(folder_dir):
                image_path = os.path.join(folder_dir, image_name)
                
                # Kiểm tra image_path có phải là file ảnh
                if os.path.isfile(image_path):
                    # Đọc và chuyển đổi ảnh thành tensor
                    image = Image.open(image_path)
                    image_tensor = transform(image)
                    lst_lable_str.append(lable_name)
                    # Thêm tensor ảnh vào danh sách folder_tensors
                    folder_tensors.append(image_tensor)

            # Tạo tensor từ danh sách folder_tensors và thêm vào danh sách image_tensors
            if folder_tensors:
                folder_tensor = torch.stack(folder_tensors)
                image_tensors.append(folder_tensor)
    # Tạo tensor chứa tất cả các ảnh từ danh sách image_tensors
    batch_tensor = torch.cat(image_tensors).to(device)
    # Tạo danh sách số lớp tương ứng
    lst_unique_lable = list(set(lst_lable_str))
    lst_lable_int = []
    for value in lst_lable_str:
        index = lst_unique_lable.index(value)
        lst_lable_int.append(index)
    # In kích thước của tensor
    # print(batch_tensor.size())
    lst_tensor_lable=torch.tensor(lst_lable_int).to(device)
    return batch_tensor,lst_unique_lable,lst_tensor_lable

def delete_images_in_folder(folder_path):
    # Kiểm tra xem folder_path có tồn tại không
    if not os.path.exists(folder_path):
        print("Thư mục không tồn tại.")
        return

    # Lặp qua tất cả các tệp trong thư mục
    for file_name in os.listdir(folder_path):
        # Kiểm tra phần mở rộng của tệp có phải là hình ảnh hay không (có thể tùy chỉnh cho từng loại hình ảnh)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_path = os.path.join(folder_path, file_name)
            # Xóa tệp
            os.remove(file_path)
            print("Đã xóa:", file_path)

    print("Đã xóa tất cả các hình ảnh trong thư mục.")