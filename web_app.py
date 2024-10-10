from flask import Flask, render_template, request,jsonify
import os.path
from prototype_model import PrototypicalNetworks
from myfunction import get_information_from_data_support, delete_images_in_folder
from torchvision.models import resnet18
from torch import nn
import torch
from PIL import Image, ImageDraw, ImageFont, ImageColor
from torchvision import transforms
from random import random
import base64
import io
import glob
import shutil
from datetime import datetime


image_size = 28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"
path_data_support=os.path.join(app.config['UPLOAD_FOLDER'] ,'data_support')
path_data_test=os.path.join(app.config['UPLOAD_FOLDER'] ,'data_test')

model_path = "./model/25w5s_SGD_FSL_FNL_6_7.pth"

convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()
model = PrototypicalNetworks(convolutional_network).cuda()

# Load mô hình từ file .pth
model = torch.load(model_path)
model.eval()

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
         try:
            img = request.files['file']
            if img:
                # Save File just upload
                delete_images_in_folder(app.config['UPLOAD_FOLDER'] +'\data_test\lable_test\\')
                path_for_save = os.path.join(app.config['UPLOAD_FOLDER'] +'\data_test\lable_test\\', img.filename)
                print("Save = ", path_for_save)
                img.save(path_for_save)
                result = get_predict()
                # print(result)
                dim = (320, 320)
                transform_img=transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=1),transforms.Resize(dim),
                     ]
                    )
                img_predict = Image.open(path_for_save)
                img_predict = transform_img(img_predict)

                # Tạo đối tượng vẽ và font chữ
                draw = ImageDraw.Draw(img_predict)
                font = ImageFont.truetype("arial.ttf", size=15)

                # Vẽ văn bản lên hình ảnh
                text = result[0]
                position = (5, 15)
                draw.text(position, text,font=font) 

                # Lưu hình ảnh
                img_predict.save(path_for_save)
                first_occurrence = path_for_save.find("\\")
                # second_occurrence = path_for_save.find("\\", first_occurrence + 1)
                path_img_predict=path_for_save[first_occurrence:].replace("\\", "/")
                print(path_img_predict)
                return render_template("index.html", user_image = path_img_predict ,
                                           msg="UpLoad Successfull!!")

            else:
                return render_template('index.html', msg='Select File For Upload')

         except Exception as ex:
            # If Error
            print(ex)
            return render_template('index.html', msg='Cannot Recognize !!')

    else:
        # If is GET -> show UI upload
        return render_template('index.html', message=0)
@app.route("/page2", methods=['POST'])
def detech_image_canvas():
    # Nhận dữ liệu ảnh từ frontend
    image_data = request.form['image_data']
    delete_images_in_folder(app.config['UPLOAD_FOLDER'] +'\data_test\lable_test\\')
    save_img(image_data)
    result = get_predict()

    dim = (320, 320)
    transform_img=transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),transforms.Resize(dim),
            ]
        )
    path_for_save = os.path.join(app.config['UPLOAD_FOLDER'] +'\data_test\lable_test\\', 'character.png')
    img_predict = Image.open(path_for_save)
    img_predict = transform_img(img_predict)

    # Tạo đối tượng vẽ và font chữ
    draw = ImageDraw.Draw(img_predict)
    font = ImageFont.truetype("arial.ttf", size=15)

    # Vẽ văn bản lên hình ảnh
    text = result[0]
    position = (5, 15)
    draw.text(position, text,font=font) 

    # Lưu hình ảnh
    img_predict.save(path_for_save)
    response = {
        'message': text
    }

    # Trả về phản hồi JSON
    return jsonify(response)

@app.route("/page2", methods=['GET'])
def home_page2():
    return render_template('index.html')

def save_img(image_data):
    # Kích thước ảnh
    width = 300
    height = 300
    
    # Tạo đối tượng hình ảnh trắng đen
    image = Image.new('L', (width, height), color=255)
    
    # Tạo đối tượng vẽ trên ảnh
    draw = ImageDraw.Draw(image)
    
    # Chuyển đổi dữ liệu ảnh từ base64 thành điểm ảnh
    byte_data = base64.b64decode(image_data.split(',')[1])
    stream = io.BytesIO(byte_data)
    
    # Vẽ lên ảnh
    draw.bitmap((0, 0), Image.open(stream))
    thumbnail_size = (105, 105)
    image_resize = image.resize(thumbnail_size)
    # Lưu ảnh trắng đen
    image_resize.save('./static/data_test/lable_test/character.png')
def get_predict():
    batch_tensor,lst_unique_lable,lst_tensor_lable=get_information_from_data_support(path_data_support,image_size,device)
    batch_tensor_test,lst_unique_lable_test,lst_tensor_lable_test=get_information_from_data_support(path_data_test,image_size,device)
    model.eval()
    example_scores = model(
        batch_tensor,
        lst_tensor_lable,
        batch_tensor_test,
    ).detach()

    _, example_predicted_labels = torch.max(example_scores.data, 1)

    result=[]
    for i in example_predicted_labels:
        result.append(lst_unique_lable[i])
    return result

@app.route('/saveNewChar', methods=['POST'])
def save():
    # Lấy dữ liệu từ request
    image_data = request.form['image_data']
    folder_name = request.form['folder_name']
    if folder_name is None or str(folder_name).replace(' ','') =='':
        response = {
        'message': "None"
        }
        return jsonify(response)
    # Tạo thư mục lưu trữ nếu chưa tồn tại
    save_folder = os.path.join(app.root_path, 'static', 'data_support', folder_name)
    os.makedirs(save_folder, exist_ok=True)

    # Lấy thời gian hiện tại
    current_time = str(datetime.now().time())
    current_time = current_time.replace(':','_').replace('.','_')

    # Lưu kí tự vào file ảnh
    image_path = os.path.join(save_folder, f'char{current_time}.png')
    # Kích thước ảnh
    width = 300
    height = 300
    
    # Tạo đối tượng hình ảnh trắng đen
    image = Image.new('L', (width, height), color=255)
    
    # Tạo đối tượng vẽ trên ảnh
    draw = ImageDraw.Draw(image)

    # Chuyển đổi dữ liệu ảnh từ base64 thành điểm ảnh
    byte_data = base64.b64decode(image_data.split(',')[1])
    stream = io.BytesIO(byte_data)
    
    # Vẽ lên ảnh
    draw.bitmap((0, 0), Image.open(stream))
    
    # Lưu ảnh trắng đen
    # image.save(image_path)
    thumbnail_size = (105, 105)
    image_resize = image.resize(thumbnail_size)
    image_resize.save(image_path)

    num_img = count_images_in_folder(save_folder)
    # return 'save char success'
    response = {
        'message': str(num_img)
    }

    # Trả về phản hồi JSON
    return jsonify(response)

@app.route('/cancel', methods=['POST'])
def cancelAddNewChar():
    folder_name = request.form['folder_name']
    if folder_name is None or str(folder_name).replace(' ','') =='':
        return render_template('index.html', message=0)
    else:
        folder_path = os.path.join(app.root_path, 'static', 'data_support', folder_name)
        if os.path.exists(folder_path):
            # Xóa toàn bộ nội dung bên trong folder
            shutil.rmtree(folder_path)
            print(f"Đã xóa folder '{folder_path}' thành công.")   
        return render_template('index.html', message=0)
def count_images_in_folder(folder_path):
    image_extensions = ['.png']  # Các định dạng ảnh được chấp nhận
    image_count = 0

    # Sử dụng glob để tìm các file ảnh trong thư mục
    image_files = glob.glob(os.path.join(folder_path, '*'))

    # Đếm số lượng file có định dạng hợp lệ
    for file in image_files:
        if os.path.isfile(file) and os.path.splitext(file)[1] in image_extensions:
            image_count += 1

    return image_count

def load_model(model):
    model = torch.load(model_path)
    model.eval()
    return model

init=get_predict()

if __name__ == '__main__':
    app.run(debug=True)
    