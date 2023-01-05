from flask_cors import CORS, cross_origin

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_json
import base64
import time
#them
import json
import cv2
from flask import Flask, request
from flask import jsonify

interpreter = tf.lite.Interpreter(model_path='data_v2.tflite')
output_details = interpreter.get_output_details()
# print(output_details)
interpreter.resize_tensor_input(0, [5, 60, 33, 1])  # đổi batch size
interpreter.allocate_tensors()
# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

characters_mb = ['K', 'M', 'C', 'e', 'g', 'k', 'u', 'z', 't', '3', 'U', 'a', '5', 'A', 'y', 'H', 'q', 'Z', 'V', '7', 'Q', '2', '4', 'Y', '-', 'h', '8', 'v', '6', 'd', 'b', 'n', 'p', 'P', 'E', 'c', 'm', 'D', 'B', '9', 'N', 'G']
img_width = 320
img_height = 80

# Số lượng tối đa trong captcha ( dài nhất là 6)
max_length = 15

char_to_num_mb = layers.StringLookup(vocabulary=list(characters_mb), mask_token=None)

num_to_char_mb = layers.StringLookup(vocabulary=char_to_num_mb.get_vocabulary(), mask_token=None, invert=True)


# Đọc ảnh base64 và mã hóa
def encode_base64x(base64):
    img = tf.io.decode_base64(base64)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    return {"image": img}


# Dịch từ mã máy thành chữ
def decode_batch_predictions(pred, type):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = []
    #for res in results:
    results = tf.strings.reduce_join(num_to_char_mb(results)).numpy().decode("utf-8")
    output_text.append(results)
    return output_text


# load model mb
json_file_mb = open('model_mb.json', 'r')
loaded_model_json = json_file_mb.read()
json_file_mb.close()
loaded_model_mb = model_from_json(loaded_model_json)
loaded_model_mb.load_weights("model_mb.h5")

# hàm để truy cập: 127.0.0.1/run -> 127.0.0.1 là ip server
@app.route("/lemanh/captcha/mbbank", methods=["POST"])
@cross_origin(origin='*')
def mb():
    content = request.json
    start_time = time.time()
    imgstring = content['imgbase64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"))["image"]
    listImage = np.array([image_encode])
    preds = loaded_model_mb.predict(listImage)
    pred_texts = decode_batch_predictions(preds, "mb")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status = "success",captcha = captcha)

    return response

#Vietcombank
@app.route("/lemanh/captcha/vietcombank", methods=["POST"])
def predict():
    base64img = request.json['imgbase64']
    # base64img = request.form['base64img']
    # print(base64img,type(base64img))
    np_img = preprocess(base64img)
    length = len(np_img)
    np_img = np_img.reshape(length, 60, 33, 1)

    interpreter.set_tensor(0, np_img)
    interpreter.invoke()
    tflite_prediction = interpreter.get_tensor(output_details[0]['index'])
    result = np.argmax(tflite_prediction, axis=1)
    result = ''.join([str(elem) for elem in result])
    return json.dumps({'status':'success','data':result})


def from_base64(base64_data):
    data = base64_data.split(',')
    if len(data) > 1:
        base64_data = data[1]

    nparr = np.frombuffer(base64.b64decode(base64_data), np.uint8)
    return cv2.imdecode(nparr, 0)


def preprocess(base64img):
    X_array = []
    img = from_base64(base64img)
    # img = cv2.imread('src-vcb-captcha-test\\12681.jpg',0)

    ret2, dst1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dst2 = cv2.bitwise_not(dst1)
    # plt.subplot(1, 5, 1), plt.imshow(dst1,'gray')
    #   plt.subplot(2, 5, j+1),plt.imshow(dst2,'gray')
    # plt.show()

    ##################################### tích chập tìm line ngang
    id_kernel = np.array([[0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0],
                          [-1, -1, -1, -1, -1],
                          [0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0],
                          ])

    flt_img = cv2.filter2D(src=dst1, ddepth=-1, kernel=id_kernel)
    ############################################################ phép xor để loại bỏ line ngang dư thừa ------------ step 1
    flt_img = cv2.bitwise_xor(flt_img, dst1)
    #   plt.subplot(2, 5, j+1),plt.imshow(flt_img,'gray')
    # plt.show()
    # plt.subplot(1, 5, 2),plt.imshow(flt_img,'gray')

    invert_Affinedst11 = cv2.bitwise_not(flt_img)
    ret2, th1 = cv2.threshold(flt_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #   plt.subplot(2, 5, j+1),plt.imshow(invert_Affinedst11,'gray')
    # plt.show()
    invert_th2 = cv2.bitwise_not(th1)
    ###################################################### copy image ra xong biến đổi hình dạng để lấy contour -------- step 2.1

    kernel = np.ones((1, 3), np.uint8)
    Affinedst2 = cv2.morphologyEx(flt_img, cv2.MORPH_CLOSE, kernel)

    Affinedst2 = cv2.adaptiveThreshold(Affinedst2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

    input2 = Affinedst2
    #   plt.subplot(2, 5, j+1),plt.imshow(input2,'gray')
    # plt.show()

    ######################################################################################### đầu vào contour -------- step 3

    input = cv2.bitwise_not(Affinedst2)
    contours, hierarchy = cv2.findContours(input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    # The Bounding Rectangles will be stored here:
    boundRect = []

    # Alright, just look for the outer bounding boxes:
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv2.approxPolyDP(c, 1, True)
            if (cv2.boundingRect(contours_poly[i])[3] > 5):  ##### chỉ contour lấy cao hơn 5
                boundRect.append(cv2.boundingRect(contours_poly[i]))

    results = sorted(boundRect, key=lambda x: x[0])  ######### sort theo tọa độ x
    boundRectSorted1 = map(lambda x: [x[0], x[1], x[0] + x[2], x[1] + x[3]], results)
    boundRectSorted = list(boundRectSorted1)

    seperated_Box = []
    len_box = len(seperated_Box)
    w = boundRectSorted[len_box - 1][2] - boundRectSorted[0][0] + 5
    each = (int)(w / 5)
    begin = boundRectSorted[0][0]

    seperated_Box.append([begin, 0, begin + each, 60])
    seperated_Box.append([begin + each, 0, begin + each * 2, 60])
    seperated_Box.append([begin + each * 2, 0, begin + each * 3, 60])
    seperated_Box.append([begin + each * 3, 0, begin + each * 4, 60])
    seperated_Box.append([begin + each * 4, 0, begin + each * 5, 60])

    for i in range(len(seperated_Box)):
        # cv2.rectangle(dst2, (seperated_Box[i][0], seperated_Box[i][1]),(seperated_Box[i][2], seperated_Box[i][3]), color, 1)
        crop_img = dst2[seperated_Box[i][1]:seperated_Box[i][3], seperated_Box[i][0]:seperated_Box[i][2]]
        # plt.subplot(5, 5, m),plt.imshow(crop_img)
        top = seperated_Box[i][1] - 0
        # if (top<0): top=3
        bot = 60 - seperated_Box[i][3]
        # if (bot<0): bot=57
        w = seperated_Box[i][2] - seperated_Box[i][0]
        pad = (33 - w) // 2
        if (w % 2 == 0):
            pad2 = pad + 1
        else:
            pad2 = pad
        if (pad < 0): pad = 0
        # print(top,bot,pad)
        final_crop = cv2.copyMakeBorder(crop_img, top, bot, pad, pad2, cv2.BORDER_CONSTANT,
                                        value=[0])  ### thêm viền màu đen xung quanh
        # plt.subplot(2, 5, i+1),plt.imshow(final_crop)

        # cv2.imwrite("cut/"+captcha_string[i]+"/"+captcha_string+"_"+str(i)+".jpg",final_crop)
        final_crop = cv2.resize(final_crop, (33, 60))
        X_array.append(final_crop)
    X_array = np.array(X_array, dtype=np.float32)
    X_array = prep_pixels(X_array)
    return X_array


def prep_pixels(X_array):
    X_array = X_array / 255.0

    return X_array
# Chạy server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='80')  # -> chú ý port, không để bị trùng với port chạy cái khác
