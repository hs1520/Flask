import os
from flask_migrate import Migrate
from flask import Flask,render_template
import config
from exts import db
from models import ClassificationModel
import paho.mqtt.client as mqtt
import numpy as np
import json
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from keras.models import load_model
from PIL import Image
import io
from ultralytics import YOLO
import torch
import time
# global para
global session
global graph
global model
global count
global start_time
count = 0
start_time = time.time()
# flask and db part
app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)
migrate = Migrate(app, db)

model = YOLO("best-mid.pt")
classes = ["Persian", "Ragdoll", "Scottish_Fold", "Singapura", "Sphynx"]
processed_files = 0

# Create session and graph for tensorflow
session = tf.compat.v1.Session(graph=tf.compat.v1.Graph())
graph = session.graph  # Define graph in global scope

# Declare session, graph, and model in global scope
# with graph.as_default(), session.as_default():
#     set_session(session)
#
#     # Initialize all variables
#     init = tf.compat.v1.global_variables_initializer()
#     session.run(init)
#
#     # Load your model here
#     model = load_model('flowers.hd5')


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("Group_99/IMAGE/classify")
    else:
        print("Connection failed with code: %d." % rc)


def classify_cats(filename):
    global model, count  # Add model and count to the global variables
    global start_time
    print("Start classifying")
    # device = 'cpu'
    # 保存图片
    # filename = filename.split("\\")[-1]
    # current_work_dir = os.path.dirname(__file__)
    # save_path = os.path.join(current_work_dir + "./static/img", filename)  # 替换为你要保存图片的目录
    # with open(save_path, "wb") as f:
    #     # 将图像数据转换为Image对象
    #     image = Image.frombytes("RGB", (128, 128), data)
    #     image.save(save_path)

    # 获取结果
    results = model(filename, augment=False, visualize=False, device=0, conf=0.25)
    probs = results[0].probs  # get classification probabilities

    # Get the index of the class with the highest probability
    top1 = torch.argmax(probs).item()

    # Prepare output data
    output = {
        "filename": filename,
        "prediction": classes[top1],
        "score": float(probs[top1]),
        "index": top1
    }

    count += 1  # Increment count
    print("Done. This is image number {0}.".format(count))  # Print count
    if count == 1:
        end_time = time.time()  # Record the end time
        print("Time elapsed: {} seconds".format(end_time - start_time))  # Print the elapsed time
    # 把结果写入数据库
    return output  # return the result

    # global graph, session, model  # Add session and model to the global variables
    # print("Start classifying")
    # filename = filename.split("\\")[-1]
    # current_work_dir = os.path.dirname(__file__)
    # save_path = os.path.join(current_work_dir + "./static/img", filename)  # 替换为你要保存图片的目录
    # with open(save_path, "wb") as f:
    #     # 将图像数据转换为Image对象
    #     image = Image.frombytes("RGB", (128, 128), data)
    #     image.save(save_path)
    # data = data.reshape(1, 249, 249, 3)  # assumes that your images are 249x249 pixels with 3 channels
    # with graph.as_default(), session.as_default():
    #     set_session(session)
    #     predictions = model.predict(data)
    # win = np.argmax(predictions)
    # print("Done.")
    # with app.app_context():
    #     result = ClassificationModel(file_name=filename, prediction=classes[win], score=float(predictions[0][win]),index =int(win),time=datetime.now())
    #     db.session.add(result)
    #     db.session.commit()
    # return {"filename": filename, "prediction": classes[win], "score": float(predictions[0][win]), "index": int(win)}  # Convert win to int


def on_message(client, userdata, msg):
    global model, processed_files
    image_data = msg.payload  # 获取接收到的图片数据
    # 处理图片数据，例如保存为文件
    filename = "./static/cat/"+str(time.time())+".jpg"
    with open(filename, "wb") as file:
        file.write(image_data)
    print("图片保存成功")
    result = classify_cats(filename)
    print("Sending results:", result)
    client.publish("Group_99/IMAGE/predict", json.dumps(result), qos=1)  # publish with QoS level 1
    # processed_files += 1  # increment the counter when a file is processed
    # recv_dicts = json.loads(msg.payload)
    # for recv_dict in recv_dicts:
    #     image_data = recv_dict["file"]
    #     filename = recv_dict["filename"]
    #     with open("./static/cat"+filename, "wb") as file:
    #         file.write(image_data)
    #     result = classify_cats(filename)
    #     print("Sending results:", result)
    #     client.publish("Group_99/IMAGE/predict", json.dumps(result), qos=1)  # publish with QoS level 1
    #     processed_files += 1  # increment the counter when a file is processed
    # if processed_files == len(recv_dicts):  # if all files have been processed
    #     client.publish("Group_99/IMAGE/done", "All images have been classified.", qos=1)  # publish with QoS level 1
    # global graph, session, model  # Add session and model to the global variables
    # recv_dict = json.loads(msg.payload)
    # img_data = np.array(recv_dict["data"])
    # with graph.as_default(), session.as_default():
    #     set_session(session)
    #     result = classify_flower(recv_dict["filename"], img_data)
    # print("Sending results:", result)
    # client.publish("Group_99/IMAGE/predict", json.dumps(result))


def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    print("success setup\n")
    return client


@app.route('/')
def show_result():  # put application's code here
    setup("127.0.0.1")
    # data = db.session.query(ClassificationModel).order_by(ClassificationModel.id.desc()).all()
    # print(data[0].id)
    # start_time = datetime.now()
    # if data==[]:
    #     # print("no data")
    #     file_name = "test"
    #     prediction = "test"
    #     score = "test"
    #     index = "test"
    # else:
    #     file_name = data[0].file_name
    #     prediction = data[0].prediction
    #     score = data[0].score
    #     index = data[0].index
    # print(file_name)
    # return render_template("index.html", file_name=file_name, prediction=prediction, score=score,index=index)
    return "ok"


if __name__ == '__main__':
    app.run(debug=True)
