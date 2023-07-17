import paho.mqtt.client as mqtt
import numpy as np
import json
from ultralytics import YOLO
import torch
# Declare model in global scope
global model
import time
start_time = time.time()  # Record the start time
# Load your model here

# model = YOLO("E:/download/ultralytics-main/yolov8m-cls.pt")
model = YOLO("E:/data/processed/best-nano.pt")  # replace this with your model's path

classes = ["Persian", "Ragdoll", "Scottish_Fold", "Singapura", "Sphynx"]
global count
count = 0


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("Group_99/IMAGE/classify", qos=1)  # subscribe with QoS level 1
    else:
        print("Connection failed with code: %d." % rc)


def classify_flower(filename):
    global model, count  # Add model and count to the global variables
    print("Start classifying")
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
    return output  # return the result


processed_files = 0  # add this line to track processed files


def on_message(client, userdata, msg):
    global model, processed_files
    recv_dicts = json.loads(msg.payload)
    for recv_dict in recv_dicts:
        filename = recv_dict["filename"]
        result = classify_flower(filename)
        print("Sending results:", result)
        client.publish("Group_99/IMAGE/predict", json.dumps(result), qos=1)  # publish with QoS level 1
        processed_files += 1  # increment the counter when a file is processed
    if processed_files == len(recv_dicts):  # if all files have been processed
        client.publish("Group_99/IMAGE/done", "All images have been classified.", qos=1)  # publish with QoS level 1


def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client


def main():
    setup("127.0.0.1")
    while True:
        pass


if __name__ == "__main__":
    main()
