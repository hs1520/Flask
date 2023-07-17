import paho.mqtt.client as mqtt
import numpy as np
from PIL import Image
import json
import os
import time

response_received = False


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected.")
        client.subscribe("Group_99/IMAGE/predict", qos=1)
        client.subscribe("Group_99/IMAGE/done", qos=1)  # subscribe to the 'done' topic with QoS level 1
    else:
        print("Failed to connect. Error code: %d." % rc)


def on_message(client, userdata, msg):
    global response_received
    print("Received message from server.")
    if msg.topic == "Group_99/IMAGE/predict":
        resp_dict = json.loads(msg.payload)
        print("Filename: %s, Prediction: %s, Score: %3.4f" % (resp_dict["filename"], resp_dict["prediction"], resp_dict["score"]))
    elif msg.topic == "Group_99/IMAGE/done":
        print(msg.payload)
        response_received = True


def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client


def load_image(filename):
    img = Image.open(filename)
    img = img.resize((128, 128))
    imgarray = np.array(img) / 255.0
    final = np.expand_dims(imgarray, axis=0)
    return final


def send_image(client, filenames):
    send_dicts = [{"filename": filename} for filename in filenames]
    client.publish("Group_99/IMAGE/classify", json.dumps(send_dicts), qos=1)  # publish with QoS level 1


def main():
    global response_received
    client = setup("127.0.0.1")
    print("sending data.")
    directory = r"./static/cat"
    filenames = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith((".jpg", ".png"))]
    send_image(client, filenames)
    print(f"Sent {len(filenames)} images")

    while True:
        try:
            if client.loop() == mqtt.MQTT_ERR_NO_CONN:
                break
        except AttributeError:
            print("Connection lost. Attempting to reconnect.")
            time.sleep(5)  # Wait a bit before attempting to reconnect
            client.reconnect()
        time.sleep(0.3)

    print("Done. All images have been classified.")


if __name__ == "__main__":
    main()
