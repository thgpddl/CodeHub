from paho.mqtt import client as mqtt_client
import random
import time


class Publish:
    def __init__(self,topic):

        self.broker = 'broker.emqx.io'
        self.port = 1883
        self.topic = topic
        self.client_id = f'python-mqtt-{random.randint(0, 1000)}'
        self.connect_mqtt()

    def connect_mqtt(self):
        """
        链接到Broker，建立链接会话的客户端
        :return:
        """
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)
            # Set Connecting Client ID
        self.client = mqtt_client.Client(self.client_id)
        self.client.on_connect = on_connect  # 客户端链接回调函数
        self.client.connect(self.broker, self.port)    # 连接到Broker，回调函数为on_connect


    def send(self,msg):
        """
        发送订阅消息到topic
        :param client: Publish客户端
        :return:
        """
        result = self.client.publish(self.topic, msg)
        status = result[0]  # result: [0, 1]
        if status == 0:
            print("send success")
        else:
            print("send failed")

if __name__=="__main__":
    publish=Publish("/test")
    while True:
        publish.send("this is a test!")