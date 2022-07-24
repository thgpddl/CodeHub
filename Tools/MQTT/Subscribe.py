from paho.mqtt import client as mqtt_client
import random

class Subscribe:
    def __init__(self,topic):
        self.broker = 'broker.emqx.io'
        self.port = 1883
        self.topic = topic
        self.client_id = f'python-mqtt-{random.randint(0, 1000)}'
        self._connect_mqtt()

    def _connect_mqtt(self):
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
        self.client.connect(self.broker, self.port)  # 连接到Broker，回调函数为on_connect

    def receive(self,Callback,is_loop_forever=True):
        """

        :param Callback: 接受回调函数
        :param is_loop_forever:
        :return:
        """
        self.client.subscribe(self.topic)  # 订阅topic
        self.client.on_message = Callback
        if is_loop_forever:
            self.client.loop_forever()

if __name__=="__main__":
    subscribe=Subscribe("/test")
    def on_message(client, userdata, msg):
        """
        接受订阅消息的回调函数
        :param client:
        :param userdata:
        :param msg: 接受的消息内容
        :return:
        """
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
    subscribe.receive(Callback=on_message)
    print("continue ")  # 无法运行
