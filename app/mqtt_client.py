import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883

TOPIC_FEED = "cat/feeding"
TOPIC_STATUS = "cat/status"

client = mqtt.Client(client_id="raspberry_cat_detector_001", protocol=mqtt.MQTTv311)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(" MQTT CONNECTED SUCCESS")
    else:
        print(" MQTT CONNECT FAILED", rc)

def connect():
    client.on_connect = on_connect
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    client.publish(TOPIC_STATUS, "RASPBERRY ONLINE")
    print(" MQTT STARTED LOOP")

def send_feed(source):
    print(f"📡 SEND MQTT → {source}")
    client.publish(TOPIC_FEED, source)
    client.publish(TOPIC_STATUS, f"{source} DETECTED → SERVO MOVE")
