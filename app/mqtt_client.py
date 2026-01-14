import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883

TOPIC_FEED = "cat/feeding"
TOPIC_STATUS = "cat/status"

client = mqtt.Client(client_id="raspberry_cat_detector_001")

def connect():
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    client.publish(TOPIC_STATUS, "RASPBERRY ONLINE")
    print(" MQTT CONNECTED TO HIVEMQ")

def send_feed(source):
    client.publish(TOPIC_FEED, source)
    client.publish(TOPIC_STATUS, f"{source} DETECTED → SERVO MOVE")
