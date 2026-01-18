import paho.mqtt.client as mqtt

BROKER = "broker.emqx.io"
PORT = 1883

TOPIC_FEED = "cat/feeding"
TOPIC_STATUS = "cat/status"

client = mqtt.Client(
    client_id="raspberry_cat_detector_001",
    protocol=mqtt.MQTTv311
)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("🟢 MQTT CONNECTED (RASPBERRY)")
        client.subscribe(TOPIC_STATUS)
        print(f"📥 SUBSCRIBED TO {TOPIC_STATUS}")
    else:
        print("🔴 MQTT CONNECT FAILED", rc)

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode()

    print("📨 MQTT MESSAGE FROM ESP32")
    print("   Topic :", topic)
    print("   Data  :", payload)

    if topic == TOPIC_STATUS:
        print(f"🦾 ESP32 STATUS → {payload}")

def connect():
    client.on_connect = on_connect
    client.on_message = on_message

    print("🔌 Connecting MQTT...")
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    client.publish(TOPIC_STATUS, "RASPBERRY ONLINE")
    print("🚀 MQTT LOOP STARTED, RASPBERRY ONLINE")

def send_feed(source):
    print(f"📡 SEND MQTT TO ESP32 → {source}")
    client.publish(TOPIC_FEED, source)
    client.publish(TOPIC_STATUS, f"{source} DETECTED BY CAMERA")