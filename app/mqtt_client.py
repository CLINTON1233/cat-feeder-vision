import paho.mqtt.client as mqtt

BROKER = "broker.emqx.io"
PORT = 1883

TOPIC_FEED = "cat/feeding"
TOPIC_STATUS = "cat/status"

# Ini sekarang dipakai sebagai STATUS ESP32 / MQTT
cooldown_remaining = "MQTT CONNECTED"

client = mqtt.Client(
    client_id="raspberry_cat_detector_001",
    protocol=mqtt.MQTTv311
)

def on_connect(client, userdata, flags, rc):
    global cooldown_remaining
    if rc == 0:
        print("🟢 MQTT CONNECTED (RASPBERRY)")
        cooldown_remaining = "MQTT CONNECTED"
        client.subscribe(TOPIC_STATUS)
        print(f"📥 SUBSCRIBED TO {TOPIC_STATUS}")
    else:
        cooldown_remaining = "MQTT FAILED"
        print("🔴 MQTT CONNECT FAILED", rc)

def on_message(client, userdata, msg):
    global cooldown_remaining
    topic = msg.topic
    payload = msg.payload.decode()

    print("📨 MQTT MESSAGE FROM ESP32")
    print("   Topic :", topic)
    print("   Data  :", payload)

    # Semua pesan dari ESP32 kita tampilkan sebagai status
    if topic == TOPIC_STATUS:
        cooldown_remaining = payload
        print(f"📡 ESP32 STATUS UPDATED → {cooldown_remaining}")

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
