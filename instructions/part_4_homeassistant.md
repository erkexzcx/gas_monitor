# A working script

Create a new file `server.py` with the following contents, but before execute it investigate it and modify accordingly:
```python
#!/bin/python3

import argparse
import paho.mqtt.client as mqtt
import time

from functions import *

video_stream = 'http://172.18.17.68:8081/0/stream'
mqtt_endpoint = '172.18.17.10'
mqtt_queue = 'energy/gas'

cap = cv2.VideoCapture(video_stream)
def reconnect_stream():
    global cap
    cap.release()
    cap = cv2.VideoCapture(video_stream)

def on_connect(client, userdata, flags, rc):
    print("Connected! Result code: " + str(rc))

print("Connecting to MQTT...")
mqtt_client = mqtt.Client("gas_monitor_producer")
mqtt_client.on_connect = on_connect
mqtt_client.loop_start()
mqtt_client.connect(mqtt_endpoint)
while not mqtt_client.is_connected():
    time.sleep(1)

samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE,responses)

def get_ocr():
    ret, img = cap.read()
    if not ret:
        reconnect_stream()
        return get_ocr()

    img = process_image(img)

    # Find contours & sort
    img = img.astype(np.uint8)
    cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    sorted_cnts, _ = sort_contours(cnts)

    # Invert image
    img = invert_image(img)

    digits_string = ""
    for c in sorted_cnts:
        x,y,w,h = cv2.boundingRect(c)
        if is_digit_contour(w, h):
            roi = img[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            digits_string += str(int((results[0][0])))
    print("readed digits: " + digits_string)
    return digits_string

entries = []
while True:
    entries.append(get_ocr())
    if len(entries) <= 3:
        continue
    else:
        entries.pop(0)

    # Skip if last 3 values are not matching
    if not (entries[0] == entries[1] == entries[2]):
        continue

    # Skip if not exactly 7 digits parsed
    if len(entries[0]) != 7:
        continue

    currEntryFloat = float(entries[0])/100
    print("Sending to MQTT " + mqtt_queue + ":" + str(currEntryFloat))
    mqtt_client.publish(mqtt_queue, currEntryFloat)

    # Wait 5 sec, but keep consuming frames to avoid queue jam
    curr_time = time.time()
    while True:
        cap.read()
        elapsed = time.time() - curr_time
        if elapsed < 5:
            continue
        break
```

Now try to execute it and you will be able to see a similar message:
```
Connecting to MQTT...
Connected! Result code: 0
readed digits: 4748874
readed digits: 4748874
readed digits: 4748874
readed digits: 4748874
Sending to MQTT energy/gas:47488.74
```

Which is great and is exactly what we need!

# Setting up a script

Upload the following files to a script host:
* /root/generalsamples.data
* /root/generalresponses.data
* /root/functions.py
* /root/server.py

Then create a following SystemD script `/etc/systemd/system/gas.service` with below contents:
```
[Unit]
Description=oil service service
After=network-online.target

[Service]
WorkingDirectory=/root
ExecStart=/usr/bin/python3 -u /root/server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then reload SystemD and start/enable it:
```
systemctl daemon-reload
systemctl enable --now gas.service
```

And you can check the output of the script:
```
journalctl -f -u gas.service
```

# Setting up in Home Assistant

Add below configuration to home assistant `configuration.yml`:
```yaml
mqtt:
  broker: 172.18.17.10

sensor:
  - platform: mqtt
    state_topic: "energy/gas"
    device_class: gas
    name: "Gas usage"
    unit_of_measurement: "m³"
    state_class: total_increasing
```

and after Home Assistant restart, you will find a new sensor. :)

![](https://github.com/erkexzcx/gas_monitor/raw/main/instructions/images/pic10.png)
