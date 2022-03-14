# Prerequisites

## Home Assistant

You will obviously need Home Assistant up & running.

## Mosquito MQTT

You must have MQTT broker up & running and this is not part of this tutorial.

## Video stream

For now focus only on the camera - it can be camera module of RPI or even ESP Camera. The choice is yours.

Example of how video stream looks like in my case:

![](https://github.com/erkexzcx/gas_monitor/raw/main/instructions/images/pic3.png)

Note the details:
* Digits are readable.
* Camera is not aligned - we can rotate image later.
* Led near the camera - ensures digits are visible even in complete darkness.
* Digits are not blurred (a slight blur is totally fine, as long as digits are readable).
* Camera is not pointing directly to the glass - this helps to avoid light reflections.
* Camera CANNOT be touched/moved/covered once it is installed, so plan your mount accordingly.

In case you want to see a bigger picture of how all of this is mounted (most importantly - *it works*):

![](https://github.com/erkexzcx/gas_monitor/raw/main/instructions/images/pic1.png)

![](https://github.com/erkexzcx/gas_monitor/raw/main/instructions/images/pic2.png)

## Device that hosts Python script

It can be your RPI that has camera attached or it can be the same host where your HA is hosted. The choice is yours.

**Note**: You will be using OpenCV on your computer (for testing/working purposes) and python script host (e.g. RPI) and the OpenCV versions **must match** on both devices. For example, Arch Linux (on PC) and Raspberry Pi OS had different OpenCV versions, so [quick Google search](https://github.com/prepkg/opencv-raspberrypi) solved this for me.

# Confirm that camera stream works

Use VLC to connect to the video stream from your PC. I can use this link:
```
http://172.18.17.68:8081/
```

# Add camera stream to Home Assistant

Add your video stream as a camera to Home Assistant. This is very helpful to compare extracted gas value to actual value from the camera.

# Install dependencies on your PC and host device

The following dependencies needs to be installed from your package manager:
```
python3
python3-pip
opencv
```

The following needs to be installed using `pip` package manager:
```
opencv-python
numpy
paho-mqtt
```

Only OpenCV version must match between both systems.
