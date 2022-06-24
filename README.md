# CS:GO Yolov5 Aimbot

An aimbot that uses Yolov5 and PyTorch to play CS:GO. It is able to identify the characters from both teams and their heads. It uses pyautogui to handle the aim movement and mouse click.

## How To Use

Currently, only Linux is supported.

### In-game configuration
You have to turn OFF the Raw Input option in the game. Otherwise, the mouse signal goes straight to the game engine without passing through the operating system and pyautogui can't handle it properly.

This setting is under the Mouse & Keyboard tab:
![Alt-test](/img/raw-input.jpg)

### Installing
To clone and install this application you will need [Git](https://git-scm.com) and [Python>=3.9.0](https://www.python.org/downloads/).

```
# Clone this repository
git clone https://github.com/daniabib/csgo-aimbot.git

# Go into the repository
cd csgo-aimbot

# Install required libraries
pip install -r requirements.txt
```

### Running
I suggest that the game is already open in the background before launching the model. 


To run the app just execute the main script:

```
python aim-bot.py
```

The model will be running in the background. As soon as it detects a CS:GO character it will activate the pyautogui engine to move the mouse and shoot.


<!-- ## How it works? -->


<!-- ## Things I'm still implementing 
- Record screen -->


<!-- ## Support

<a href="https://www.buymeacoffee.com/danielabib?" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/purple_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a> -->


## Disclaimer
The intent of this repository is esclusively educational. It has no intention of being a way of cheating in the game. I advise that you use it only on your own server. If you use it in a Valve server it will probably detect suspicious activities and your account can be banned. 