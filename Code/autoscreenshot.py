import pyautogui
import keyboard
import time

width, height = pyautogui.size()
i = 1

pyautogui.moveTo(390, 240)
pyautogui.click(button='left')
# pyautogui.click(button='left')
pyautogui.moveTo(300, 237)
with open('train.txt', 'r', encoding='utf-8') as file:
    for line in file:
        time.sleep(1.0)
        pyautogui.screenshot(imageFilename=line[0:-1], region=(375, 237, 1489, 1117))
        print(f"Скриншот {i} сделан.")
        i += 1
        time.sleep(1.0)
        pyautogui.press('f')
        
        
