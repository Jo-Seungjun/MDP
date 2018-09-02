'''
https://gpiozero.readthedocs.io/en/stable/recipes.html
http://raspberry.io/projects/view/reading-and-writing-from-gpio-ports-from-python/
https://pimylifeup.com/raspberry-pi-gpio/
'''

from gpiozero import Motor
from time import sleep

motor = Motor(forward=4, backward=14)

def isopen():
    return True

def open():
    motor.forward()
    pass

def close():
    motor.backward()
    pass

if __name__ == "__main__":
    print('run as it self')
    while True:
        open()
        sleep(1)
        close()
        sleep(1)
