import sys
sys.path.append("/Users/jangjiyun/Desktop/JY/code")
from oculus_reader.oculus_reader.reader import OculusReader
import time
import threading


def main():
    oculus_reader = OculusReader()

    while True:
        time.sleep(0.3)
        print(oculus_reader.get_transformations_and_buttons())

if __name__ == '__main__':
    main()