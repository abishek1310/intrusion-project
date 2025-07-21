import serial
import time
ser = serial.Serial('COM10', 9600)
time.sleep(2)
ser.write('f \n'.encode())
time.sleep(3)
data = ser.readline().decode().strip()
print(data)
ser.close()