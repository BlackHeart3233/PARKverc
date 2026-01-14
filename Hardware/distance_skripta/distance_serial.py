import serial

PORT = 'COM3'
BAUDRATE = 115200

ser = serial.Serial(PORT, BAUDRATE, timeout=1)

print("listening for distance data...")

try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(f"razdalja: {line}")
except KeyboardInterrupt:
    print("\nprekinjam...")
finally:
    ser.close()
