import serial
import time
import threading

PORT = "COM9"
BAUD = 115200

ser = serial.Serial(
    port=PORT,
    baudrate=BAUD,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)

time.sleep(2)  # USB CDC settle

def send_rotate(deg):
    cmd = f"ROTATE:{deg}\n"
    ser.write(cmd.encode("ascii"))
    print("Sent:", cmd.strip())


def rx_thread():
    while True:
        try:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue

            print("RX:", line)

            # pričakovan format:
            # ROT:xx.xx DIST:yy.yy
            parts = line.split()

            rot = None
            dist = None

            # Ločimo in zajamemo oba podatka ločeno
            for p in parts:
                if p.startswith("ROT:"):
                    rot = float(p.split(":")[1])  # le ROT
                elif p.startswith("DIST:"):
                    dist = float(p.split(":")[1])  # le DIST

            if rot is not None and dist is not None:
                print(f"  -> Rotary = {rot:.2f} deg | Distance = {dist:.2f} cm")

        except Exception as e:
            print("RX error:", e)
            break


# RX thread
threading.Thread(target=rx_thread, daemon=True).start()

# test ukazi
send_rotate(540)
time.sleep(5)
send_rotate(0)
time.sleep(5)
send_rotate(-540)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
    ser.close()
