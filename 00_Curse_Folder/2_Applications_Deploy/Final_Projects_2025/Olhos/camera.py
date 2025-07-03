import serial
import numpy as np
import matplotlib.pyplot as plt

# Configura a serial e aguarda o Arduino reiniciar
ser = serial.Serial('/dev/ttyACM0', 115200)
ser.flush()
import time
time.sleep(2)

# Parâmetros da imagem
width, height = 160, 120

# Inicializa o matplotlib
plt.ion()  # modo interativo
fig, ax = plt.subplots()
image = np.zeros((height, width), dtype=np.uint8)
im = ax.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title("Camera do Arduino")
plt.axis('off')

while True:
    # Lê um frame da serial (espera até receber tudo)
    data = ser.read(width * height)
    frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width))

    # Atualiza a imagem no matplotlib
    im.set_data(frame)
    plt.draw()
    plt.pause(0.001)  # pequeno atraso para renderizar