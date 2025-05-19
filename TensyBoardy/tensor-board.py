from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'runs/', '--port', '6006'])
url = tb.launch()
print(f"TensorBoard running at {url}")

while True:
    try:
        pass
    except KeyboardInterrupt:
        print("Stopping TensorBoard...")
        tb.shutdown()
        break
