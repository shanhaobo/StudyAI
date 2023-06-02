import threading

def main_loop():
    # 在这里编写主循环的逻辑
    print("正在执行主循环...")

def input_thread(signal):
    while True:
        user_input = input("请输入：")
        if user_input == "stop":
            signal.set()  # 发送信号，告诉主循环停止
            break

if __name__ == '__main__':
    stop_signal = threading.Event()
    keyboard_thread = threading.Thread(target=input_thread, args=(stop_signal,))
    keyboard_thread.start()

    while not stop_signal.is_set():
        main_loop()

    print("停止主循环")
