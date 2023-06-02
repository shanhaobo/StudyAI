import os
import threading
import time
import queue

# 初始化消息列表
messages = []
# 控制循环的变量
running = True

# 创建一个队列用于线程之间的通信
command_queue = queue.Queue()

# 消息处理线程的函数
def message_loop():
    global running
    while running:
        # 进行主循环输出
        message = f"Output from main loop: {os.getpid()}"
        messages.append(message)

        # 如果消息数量超过4，则删除最老的一条消息
        if len(messages) > 4:
            del messages[0]

        # 输出所有消息
        if command_queue.empty():  # 只有当命令队列为空的时候才清屏和输出消息
            os.system("cls")  # 清空控制台输出
            print("=" * 20)
            for msg in messages:
                print(msg)

        # 等待一会儿，否则输出会太快
        time.sleep(1)

# 用户输入处理线程的函数
def input_loop():
    global running
    while running:
        # 等待用户输入命令
        command = input("> ")
        # 把命令放入队列
        command_queue.put(command)

        # 执行命令
        if command == "exit":
            running = False
            break
        elif command == "stop":
            running = False
        elif command == "start":
            if not thread.is_alive():
                running = True
                thread = threading.Thread(target=message_loop)
                thread.start()

# 创建并启动消息处理线程
thread = threading.Thread(target=message_loop)
thread.start()

# 创建并启动用户输入处理线程
input_thread = threading.Thread(target=input_loop)
input_thread.start()
