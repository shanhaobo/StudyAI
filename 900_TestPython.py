import os

# 初始化消息列表
messages = []

while True:
    # 等待用户输入命令
    command = input("> ")

    # 执行命令
    if command == "exit":
        break

    # 进行主循环输出
    message = f"Output from main loop: {os.getpid()}"
    messages.append(message)

    # 如果消息数量超过4，则删除最老的一条消息
    if len(messages) > 4:
        del messages[0]

    # 输出所有消息
    os.system("cls") # 清空控制台输出
    print(command)
    print("=" * 20)
    for msg in messages:
        print(msg)
