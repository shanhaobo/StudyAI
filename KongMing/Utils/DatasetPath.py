import os


def ResolveDatasetPath() -> str:
    """解析数据集根目录，按优先级：

    1. 环境变量 STUDYAI_DATA_DIR
    2. D:/AI/Datasets
    3. D:/__DevAI__/Datasets
    4. 仓库本地 ./data/Datasets （torchvision 可自动下载到此处）

    始终返回一个目录路径（必要时会创建），不会返回 None。
    """
    EnvPath = os.environ.get("STUDYAI_DATA_DIR")
    if EnvPath:
        Resolved = EnvPath if EnvPath.endswith("Datasets") else os.path.join(EnvPath, "Datasets")
    elif os.path.exists("D:/AI/"):
        Resolved = "D:/AI/Datasets"
    elif os.path.exists("D:/__DevAI__/"):
        Resolved = "D:/__DevAI__/Datasets"
    else:
        Resolved = os.path.join("data", "Datasets")

    os.makedirs(Resolved, exist_ok=True)
    return Resolved
