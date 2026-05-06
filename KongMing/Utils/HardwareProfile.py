import os
import torch


def DetectHardwareProfile(inMemoryFactor: float = 1.0) -> dict:
    """检测当前硬件并返回 DataLoader 推荐参数。

    返回 dict 字段：
        Device       : "cuda" / "mps" / "cpu"
        DeviceName   : 设备显示名
        VRAMGiB      : 显存/统一内存 GiB（CPU 返回 0.0）
        BatchSize    : DataLoader batch_size 推荐值
        NumWorkers   : DataLoader num_workers 推荐值
        PinMemory    : DataLoader pin_memory 推荐值

    inMemoryFactor:
        模型相对显存占用倍数。基准 = DDPM/DCGAN 这类 ~5M 参数 / 64x64 图。
        VGG16 (138M / 224x224) 这类大模型应传 4.0~8.0 让 BatchSize 等比缩小。

    可通过环境变量强制覆盖：
        STUDYAI_BATCH_SIZE   : 整数，强制 BatchSize
        STUDYAI_NUM_WORKERS  : 整数，强制 NumWorkers
    """
    Profile = {
        "Device":       "cpu",
        "DeviceName":   "CPU",
        "VRAMGiB":      0.0,
        "BatchSize":    16,
        "NumWorkers":   0,
        "PinMemory":    False,
    }

    if torch.cuda.is_available():
        Profile["Device"]       = "cuda"
        Profile["DeviceName"]   = torch.cuda.get_device_name(0)
        VRAMBytes               = torch.cuda.get_device_properties(0).total_memory
        Profile["VRAMGiB"]      = VRAMBytes / (1024 ** 3)
        Profile["PinMemory"]    = True
        if Profile["VRAMGiB"] >= 15.0:    # 16GB 实际可用约 15.x GiB
            Profile["BatchSize"]    = 256
            Profile["NumWorkers"]   = 4
        elif Profile["VRAMGiB"] >= 10.0:
            Profile["BatchSize"]    = 128
            Profile["NumWorkers"]   = 4
        elif Profile["VRAMGiB"] >= 6.0:
            Profile["BatchSize"]    = 64
            Profile["NumWorkers"]   = 2
        else:
            Profile["BatchSize"]    = 32
            Profile["NumWorkers"]   = 2
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        Profile["Device"]       = "mps"
        Profile["DeviceName"]   = "Apple Silicon (MPS)"
        # 统一内存 + MPS 后端某些 op 仍 fallback 到 CPU，先给中等 batch
        Profile["BatchSize"]    = 64
        Profile["NumWorkers"]   = 0     # MPS + workers>0 在 macOS 上易出 bug
        Profile["PinMemory"]    = False

    # 按模型显存占用倍数缩放
    if inMemoryFactor > 1.0:
        Scaled = max(1, int(Profile["BatchSize"] / inMemoryFactor))
        Profile["BatchSize"] = Scaled

    # 环境变量覆盖（最后一步，不受 inMemoryFactor 影响）
    EnvBatchSize = os.environ.get("STUDYAI_BATCH_SIZE")
    if EnvBatchSize:
        Profile["BatchSize"] = int(EnvBatchSize)
    EnvNumWorkers = os.environ.get("STUDYAI_NUM_WORKERS")
    if EnvNumWorkers:
        Profile["NumWorkers"] = int(EnvNumWorkers)

    return Profile


def FormatProfileLine(inProfile: dict) -> str:
    """生成单行人类可读的 profile 摘要，供入口脚本 [Run] 行使用"""
    return "{} ({}, {:.1f} GiB) | BatchSize={} | NumWorkers={} | PinMemory={}".format(
        inProfile["DeviceName"],
        inProfile["Device"],
        inProfile["VRAMGiB"],
        inProfile["BatchSize"],
        inProfile["NumWorkers"],
        inProfile["PinMemory"],
    )
