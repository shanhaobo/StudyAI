r"""一次性下载 Anime Face Dataset (splcher/animefacedataset) 到 DatasetPath/cartoon_faces/。

用法（在仓库根下执行）：
    python tools/download_cartoon_faces.py

前置条件（脚本会逐项检查并给出修复指引）：
1. 安装 kagglehub：     pip install kagglehub
2. 配置 Kaggle 凭据：    在 https://www.kaggle.com → Account → "Create New API Token"
                        下载 kaggle.json，放到：
                            Windows: %USERPROFILE%\.kaggle\kaggle.json
                            Linux/Mac: ~/.kaggle/kaggle.json
                        或通过环境变量：
                            $env:KAGGLE_USERNAME / $env:KAGGLE_KEY (PowerShell)
                            export KAGGLE_USERNAME=... / KAGGLE_KEY=... (bash)

成功后目录结构：
    <DatasetPath>/cartoon_faces/images/*.png    （≈ 63k 张 64×64 头像）
torchvision.datasets.ImageFolder 会把 "images" 当作单一 class 使用，
正好满足 002/004/005 入口脚本的要求。
"""

import os
import shutil
import sys

# 允许从仓库任意位置直接 `python tools/download_cartoon_faces.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KongMing.Utils.DatasetPath import ResolveDatasetPath


###################################################################################################

KaggleDatasetSlug = "splcher/animefacedataset"
TargetSubdir      = "cartoon_faces"


###################################################################################################

def CountImages(inDir : str) -> int:
    """递归统计 .png/.jpg 数量；用来判断目录是否已经被填充过。"""
    if not os.path.isdir(inDir):
        return 0
    Cnt = 0
    for Root, _Dirs, Files in os.walk(inDir):
        for F in Files:
            if F.lower().endswith((".png", ".jpg", ".jpeg")):
                Cnt += 1
    return Cnt


def CheckKaggleHub() -> bool:
    try:
        import kagglehub  # noqa: F401
        return True
    except ImportError:
        print("[ERROR] kagglehub 未安装。请先：")
        print("    pip install kagglehub")
        return False


def CheckKaggleCredentials() -> bool:
    """kagglehub 与 kaggle CLI 共用同一份凭据：~/.kaggle/kaggle.json 或 KAGGLE_USERNAME/KAGGLE_KEY 环境变量。"""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    KaggleJsonPath = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if os.path.exists(KaggleJsonPath):
        return True
    print("[ERROR] 未找到 Kaggle 凭据。请二选一：")
    print("  A) 在 https://www.kaggle.com → Account → Create New API Token，")
    print("     把下载的 kaggle.json 放到：{}".format(KaggleJsonPath))
    print("  B) 设环境变量 KAGGLE_USERNAME 和 KAGGLE_KEY")
    return False


def MergeIntoTarget(inSrcDir : str, inTargetDir : str) -> None:
    """把 kagglehub 下载到 cache 里的内容搬到 inTargetDir。
    保留原 zip 内的子目录结构（这个数据集的根下有 images/ 子目录）。
    已存在的同名文件会被覆盖。
    """
    os.makedirs(inTargetDir, exist_ok=True)
    for Entry in os.listdir(inSrcDir):
        SrcPath = os.path.join(inSrcDir, Entry)
        DstPath = os.path.join(inTargetDir, Entry)
        if os.path.isdir(SrcPath):
            # 用 copytree + dirs_exist_ok（Py3.8+），避免删除目标里已有的别的子目录
            shutil.copytree(SrcPath, DstPath, dirs_exist_ok=True)
        else:
            shutil.copy2(SrcPath, DstPath)


###################################################################################################

def Main() -> int:
    DatasetPath = ResolveDatasetPath()
    TargetDir   = os.path.join(DatasetPath, TargetSubdir)

    print("[Resolve] DatasetPath = {}".format(DatasetPath))
    print("[Resolve] Target dir  = {}".format(TargetDir))

    Existing = CountImages(TargetDir)
    if Existing > 0:
        print("[Skip] 目标目录已有 {} 张图片，无需重新下载。".format(Existing))
        print("       想强制重下：先删掉 {} 再跑本脚本。".format(TargetDir))
        return 0

    if not CheckKaggleHub():
        return 1
    if not CheckKaggleCredentials():
        return 1

    print("[Download] kagglehub.dataset_download('{}')...".format(KaggleDatasetSlug))
    import kagglehub
    CachePath = kagglehub.dataset_download(KaggleDatasetSlug)
    print("[Download] cached at: {}".format(CachePath))

    print("[Move] -> {}".format(TargetDir))
    MergeIntoTarget(CachePath, TargetDir)

    Final = CountImages(TargetDir)
    print("[Done] 共 {} 张图片就位。".format(Final))

    if Final == 0:
        print("[WARN] 目标目录里没找到图片，可能数据集结构变了。手动检查 {} 的内容。".format(TargetDir))
        return 1

    # 列出顶层结构方便用户确认 ImageFolder 能识别
    print("[Layout] {} 下顶层条目：".format(TargetDir))
    for Entry in sorted(os.listdir(TargetDir))[:10]:
        FullPath = os.path.join(TargetDir, Entry)
        Tag = "DIR " if os.path.isdir(FullPath) else "FILE"
        print("  {} {}".format(Tag, Entry))
    return 0


if __name__ == "__main__":
    sys.exit(Main())
