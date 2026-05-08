r"""一次性下载公开 Anime Face 数据集到 DatasetPath/cartoon_faces/。

数据来源：HuggingFace `huggan/anime-faces`（~21k 张 64×64 动漫头像，公开数据集，零凭据）。
使用 `datasets` 库直接 HTTP 拉取，无需 Kaggle / 任何 token。

用法（在仓库根下执行）：
    python tools/download_cartoon_faces.py

成功后目录结构：
    <DatasetPath>/cartoon_faces/images/000000.png
    <DatasetPath>/cartoon_faces/images/000001.png
    ...
torchvision.datasets.ImageFolder 会把 "images" 当作单一 class 使用，
正好满足 002/004/005 入口脚本的要求。
"""

import os
import sys

# 允许从仓库任意位置直接 `python tools/download_cartoon_faces.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KongMing.Utils.DatasetPath import ResolveDatasetPath


###################################################################################################

HFDatasetName = "huggan/anime-faces"
TargetSubdir  = "cartoon_faces"
ImagesSubdir  = "images"


###################################################################################################

def CountImages(inDir : str) -> int:
    if not os.path.isdir(inDir):
        return 0
    Cnt = 0
    for Root, _Dirs, Files in os.walk(inDir):
        for F in Files:
            if F.lower().endswith((".png", ".jpg", ".jpeg")):
                Cnt += 1
    return Cnt


def CheckDatasets() -> bool:
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        print("[ERROR] HuggingFace `datasets` 未安装。请先：")
        print("    pip install datasets pillow")
        return False


###################################################################################################

def Main() -> int:
    DatasetPath = ResolveDatasetPath()
    TargetDir   = os.path.join(DatasetPath, TargetSubdir)
    ImagesDir   = os.path.join(TargetDir, ImagesSubdir)

    print("[Resolve] DatasetPath = {}".format(DatasetPath))
    print("[Resolve] Target dir  = {}".format(TargetDir))

    Existing = CountImages(TargetDir)
    if Existing > 0:
        print("[Skip] 目标目录已有 {} 张图片，无需重新下载。".format(Existing))
        print("       想强制重下：先删掉 {} 再跑本脚本。".format(TargetDir))
        return 0

    if not CheckDatasets():
        return 1

    print("[Download] HuggingFace dataset '{}' (split=train)...".format(HFDatasetName))
    from datasets import load_dataset
    DS = load_dataset(HFDatasetName, split="train")
    print("[Download] {} 条样本就位".format(len(DS)))

    os.makedirs(ImagesDir, exist_ok=True)
    print("[Save] -> {}".format(ImagesDir))

    Saved = 0
    for I, Item in enumerate(DS):
        # 数据集字段一般是 'image'，类型 PIL.Image
        Img = Item.get("image") or Item.get("img") or Item.get("picture")
        if Img is None:
            # 兜底：取第一个 PIL.Image 字段
            from PIL import Image
            for V in Item.values():
                if isinstance(V, Image.Image):
                    Img = V
                    break
        if Img is None:
            continue

        OutPath = os.path.join(ImagesDir, "{:06d}.png".format(I))
        Img.save(OutPath, "PNG")
        Saved += 1
        if Saved % 1000 == 0:
            print("  ... saved {} / {}".format(Saved, len(DS)))

    Final = CountImages(TargetDir)
    print("[Done] 共 {} 张图片就位。".format(Final))

    if Final == 0:
        print("[WARN] 目标目录里没找到图片，HF 数据集字段可能变了。手动检查 {}。".format(TargetDir))
        return 1

    print("[Layout] {} 下顶层条目：".format(TargetDir))
    for Entry in sorted(os.listdir(TargetDir))[:10]:
        FullPath = os.path.join(TargetDir, Entry)
        Tag = "DIR " if os.path.isdir(FullPath) else "FILE"
        print("  {} {}".format(Tag, Entry))
    return 0


if __name__ == "__main__":
    sys.exit(Main())
