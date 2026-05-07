import os


def BuildOutputPath(inEntryScriptFile : str, inDatasetName : str = None) -> str:
    """统一拼 `output/<脚本basename>/[<数据集名>/]` 的路径并 ensure 目录存在。

    用法（在入口脚本顶部）：
        from KongMing.Utils.OutputPath import BuildOutputPath
        OutputPath          = BuildOutputPath(__file__)                      # output/005_DDPM
        ModelRootFolderPath = BuildOutputPath(__file__, "FashionMNIST")      # output/005_DDPM/FashionMNIST

    设计取舍：让入口脚本传 __file__ 而不是默认从 sys.argv[0] 推导——
    后者在 jupyter / pytest / 嵌套调用下会指错地方。
    """
    BaseName = os.path.splitext(os.path.basename(inEntryScriptFile))[0]
    if inDatasetName:
        Path = os.path.join("output", BaseName, inDatasetName)
    else:
        Path = os.path.join("output", BaseName)
    os.makedirs(Path, exist_ok=True)
    return Path
