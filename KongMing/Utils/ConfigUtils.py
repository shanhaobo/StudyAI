"""轻量 Config 助手：让入口脚本顶部那批超参以 dataclass 形式存在，并用 CLI --K=V 覆盖。

不引入 yaml / hydra / omegaconf 等配置框架，遵循 KongMing "无配置文件" 哲学。
匹配是 case-insensitive（与 Executor 的 KVArgsForML 行为一致）。
"""

import sys
from dataclasses import is_dataclass, fields
from typing import Any, List


def ParseSysArgvKV() -> dict :
    """从 sys.argv 抽出所有 --KEY=VAL 对。key casefold，value 保留原文（路径/tag 大小写不能丢）。
    用于 Factory 构造之前的早期参数解析——Executor 实例此时还没造出来。"""
    Out = {}
    for raw in sys.argv:
        if not isinstance(raw, str):
            continue
        if not raw.startswith("--") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        Out[key[2:].casefold()] = value
    return Out


def ApplyConfigFromKV(inConfig, inKVArgs : dict = None) -> List[str] :
    """从 inKVArgs 抽出与 inConfig 字段同名的项，按字段当前值的类型 cast 后回写。

    Args:
        inConfig: dataclass 实例（或任意带属性的对象）
        inKVArgs: dict[str, str]；为 None 时自动 ParseSysArgvKV()

    Returns:
        list[str]：成功覆盖的字段名（保持原始大小写）。
                   入口脚本可用它判断"用户是否显式给了某个值"（用于 dataset → 默认值的二级回填）。
    """
    if inConfig is None:
        return []
    if inKVArgs is None:
        inKVArgs = ParseSysArgvKV()
    if not inKVArgs:
        return []

    if is_dataclass(inConfig):
        Names = [f.name for f in fields(inConfig)]
    else:
        # 退化：所有非下划线开头、非 callable 的属性
        Names = [n for n in dir(inConfig)
                 if not n.startswith("_") and not callable(getattr(inConfig, n, None))]

    NameMap = {n.casefold(): n for n in Names}
    Overridden : List[str] = []

    for K, V in inKVArgs.items():
        if not isinstance(K, str):
            continue
        Name = NameMap.get(K.casefold())
        if Name is None:
            continue
        Current = getattr(inConfig, Name)
        try:
            Casted = _CastByCurrent(V, Current)
        except Exception as e:
            print("[Config] cannot cast {}={!r} to {}: {}".format(
                Name, V, type(Current).__name__, e))
            continue
        setattr(inConfig, Name, Casted)
        Overridden.append(Name)
    return Overridden


def _CastByCurrent(inValue : Any, inCurrent : Any) :
    """按 inCurrent 当前值的类型把 inValue 转过去。
    支持 bool/int/float/str/tuple/list；其它类型走 type(inCurrent)(inValue) 兜底。"""
    if type(inValue) is type(inCurrent):
        return inValue

    T = type(inCurrent)

    if T is bool:
        if isinstance(inValue, str):
            v = inValue.strip().lower()
            return v in ("1", "true", "yes", "y", "on")
        return bool(inValue)

    if T is tuple or T is list:
        # "0.9,0.999" → (0.9, 0.999)；元素类型由 inCurrent 第一个元素决定
        if isinstance(inValue, str):
            Parts = [p.strip() for p in inValue.split(",")]
        else:
            Parts = list(inValue)
        if len(inCurrent) > 0:
            ElemT = type(inCurrent[0])
            Casted = [ElemT(p) for p in Parts]
        else:
            Casted = Parts
        return T(Casted)

    return T(inValue)
