import os
import re

#match  anystr_digitals_anystr
def FindFileWithMaxNum(inFileList, inPrefix, inSuffix, inExtension=None):
    MaxNum = -1
    ResultFile = None

    # 添加了转义字符 '\' 到 prefix 和 suffix
    # 因为 '.' 在正则表达式中有特殊含义
    PartialPrefix = re.escape(inPrefix).replace('\\*', '.*')
    PartialSuffix = re.escape(inSuffix).replace('\\*', '.*')

    if inExtension:
        inExtension = re.escape(inExtension)
        Pattern = f'^{PartialPrefix}.*?(\d+){PartialSuffix}\.{inExtension}$'
    else:
        Pattern = f'^{PartialPrefix}.*?(\d+){PartialSuffix}\..*$'

    for File in inFileList:
        Match = re.match(Pattern, File)
        if Match:
            Num = int(Match.group(1))  # 这是括号中的数字
            if Num > MaxNum:
                MaxNum = Num
                ResultFile = File

    return ResultFile, MaxNum

def FindFileWithMaxNumByFolderPath(inFolderPath, inPrefix, inSuffix, inExtension=None):
    FlieName, MaxNum = FindFileWithMaxNum(os.listdir(inFolderPath), inPrefix, inSuffix, inExtension)
    return os.path.join(inFolderPath, FlieName), MaxNum

if __name__ == "__main__" :
    folder_path = "trained_models\CFGAN_202305291532"  # 替换为你的文件夹路径
    prefix = 'Discr*'  # 替换为你的前缀
    suffix = ''  # 替换为你的后缀
    max_file, max_num = FindFileWithMaxNumByFolderPath(folder_path, prefix, suffix)
    print(f'File with the largest number: {max_file}, Largest number: {max_num}')
