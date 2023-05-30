import os

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def print_tree(self, padding=''):
        result = padding + self.name + '\n'
        padding += ' '
        for child in self.children:
            if isinstance(child, TreeNode):
                result += child.print_tree(padding + '|--')
            else:
                result += padding + '|--' + child + '\n'
        return result

def build_tree(dir_path):
    node = TreeNode(os.path.basename(dir_path))
    files = sorted(os.listdir(dir_path))
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        if os.path.isdir(file_path):
            child_node = build_tree(file_path)
            node.add_child(child_node)
        else:
            node.add_child(file_name)
    return node

# 指定要查找的文件夹路径
folder_path = './models/CFGAN'

# 建立目录树
root = build_tree(folder_path)
root = build_tree(folder_path)

print(root.print_tree())
# 将目录树写入文本文件
with open('tree.txt', 'w') as f:
    f.write(root.print_tree())
