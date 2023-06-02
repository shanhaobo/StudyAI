import os
import re

class FileManager:
    def __init__(self, root, file_prefix, range_size, extension):
        self.root = root
        self.file_prefix = file_prefix
        self.range_size = range_size
        self.extension = extension

    def _get_existing_range(self, category, num):
        category_dir = os.path.join(self.root, str(category))
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        ranges = [re.match(r'(\d+)_(\d+)', d) for d in os.listdir(category_dir)]
        ranges = [(int(m.group(1)), int(m.group(2))) for m in ranges if m]
        ranges.sort(key=lambda x: x[1])
        for r in ranges:
            if r[0] <= num <= r[1]:
                return r
        return None
    
    def get_file(self, category, file_prefix, num):
        range_ = self._get_existing_range(category, num)
        if range_:
            path = os.path.join(self.root, str(category), f"{range_[0]}_{range_[1]}", f"{file_prefix}{num}.{self.extension}")
            
            return path
        
        return None

    def store_file(self, category, file_prefix, num, content):
        range_ = self._get_existing_range(category, num)
        if not range_:
            range_ = (num, num + self.range_size)  # 创建新的范围
        path = os.path.join(self.root, str(category), f"{range_[0]}_{range_[1]}")
        if not os.path.exists(path):
            os.makedirs(path)
        
        return os.path.join(path, f"{file_prefix}{num}.{self.extension}")

    def get_max_num_file(self, category, file_prefix):
        category_dir = os.path.join(self.root, str(category))
        if not os.path.exists(category_dir):
            print(f"类别'{category}'不存在.")
            return None
        max_num = -1
        max_file_path = None
        for root, dirs, files in os.walk(category_dir):
            for file in files:
                match = re.match(file_prefix + r'(\d+).' + self.extension, file)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
                        max_file_path = os.path.join(root, file)
        return max_file_path
