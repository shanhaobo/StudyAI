import os
import re

class FileRangeManager:
    def __init__(self, root, range_size, extension):
        self.root = root
        self.range_size = range_size
        self.extension = extension

    def _GetExistingRangeAndPath(self, num):
        ranges = [(int(m.group(1)), int(m.group(2))) for d in os.listdir(self.root)
                  if (m := re.match(r'(\d+)_(\d+)', d))]
        ranges.sort(key=lambda x: x[1])
        for r in ranges:
            if r[0] <= num <= r[1]:
                path = os.path.join(self.root, f"{r[0]}_{r[1]}")
                return r, path
        return None, None

    def _create_file_path(self, file_prefix, path, num):
        return os.path.join(path, f"{file_prefix}{num}.{self.extension}")

    def get_file(self, file_prefix, num):
        range_, path = self._GetExistingRangeAndPath(num)
        if range_:
            return self._create_file_path(path, file_prefix, num)
        return None

    def store_file(self, file_prefix,  num):
        range_, path = self._GetExistingRangeAndPath(num)
        if not range_:
            range_ = (num, num + self.range_size)
            path = os.path.join(self.root, f"{range_[0]}_{range_[1]}")
            os.makedirs(path, exist_ok=True)
        return self._create_file_path(path,file_prefix,num)

    def get_max_num_file(self):
        max_num = -1
        max_file_path = None
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if (match := re.match(self.file_prefix + r'(\d+).' + self.extension, file)):
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
                        max_file_path = os.path.join(root, file)
        return max_file_path
