from collections import UserDict

class CaseInsensitiveDict(UserDict):
    def __init__(self, *args, **kwargs):
        super(CaseInsensitiveDict, self).__init__(*args, **kwargs)
        self.data = {k.lower(): v for k, v in self.data.items()}

    def __setitem__(self, key, value):
        self.data[key.lower()] = value

    def __getitem__(self, key):
        return self.data[key.lower()]

    def __delitem__(self, key):
        del self.data[key.lower()]


class CaseInsensitiveList(list):
    def __contains__(self, item):
        return super(CaseInsensitiveList, self).__contains__(item.lower())

    def append(self, item):
        super(CaseInsensitiveList, self).append(item.lower())
