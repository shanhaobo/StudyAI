from collections import UserDict, UserList
from collections import UserDict

##################################################################################################

class CaseInsensitiveDict(UserDict):
    def __init__(self, initial_data=None, **kwargs):
        super().__init__()
        if initial_data:
            self.update(initial_data)
        if kwargs:
            self.update(kwargs)

    def __setitem__(self, key, value):
        self.data[key.lower()] = value

    def __getitem__(self, key):
        return self.data.get(key.lower())

    def __delitem__(self, key):
        self.data.pop(key.lower(), None)

    def __contains__(self, key):
        return key.lower() in self.data

    def get(self, key, default=None):
        return self.data.get(key.lower(), default)

    def pop(self, key, default=None):
        return self.data.pop(key.lower(), default)

    def update(self, other=None, **kwargs):
        if other is not None:
            if isinstance(other, dict):
                for key, value in other.items():
                    self.data[key.lower()] = value
            else:
                for key, value in other:
                    self.data[key.lower()] = value
        for key, value in kwargs.items():
            self.data[key.lower()] = value

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()


##################################################################################################


class CaseInsensitiveList(UserList):
    def __init__(self, initial_data=None):
        if initial_data:
            super().__init__(map(str.lower, initial_data))
        else:
            super().__init__()

    def __contains__(self, item):
        return super().__contains__(item.lower())

    def append(self, item):
        super().append(item.lower())

    def insert(self, i, item):
        super().insert(i, item.lower())

    def extend(self, iterable):
        super().extend(map(str.lower, iterable))

    def __setitem__(self, i, item):
        super().__setitem__(i, item.lower())

    def __eq__(self, other):
        if isinstance(other, CaseInsensitiveList):
            return super().__eq__(map(str.lower, other))
        else:
            return super().__eq__(other)
