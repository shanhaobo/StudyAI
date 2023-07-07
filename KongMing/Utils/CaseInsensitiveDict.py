from collections import defaultdict

class CaseInsensitiveDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super(CaseInsensitiveDict, self).__init__(*args, **kwargs)
        self._convert_key = self._str_lower

    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(self._convert_key(key))

    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(self._convert_key(key), value)

    def __delitem__(self, key):
        super(CaseInsensitiveDict, self).__delitem__(self._convert_key(key))

    def __contains__(self, key):
        return super(CaseInsensitiveDict, self).__contains__(self._convert_key(key))

    @staticmethod
    def _str_lower(s):
        if isinstance(s, str):
            return s.lower()
        return s
