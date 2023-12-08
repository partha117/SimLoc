import json


class Configuration(dict):
    def __init__(self, path=None, content=None):
        super(Configuration, self).__init__()
        if path is None and content is None:
            raise AttributeError("Both can not be none")
        if path:
            with open(path, "r") as f:
                content = json.load(f)
        super(Configuration, self).__init__(content)
        self.__dict__ = self
