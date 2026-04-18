class SingletonMeta(type):
    """
    Metaclass implementing the Singleton Design Pattern.
    Ensures only one instance of a class ever exists in memory across the entire Python process.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]