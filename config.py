import contextlib


class Config:
    enable_backprop = True

    @classmethod
    @contextlib.contextmanager
    def using_config(cls, **kwargs):
        old_values = {}
        for key, new_val in kwargs.items():
            # if not hasattr(cls, key):
            #     raise AttributeError(f"{cls.__name__} has no attribute '{key}'")
            old_values[key] = getattr(cls, key)
            setattr(cls, key, new_val)
        try:
            yield
        finally:
            for key, old_val in old_values.items():
                setattr(cls, key, old_val)

    @classmethod
    def no_grad(cls):
        return cls.using_config(enable_backprop=False)
