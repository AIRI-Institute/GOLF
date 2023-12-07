import inspect


def ignore_extra_args(foo):
    def indifferent_foo(**kwargs):
        signature = inspect.signature(foo)
        expected_keys = [
            p.name
            for p in signature.parameters.values()
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        filtered_kwargs = {k: kwargs[k] for k in kwargs if k in expected_keys}
        return foo(**filtered_kwargs)

    return indifferent_foo
