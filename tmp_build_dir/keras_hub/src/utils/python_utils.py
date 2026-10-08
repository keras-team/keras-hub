"""Utilities with miscellaneous python extensions."""


class classproperty(property):
    """Define a class level property."""

    def __get__(self, _, owner_cls):
        return self.fget(owner_cls)
