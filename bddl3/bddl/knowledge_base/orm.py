from functools import cached_property, cache
from collections import defaultdict
from dataclasses import field, fields
import uuid


class IntegrityError(Exception):
    pass


class Model:
    _CLASS_REGISTRY = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Model._CLASS_REGISTRY[cls.__name__] = cls

    @staticmethod
    def get_model_class(target_model):
        return Model._CLASS_REGISTRY[target_model]

    _OBJECT_REGISTRY = defaultdict(dict)

    def __post_init__(self):
        key = self._key
        assert (
            key not in self._OBJECT_REGISTRY[self.__class__.__name__]
        ), f"Duplicate key {key} for {self.__class__.__name__}"

        self._OBJECT_REGISTRY[self.__class__.__name__][key] = self

        # Also bind the "this_instance" member of each fk field
        for field in fields(self):
            if field.type in (ManyToOne, OneToMany, ManyToMany):
                fk = getattr(self, field.name)
                fk._this_inst = self

    @cached_property
    def _key(self):
        return getattr(self, self.Meta.pk)

    @classmethod
    def get(cls, *args, **kwargs):
        if args:
            assert (
                not kwargs
            ), "Cannot specify both positional and keyword arguments to get"
            (pk,) = args
            kwargs = {cls.Meta.pk: pk}
        assert kwargs, "Must specify either positional or keyword arguments to get"
        # Use a fast lookup for the PK.
        if len(kwargs) == 1 and cls.Meta.pk in kwargs:
            key = kwargs[cls.Meta.pk]
            if key is None:
                return None
            return (
                cls._OBJECT_REGISTRY[cls.__name__][key]
                if key in cls._OBJECT_REGISTRY[cls.__name__]
                else None
            )
        objs = [
            x
            for x in cls.all_objects()
            if all(getattr(x, attr) == value for attr, value in kwargs.items())
        ]
        if not objs:
            return None
        assert (
            len(objs) == 1
        ), f"Expected exactly one object matching {kwargs} but found {objs}"
        return objs[0]

    @classmethod
    def exists(cls, *args, **kwargs):
        if args:
            assert (
                not kwargs
            ), "Cannot specify both positional and keyword arguments to get"
            (pk,) = args
            kwargs = {cls.Meta.pk: pk}
        assert kwargs, "Must specify either positional or keyword arguments to get"
        # Use a fast lookup for the PK.
        if len(kwargs) == 1 and cls.Meta.pk in kwargs:
            return kwargs[cls.Meta.pk] in cls._OBJECT_REGISTRY[cls.__name__]
        objs = [
            x
            for x in cls.all_objects()
            if all(getattr(x, attr) == value for attr, value in kwargs.items())
        ]
        return bool(objs)

    @classmethod
    def get_or_create(cls, defaults=None, *args, **kwargs):
        if cls.exists(*args, **kwargs):
            return cls.get(*args, **kwargs), False
        else:
            create_args = {}
            if defaults:
                create_args.update(defaults)
            if args:
                assert (
                    not kwargs
                ), "Cannot specify both positional and keyword arguments to get"
                (pk,) = args
                create_args[cls.Meta.pk] = pk
            if kwargs:
                create_args.update(kwargs)
            return cls.create(**create_args), True

    @classmethod
    def create(cls, **kwargs):
        # Check the kwargs to see if any are foreign keys
        sanitized_kwargs = {
            k: v for k, v in kwargs.items() if k not in cls.foreign_key_field_names()
        }
        obj = cls(**sanitized_kwargs)

        foreign_kwargs = {
            k + "_fk": v
            for k, v in kwargs.items()
            if k in cls.foreign_key_field_names()
        }
        for field_name, field_value in foreign_kwargs.items():
            fk = getattr(obj, field_name)
            if isinstance(fk, ManyToOne):
                fk.set(field_value)
            elif isinstance(fk, ManyToMany):
                for obj in field_value:
                    fk.add(obj)
            elif isinstance(fk, OneToMany):
                for obj in field_value:
                    fk.add(obj)
                fk.add(obj, field_value)
            else:
                raise ValueError(f"Unknown foreign key type {fk}")

        # Build the unique-together key
        # TODO: Make this work beyond just creation.
        # if hasattr(cls.Meta, "unique_together"):
        #     unique_together_keys = cls.Meta.unique_together
        #     this_key = tuple([getattr(obj, attr) for attr in unique_together_keys])
        #     for other in cls._OBJECT_REGISTRY[cls.__name__].values():
        #         if other is obj:
        #             continue
        #         other_key = tuple([getattr(other, attr) for attr in unique_together_keys])
        #         if this_key == other_key:
        #             raise IntegrityError(f"Unique together constraint violated for {cls.__name__}. New item: {obj!r}. Existing item: {other!r}. Uniqueness constraint: {unique_together_keys}")

        return obj

    @classmethod
    def foreign_key_field_names(cls):
        fk_fields = [
            field.name
            for field in fields(cls)
            if field.type in (ManyToOne, OneToMany, ManyToMany)
        ]
        assert all(
            field.endswith("_fk") for field in fk_fields
        ), f"Foreign key fields must end with _fk. Fields: {fk_fields}"
        return [field[:-3] for field in fk_fields]

    @classmethod
    def all_objects(cls):
        return sorted(x for x in cls._OBJECT_REGISTRY[cls.__name__].values())

    def __lt__(self, other):
        if hasattr(self.Meta, "ordering"):
            order_attrs = self.Meta.ordering
        else:
            order_attrs = [self.Meta.pk]
        self_order_key = [getattr(self, attr) for attr in order_attrs]
        other_order_key = [getattr(other, attr) for attr in order_attrs]
        return self_order_key < other_order_key

    def __getattr__(self, key):
        try:
            fk = self.__getattribute__(key + "_fk")
            if hasattr(fk, "get"):  # ManyToOne
                return fk.get()
            else:
                return fk
        except AttributeError:
            return self.__getattribute__(key)

    def __setattr__(self, key, value):
        try:
            fk = self.__getattribute__(key + "_fk")
            if hasattr(fk, "set"):  # ManyToOne
                return fk.set(value)
            else:
                raise ValueError(
                    f"Cannot set {key} because it is a OneToMany or ManyToMany field."
                )
        except AttributeError:
            return super().__setattr__(key, value)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, __value: object) -> bool:
        return __value is self


def has_field_with_type(cls, field_name, field_type):
    cls_fields = [x for x in fields(cls) if x.name == field_name]
    if not cls_fields:
        return False
    (field,) = cls_fields
    return field.type == field_type


def get_model_class(target_model):
    if isinstance(target_model, str):
        return Model.get_model_class(target_model)
    elif isinstance(target_model, type) and issubclass(target_model, Model):
        return target_model
    else:
        raise ValueError(f"Invalid target model {target_model}")


class ManyToOne:
    def __init__(self, target_model, target_field):
        target_model = get_model_class(target_model)
        target_field = target_field + "_fk"
        assert has_field_with_type(
            target_model, target_field, OneToMany
        ), f"Target model {target_model} does not have a OneToMany field {target_field}. Fields: {[x for x in fields(target_model)]}"

        self._target_model = target_model
        self._target_field = target_field

        self._target_key = None
        self._this_inst = None

    def get(self):
        assert self._this_inst is not None, "Unbound foreign key object"
        return self._target_model.get(self._target_key)

    def set(self, target_inst):
        assert self._target_key is None, "Foreign key object already bound"
        assert self._this_inst is not None, "Unbound foreign key object"
        assert (
            target_inst.__class__ == self._target_model
        ), f"Target instance {target_inst} is not of type {self._target_model}"
        self._target_key = target_inst._key
        assert (
            self._this_inst._key not in getattr(target_inst, self._target_field)._values
        ), f"Foreign key constraint violated: {self._this_inst} already has a foreign key to {target_inst}"
        getattr(target_inst, self._target_field)._values.append(self._this_inst._key)


class OneToMany:
    def __init__(self, target_model, target_field) -> None:
        target_model = get_model_class(target_model)
        target_field = target_field + "_fk"
        assert has_field_with_type(
            target_model, target_field, ManyToOne
        ), f"Target model {target_model} does not have a ManyToOne field {target_field}. Fields: {[x for x in fields(target_model)]}"

        self._target_model = target_model
        self._target_field = target_field

        self._values = []
        self._this_inst = None

    def __iter__(self):
        assert self._this_inst is not None, "Unbound foreign key object"
        return (self._target_model.get(key) for key in self._values)

    def __len__(self):
        assert self._this_inst is not None, "Unbound foreign key object"
        return len(self._values)

    def __getitem__(self, key):
        assert self._this_inst is not None, "Unbound foreign key object"
        if isinstance(key, slice):
            return (self._target_model.get(key) for key in self._values[key])
        else:
            return self._target_model.get(self._values[key])

    def add(self, target_inst):
        # Directly call their setter for verification
        assert self._this_inst is not None, "Unbound foreign key object"
        getattr(target_inst, self._target_field).set(self._this_inst._key)


class ManyToMany:
    def __init__(self, target_model, target_field) -> None:
        target_model = get_model_class(target_model)
        target_field = target_field + "_fk"
        assert has_field_with_type(
            target_model, target_field, ManyToMany
        ), f"Target model {target_model} does not have a ManyToMany field {target_field}. Fields: {[x for x in fields(target_model)]}"

        self._target_model = target_model
        self._target_field = target_field

        self._values = []
        self._this_inst = None

    def __iter__(self):
        assert self._this_inst is not None, "Unbound foreign key object"
        return (self._target_model.get(key) for key in self._values)

    def __getitem__(self, key):
        assert self._this_inst is not None, "Unbound foreign key object"
        if isinstance(key, slice):
            return (self._target_model.get(key) for key in self._values[key])
        else:
            return self._target_model.get(self._values[key])

    def __len__(self):
        assert self._this_inst is not None, "Unbound foreign key object"
        return len(self._values)

    def add(self, target_inst):
        assert self._this_inst is not None, "Unbound foreign key object"
        assert (
            target_inst.__class__ == self._target_model
        ), f"Target instance {target_inst} is not of type {self._target_model}"
        assert (
            target_inst._key not in self._values
        ), f"Foreign key constraint violated: {self._this_inst} already has a foreign key to {target_inst}"
        assert (
            self._this_inst._key not in getattr(target_inst, self._target_field)._values
        ), f"Foreign key constraint violated: {self._this_inst} already has a foreign key to {target_inst}"
        self._values.append(target_inst._key)
        getattr(target_inst, self._target_field)._values.append(self._this_inst._key)


def ManyToOneField(target_model, target_field, **kwargs):
    return field(
        default_factory=lambda: ManyToOne(target_model, target_field),
        repr=False,
        compare=False,
        **kwargs,
    )


def OneToManyField(target_model, target_field, **kwargs):
    return field(
        default_factory=lambda: OneToMany(target_model, target_field),
        repr=False,
        compare=False,
        **kwargs,
    )


def ManyToManyField(target_model, target_field, **kwargs):
    return field(
        default_factory=lambda: ManyToMany(target_model, target_field),
        repr=False,
        compare=False,
        **kwargs,
    )


def UUIDField(**kwargs):
    return field(default_factory=lambda: str(uuid.uuid4()), **kwargs)
