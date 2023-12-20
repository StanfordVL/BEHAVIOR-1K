from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class StepRequest(_message.Message):
    __slots__ = ["action"]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: bytes
    def __init__(self, action: _Optional[bytes] = ...) -> None: ...

class StepResponse(_message.Message):
    __slots__ = ["observation", "reward", "terminated", "truncated", "info"]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    observation: bytes
    reward: float
    terminated: bool
    truncated: bool
    info: bytes
    def __init__(self, observation: _Optional[bytes] = ..., reward: _Optional[float] = ..., terminated: bool = ..., truncated: bool = ..., info: _Optional[bytes] = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ["seed", "options"]
    SEED_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    seed: int
    options: bytes
    def __init__(self, seed: _Optional[int] = ..., options: _Optional[bytes] = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ["observation", "reset_info"]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    RESET_INFO_FIELD_NUMBER: _ClassVar[int]
    observation: bytes
    reset_info: bytes
    def __init__(self, observation: _Optional[bytes] = ..., reset_info: _Optional[bytes] = ...) -> None: ...

class RenderRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class RenderResponse(_message.Message):
    __slots__ = ["render_data"]
    RENDER_DATA_FIELD_NUMBER: _ClassVar[int]
    render_data: bytes
    def __init__(self, render_data: _Optional[bytes] = ...) -> None: ...

class CloseRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class CloseResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetSpacesRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetSpacesResponse(_message.Message):
    __slots__ = ["observation_space", "action_space"]
    OBSERVATION_SPACE_FIELD_NUMBER: _ClassVar[int]
    ACTION_SPACE_FIELD_NUMBER: _ClassVar[int]
    observation_space: bytes
    action_space: bytes
    def __init__(self, observation_space: _Optional[bytes] = ..., action_space: _Optional[bytes] = ...) -> None: ...

class EnvMethodRequest(_message.Message):
    __slots__ = ["method_name", "arguments"]
    METHOD_NAME_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    method_name: str
    arguments: bytes
    def __init__(self, method_name: _Optional[str] = ..., arguments: _Optional[bytes] = ...) -> None: ...

class EnvMethodResponse(_message.Message):
    __slots__ = ["result"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: bytes
    def __init__(self, result: _Optional[bytes] = ...) -> None: ...

class GetAttrRequest(_message.Message):
    __slots__ = ["attribute_name"]
    ATTRIBUTE_NAME_FIELD_NUMBER: _ClassVar[int]
    attribute_name: str
    def __init__(self, attribute_name: _Optional[str] = ...) -> None: ...

class GetAttrResponse(_message.Message):
    __slots__ = ["attribute_value"]
    ATTRIBUTE_VALUE_FIELD_NUMBER: _ClassVar[int]
    attribute_value: bytes
    def __init__(self, attribute_value: _Optional[bytes] = ...) -> None: ...

class SetAttrRequest(_message.Message):
    __slots__ = ["attribute_name", "attribute_value"]
    ATTRIBUTE_NAME_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTE_VALUE_FIELD_NUMBER: _ClassVar[int]
    attribute_name: str
    attribute_value: bytes
    def __init__(self, attribute_name: _Optional[str] = ..., attribute_value: _Optional[bytes] = ...) -> None: ...

class SetAttrResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class IsWrappedRequest(_message.Message):
    __slots__ = ["wrapper_type"]
    WRAPPER_TYPE_FIELD_NUMBER: _ClassVar[int]
    wrapper_type: str
    def __init__(self, wrapper_type: _Optional[str] = ...) -> None: ...

class IsWrappedResponse(_message.Message):
    __slots__ = ["is_wrapped"]
    IS_WRAPPED_FIELD_NUMBER: _ClassVar[int]
    is_wrapped: bool
    def __init__(self, is_wrapped: bool = ...) -> None: ...

class RegisterEnvironmentRequest(_message.Message):
    __slots__ = ["ip", "port"]
    IP_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    ip: str
    port: int
    def __init__(self, ip: _Optional[str] = ..., port: _Optional[int] = ...) -> None: ...

class RegisterEnvironmentResponse(_message.Message):
    __slots__ = ["success"]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...
