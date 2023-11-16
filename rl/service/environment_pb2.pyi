from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EnvironmentRequest(_message.Message):
    __slots__ = ["step", "reset", "render", "close", "get_spaces", "env_method", "get_attr", "set_attr", "is_wrapped"]
    STEP_FIELD_NUMBER: _ClassVar[int]
    RESET_FIELD_NUMBER: _ClassVar[int]
    RENDER_FIELD_NUMBER: _ClassVar[int]
    CLOSE_FIELD_NUMBER: _ClassVar[int]
    GET_SPACES_FIELD_NUMBER: _ClassVar[int]
    ENV_METHOD_FIELD_NUMBER: _ClassVar[int]
    GET_ATTR_FIELD_NUMBER: _ClassVar[int]
    SET_ATTR_FIELD_NUMBER: _ClassVar[int]
    IS_WRAPPED_FIELD_NUMBER: _ClassVar[int]
    step: StepCommand
    reset: ResetCommand
    render: RenderCommand
    close: CloseCommand
    get_spaces: GetSpacesCommand
    env_method: EnvMethodCommand
    get_attr: GetAttrCommand
    set_attr: SetAttrCommand
    is_wrapped: IsWrappedCommand
    def __init__(self, step: _Optional[_Union[StepCommand, _Mapping]] = ..., reset: _Optional[_Union[ResetCommand, _Mapping]] = ..., render: _Optional[_Union[RenderCommand, _Mapping]] = ..., close: _Optional[_Union[CloseCommand, _Mapping]] = ..., get_spaces: _Optional[_Union[GetSpacesCommand, _Mapping]] = ..., env_method: _Optional[_Union[EnvMethodCommand, _Mapping]] = ..., get_attr: _Optional[_Union[GetAttrCommand, _Mapping]] = ..., set_attr: _Optional[_Union[SetAttrCommand, _Mapping]] = ..., is_wrapped: _Optional[_Union[IsWrappedCommand, _Mapping]] = ...) -> None: ...

class EnvironmentResponse(_message.Message):
    __slots__ = ["step_response", "reset_response", "render_response", "close_response", "get_spaces_response", "env_method_response", "get_attr_response", "set_attr_response", "is_wrapped_response"]
    STEP_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    RESET_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    RENDER_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    CLOSE_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    GET_SPACES_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    ENV_METHOD_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    GET_ATTR_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    SET_ATTR_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    IS_WRAPPED_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    step_response: StepResponse
    reset_response: ResetResponse
    render_response: RenderResponse
    close_response: CloseResponse
    get_spaces_response: GetSpacesResponse
    env_method_response: EnvMethodResponse
    get_attr_response: GetAttrResponse
    set_attr_response: SetAttrResponse
    is_wrapped_response: IsWrappedResponse
    def __init__(self, step_response: _Optional[_Union[StepResponse, _Mapping]] = ..., reset_response: _Optional[_Union[ResetResponse, _Mapping]] = ..., render_response: _Optional[_Union[RenderResponse, _Mapping]] = ..., close_response: _Optional[_Union[CloseResponse, _Mapping]] = ..., get_spaces_response: _Optional[_Union[GetSpacesResponse, _Mapping]] = ..., env_method_response: _Optional[_Union[EnvMethodResponse, _Mapping]] = ..., get_attr_response: _Optional[_Union[GetAttrResponse, _Mapping]] = ..., set_attr_response: _Optional[_Union[SetAttrResponse, _Mapping]] = ..., is_wrapped_response: _Optional[_Union[IsWrappedResponse, _Mapping]] = ...) -> None: ...

class StepCommand(_message.Message):
    __slots__ = ["action"]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: bytes
    def __init__(self, action: _Optional[bytes] = ...) -> None: ...

class StepResponse(_message.Message):
    __slots__ = ["observation", "reward", "done", "info", "reset_info"]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    RESET_INFO_FIELD_NUMBER: _ClassVar[int]
    observation: bytes
    reward: float
    done: bool
    info: bytes
    reset_info: bytes
    def __init__(self, observation: _Optional[bytes] = ..., reward: _Optional[float] = ..., done: bool = ..., info: _Optional[bytes] = ..., reset_info: _Optional[bytes] = ...) -> None: ...

class ResetCommand(_message.Message):
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

class RenderCommand(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class RenderResponse(_message.Message):
    __slots__ = ["render_data"]
    RENDER_DATA_FIELD_NUMBER: _ClassVar[int]
    render_data: bytes
    def __init__(self, render_data: _Optional[bytes] = ...) -> None: ...

class CloseCommand(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class CloseResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetSpacesCommand(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetSpacesResponse(_message.Message):
    __slots__ = ["observation_space", "action_space"]
    OBSERVATION_SPACE_FIELD_NUMBER: _ClassVar[int]
    ACTION_SPACE_FIELD_NUMBER: _ClassVar[int]
    observation_space: bytes
    action_space: bytes
    def __init__(self, observation_space: _Optional[bytes] = ..., action_space: _Optional[bytes] = ...) -> None: ...

class EnvMethodCommand(_message.Message):
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

class GetAttrCommand(_message.Message):
    __slots__ = ["attribute_name"]
    ATTRIBUTE_NAME_FIELD_NUMBER: _ClassVar[int]
    attribute_name: str
    def __init__(self, attribute_name: _Optional[str] = ...) -> None: ...

class GetAttrResponse(_message.Message):
    __slots__ = ["attribute_value"]
    ATTRIBUTE_VALUE_FIELD_NUMBER: _ClassVar[int]
    attribute_value: bytes
    def __init__(self, attribute_value: _Optional[bytes] = ...) -> None: ...

class SetAttrCommand(_message.Message):
    __slots__ = ["attribute_name", "attribute_value"]
    ATTRIBUTE_NAME_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTE_VALUE_FIELD_NUMBER: _ClassVar[int]
    attribute_name: str
    attribute_value: bytes
    def __init__(self, attribute_name: _Optional[str] = ..., attribute_value: _Optional[bytes] = ...) -> None: ...

class SetAttrResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class IsWrappedCommand(_message.Message):
    __slots__ = ["wrapper_type"]
    WRAPPER_TYPE_FIELD_NUMBER: _ClassVar[int]
    wrapper_type: str
    def __init__(self, wrapper_type: _Optional[str] = ...) -> None: ...

class IsWrappedResponse(_message.Message):
    __slots__ = ["is_wrapped"]
    IS_WRAPPED_FIELD_NUMBER: _ClassVar[int]
    is_wrapped: bool
    def __init__(self, is_wrapped: bool = ...) -> None: ...
