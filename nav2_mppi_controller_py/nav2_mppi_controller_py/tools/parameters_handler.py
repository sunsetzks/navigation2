from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

try:
    from rcl_interfaces.msg import Parameter, SetParametersResult
except ImportError:  # pragma: no cover
    Parameter = object  # type: ignore
    SetParametersResult = object  # type: ignore


DynamicCallback = Callable[[Any], None]


class ParametersHandler:
    """Python helper that mimics the C++ dynamic parameter handler."""

    def __init__(self, node=None) -> None:
        self._node = node
        self._node_name = node.get_name() if node is not None else ""
        self._logger = getattr(node, "get_logger", lambda: None)()

        self._lock = threading.Lock()
        self._dynamic_callbacks: Dict[str, DynamicCallback] = {}
        self._post_callbacks: List[Callable[[], None]] = []
        self._pre_callbacks: List[Callable[[], None]] = []
        self._callback_handle = None

    def start(self) -> None:
        if self._node is None or not hasattr(self._node, "add_on_set_parameters_callback"):
            return

        self._callback_handle = self._node.add_on_set_parameters_callback(self._dynamic_params_callback)

    def get_param_getter(self, namespace: str) -> Callable[..., Any]:
        def getter(
            name: str,
            default_value: Any,
            param_type: str = "dynamic",
            on_change: Optional[DynamicCallback] = None,
        ) -> Any:
            fully_qualified = f"{namespace}.{name}" if namespace else name
            value = self._declare_and_get(fully_qualified, default_value)
            if param_type == "dynamic":
                if on_change is None:
                    on_change = lambda new_value: None  # noqa: E731
                self._dynamic_callbacks[fully_qualified] = on_change
            return value

        return getter

    def add_post_callback(self, callback: Callable[[], None]) -> None:
        self._post_callbacks.append(callback)

    def add_pre_callback(self, callback: Callable[[], None]) -> None:
        self._pre_callbacks.append(callback)

    def add_dynamic_param_callback(self, name: str, callback: DynamicCallback) -> None:
        self._dynamic_callbacks[name] = callback

    def get_lock(self) -> threading.Lock:
        return self._lock

    def _declare_and_get(self, name: str, default_value: Any) -> Any:
        if self._node is None:
            return default_value

        if not self._node.has_parameter(name):
            self._node.declare_parameter(name, default_value)

        param = self._node.get_parameter(name)
        return self._from_parameter_value(param)

    @staticmethod
    def _from_parameter_value(param) -> Any:
        if hasattr(param, "value"):
            return param.value

        value = getattr(param, "get_parameter_value", lambda: param)()
        candidates = (
            "bool_value",
            "integer_value",
            "double_value",
            "string_value",
            "bool_array_value",
            "integer_array_value",
            "double_array_value",
            "string_array_value",
        )
        for attr in candidates:
            if hasattr(value, attr):
                raw = getattr(value, attr)
                return list(raw) if isinstance(raw, (tuple, list)) else raw
        return value

    def _dynamic_params_callback(self, parameters: List[Parameter]) -> SetParametersResult:
        result = SetParametersResult() if SetParametersResult is not object else None
        if result is not None:
            result.successful = True
        with self._lock:
            for callback in self._pre_callbacks:
                callback()

            for param in parameters:
                callback = self._dynamic_callbacks.get(param.name)
                if callback:
                    callback(self._from_parameter_value(param))
                elif self._logger:
                    self._logger.warn(f"Parameter {param.name} not handled")

            for callback in self._post_callbacks:
                callback()
        return result
