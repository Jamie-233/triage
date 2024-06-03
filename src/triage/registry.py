"""Tool registry: Python functions in, JSON-Schema tool specs out.

Hand-writing a JSON schema next to every tool means two definitions that
drift apart, and the model then calls the tool with arguments the code no
longer accepts. Deriving the schema from the signature keeps one source of
truth; the docstring's first paragraph becomes the description the model
sees, and ``Args:`` lines become per-parameter descriptions.

Tool descriptions are prompt engineering. A model picks tools from these
strings alone, so they say *when* to use a tool and what its output means,
not just what it does.
"""

from __future__ import annotations

import inspect
import re
import typing
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, get_args, get_origin

ToolFn = Callable[..., Awaitable[dict]]

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: ToolFn

    def spec(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def _json_type(annotation: Any) -> dict:
    origin = get_origin(annotation)
    if origin is Literal:
        values = list(get_args(annotation))
        return {"type": _PY_TO_JSON[type(values[0])], "enum": values}
    if origin in (list, typing.List):
        (inner,) = get_args(annotation) or (str,)
        return {"type": "array", "items": _json_type(inner)}
    if origin is typing.Union or str(origin) == "types.UnionType":
        # Optional[X] → X; the "optional" part is expressed via `required`.
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return _json_type(non_none[0])
    if annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}
    raise TypeError(f"unsupported tool parameter type: {annotation!r}")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    doc = inspect.cleandoc(doc or "")
    summary, _, rest = doc.partition("\nArgs:")
    params = {}
    for m in re.finditer(r"^\s+(\w+):\s*(.+(?:\n\s{8,}.+)*)", rest, re.M):
        params[m.group(1)] = " ".join(m.group(2).split())
    return " ".join(summary.split()), params


class Registry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def tool(self, fn: ToolFn) -> ToolFn:
        hints = typing.get_type_hints(fn, include_extras=False)
        description, arg_docs = _parse_docstring(fn.__doc__ or "")
        props, required = {}, []
        for name, param in inspect.signature(fn).parameters.items():
            schema = _json_type(hints[name])
            if name in arg_docs:
                schema["description"] = arg_docs[name]
            if param.default is inspect.Parameter.empty:
                required.append(name)
            else:
                schema["default"] = param.default
            props[name] = schema
        self.tools[fn.__name__] = Tool(
            fn.__name__, description,
            {"type": "object", "properties": props, "required": required, "additionalProperties": False},
            fn,
        )
        return fn

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def coerce(self, name: str, args: dict) -> dict:
        """Validate model-supplied arguments before anything runs.

        Models occasionally invent parameters or send "443" for an integer.
        Rejecting unknown keys and coercing obvious scalars here turns those
        into a readable error the model can correct on its next step, instead
        of a TypeError deep inside a tool.
        """
        tool = self.tools[name]
        props = tool.parameters["properties"]
        unknown = set(args) - set(props)
        if unknown:
            raise ValueError(f"unknown argument(s) for {name}: {', '.join(sorted(unknown))}")
        missing = [r for r in tool.parameters["required"] if r not in args]
        if missing:
            raise ValueError(f"missing required argument(s) for {name}: {', '.join(missing)}")
        out = {}
        for key, value in args.items():
            kind = props[key].get("type")
            try:
                if kind == "integer" and not isinstance(value, bool):
                    value = int(value)
                elif kind == "number":
                    value = float(value)
                elif kind == "string":
                    value = str(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name}.{key} must be {kind}, got {value!r}") from None
            if "enum" in props[key] and value not in props[key]["enum"]:
                raise ValueError(f"{name}.{key} must be one of {props[key]['enum']}")
            out[key] = value
        return out
