from collections.abc import Sequence
from typing import Any, cast

from lumilake_server.common import Message
from lumilake_server.ops.ops import FunctionalOp, Op
from lumilake_server.utils.utils import check_and_cast_list


class OpMessage:
    role: str
    content: str | Op

    def __init__(self, role: str, content: str | list[str] | Op) -> None:
        self.role = role
        self.content = DataOp(data=content) if isinstance(content, list) else content

    def content_or_id(self) -> str:
        return self.content.id if isinstance(self.content, Op) else self.content


@Op.registry.register("DataOp")
class DataOp(FunctionalOp):
    data: list[str]

    def __init__(self, data: list[str]) -> None:
        super().__init__()
        self.data = data

    def _serialize(self) -> dict[str, Any]:
        return dict(data=self.data)

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "DataOp":
        return cls(data=data["data"])


@Op.registry.register("DataRetrievalOp")
class DataRetrievalOp(FunctionalOp):
    data_spec: dict[str, Any]

    def __init__(
        self, data_spec: dict[str, Any], inputs: list[Op] | None = None
    ) -> None:
        super().__init__(inputs or [])
        self.data_spec = data_spec

    def _serialize(self) -> dict[str, Any]:
        return {"data_spec": self.data_spec}

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "DataRetrievalOp":
        inputs = [other_ops[op_id] for op_id in data.get("_inputs", [])]
        return cls(data_spec=data["data_spec"], inputs=inputs)


@Op.registry.register("MessageOp")
class MessageOp(FunctionalOp):
    def __init__(self, messages: list[OpMessage]) -> None:
        inputs: list[Op] = []
        message_refs: list[int | str] = []
        for msg in messages:
            if isinstance(msg.content, str):
                message_refs.append(msg.content)
            else:
                try:
                    i = inputs.index(msg.content)
                    message_refs.append(i)
                except ValueError:
                    message_refs.append(len(inputs))
                    inputs.append(msg.content)
        super().__init__(inputs)
        self._message_refs = message_refs
        self._roles = [msg.role for msg in messages]

    @property
    def messages(self) -> list[OpMessage]:
        return [
            OpMessage(
                role=role, content=self.inputs[ref] if isinstance(ref, int) else ref
            )
            for role, ref in zip(self._roles, self._message_refs)
        ]

    @property
    def roles(self) -> list[str]:
        return self._roles

    @property
    def message_refs(self) -> list[int | str]:
        return self._message_refs

    def _serialize(self) -> dict[str, Any]:
        return dict(
            messages=[
                {"role": message.role, "content": message.content_or_id()}
                for message in self.messages
            ]
        )

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "MessageOp":
        def get_content(content_or_id: str) -> str | Op:
            if content_or_id in other_ops:
                return other_ops[content_or_id]
            return content_or_id

        messages = [
            OpMessage(role=message["role"], content=get_content(message["content"]))
            for message in data["messages"]
        ]
        return cls(messages)


def message_data(data: Sequence[Message | OpMessage]) -> MessageOp:
    return MessageOp(
        [
            (
                msg
                if isinstance(msg, OpMessage)
                else OpMessage(role="user", content=msg.content)
            )
            for msg in data
        ]
    )


@Op.registry.register("InputOp")
class InputOp(FunctionalOp):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def _serialize(self) -> dict[str, Any]:
        return dict(name=self.name)

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "InputOp":
        return cls(name=data["name"])


def input_placeholder(name: str) -> InputOp:
    return InputOp(name)


@Op.registry.register("OutputOp")
class OutputOp(FunctionalOp):
    name: str
    path: str | None

    def __init__(
        self,
        name: str,
        output: list[str] | Op,
        path: str | None = None,
    ) -> None:
        inp = output if isinstance(output, Op) else DataOp(output)
        super().__init__([inp])
        self.name = name
        self.path = path

    @property
    def op(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.path is not None:
            payload["path"] = self.path
        return payload

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "OutputOp":
        input_ids = data["_inputs"]
        if len(input_ids) != 1:
            raise ValueError("OutputOp must have exactly one input")
        return cls(
            name=data["name"],
            output=other_ops[input_ids[0]],
            path=data.get("path"),
        )


def as_output(name: str, output: list[str] | Op, path: str | None = None) -> OutputOp:
    return OutputOp(name, output, path=path)


def data(
    data: str | list[str] | list[Message | OpMessage],
) -> DataOp | MessageOp:
    if isinstance(data, str):
        return DataOp(data=[data])
    if len(data) == 0:
        return DataOp(data=cast(list, data))
    if isinstance(data[0], str):
        return DataOp(data=check_and_cast_list(str, data))
    return message_data(cast(list[Message | OpMessage], data))
