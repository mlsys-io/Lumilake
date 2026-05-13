import json
from collections.abc import Hashable
from typing import Any

from lumilake.common import GenerationConfig, Slice
from lumilake.ops.data_ops import DataOp, MessageOp, OpMessage
from lumilake.ops.ops import Op
from lumilake.utils.prefix.radix_tree import (
    MessagePrefixType,
    Placeholder,
    PrefixType,
    TextPrefixType,
    to_message_prefix,
    to_text_prefix,
)


def _structural_outputs_signature(
    value: dict[str, Any] | list[dict[str, Any]] | None,
) -> Hashable | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return ("dict", json.dumps(value, sort_keys=True))
    return ("list", tuple(tuple(sorted(item.items())) for item in value))


class LLMOp(Op):
    config: GenerationConfig
    cacheable: bool

    def __init__(
        self,
        inputs: list["Op"] | None,
        config: GenerationConfig | None,
        cacheable: bool,
    ) -> None:
        super().__init__(inputs)
        self.config = config or GenerationConfig.from_env()
        self.cacheable = cacheable

    def _serialize(self) -> dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "LLMOp":
        raise NotImplementedError()

    def _state_signature(self) -> Hashable | None:
        raise NotImplementedError()

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        raise NotImplementedError()

    def get_input_prefix_template(
        self, input_templates: dict[str, PrefixType]
    ) -> PrefixType:
        raise NotImplementedError()

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        raise NotImplementedError()


@Op.registry.register("LLMCompletionOp")
class LLMCompletionOp(LLMOp):
    echo: bool

    def __init__(
        self,
        prompt: list[str] | Op,
        config: GenerationConfig | None = None,
        echo: bool = False,
        cacheable: bool = False,
    ) -> None:
        prompt = prompt if isinstance(prompt, Op) else DataOp(data=prompt)
        super().__init__([prompt], config, cacheable)
        self.echo = echo

    @property
    def prompt(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        return dict(
            prompt=self.prompt.id,
            config=self.config.to_dict(),
            echo=self.echo,
            cacheable=self.cacheable,
        )

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "LLMCompletionOp":
        config = GenerationConfig(**data["config"])
        return cls(other_ops[data["prompt"]], config, data["echo"], data["cacheable"])

    def _state_signature(self) -> Hashable | None:
        return (tuple(self.config.to_dict().items()), self.echo, self.cacheable)

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> TextPrefixType:
        op_id = self.id
        op_id = sliced_op_map.get(op_id, op_id)
        history = self.get_input_prefix_template(input_templates) if self.echo else ()
        return history + (Placeholder(op_id),)

    def get_input_prefix_template(
        self, input_templates: dict[str, PrefixType]
    ) -> TextPrefixType:
        return to_text_prefix(input_templates[self.prompt.id])

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return input_slices[self.prompt.id]


def llm_completion(
    prompt: list[str] | Op,
    config: GenerationConfig | None = None,
    echo: bool = False,
    cacheable: bool = False,
) -> LLMCompletionOp:
    return LLMCompletionOp(prompt, config, echo, cacheable)


@Op.registry.register("LLMChatOp")
class LLMChatOp(LLMOp):
    config: GenerationConfig
    return_history: bool
    structural_outputs: dict[str, Any] | list[dict[str, Any]] | None
    aggregate_table: list[dict[str, Any]] | None
    rowwise_template: str | None
    rowwise_columns: list[dict[str, Any]] | None
    system_messages: list[str] | None
    condition: dict[str, str] | None

    def __init__(
        self,
        messages: list[OpMessage] | Op,
        config: GenerationConfig | None = None,
        return_history: bool = False,
        structural_outputs: dict[str, Any] | list[dict[str, Any]] | None = None,
        aggregate_table: list[dict[str, Any]] | None = None,
        rowwise_template: str | None = None,
        rowwise_columns: list[dict[str, Any]] | None = None,
        system_messages: list[str] | None = None,
        cacheable: bool = False,
        condition: dict[str, str] | None = None,
    ) -> None:
        messages = messages if isinstance(messages, Op) else MessageOp(messages)
        super().__init__([messages], config, cacheable)
        self.return_history = return_history
        self.structural_outputs = structural_outputs
        self.aggregate_table = aggregate_table
        self.rowwise_template = rowwise_template
        self.rowwise_columns = rowwise_columns
        self.system_messages = system_messages
        self.condition = condition

    @property
    def messages(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(
            messages=self.messages.id,
            config=self.config.to_dict(),
            return_history=self.return_history,
            cacheable=self.cacheable,
        )
        if self.structural_outputs is not None:
            payload["structural_outputs"] = self.structural_outputs
        if self.aggregate_table is not None:
            payload["aggregate_table"] = self.aggregate_table
        if self.rowwise_template is not None:
            payload["rowwise_template"] = self.rowwise_template
        if self.rowwise_columns is not None:
            payload["rowwise_columns"] = self.rowwise_columns
        if self.system_messages is not None:
            payload["system_messages"] = self.system_messages
        if self.condition is not None:
            payload["condition"] = self.condition
        return payload

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "LLMChatOp":
        message = other_ops[data["messages"]]
        config = GenerationConfig(**data["config"])
        return_history = data["return_history"]
        cacheable = data["cacheable"]
        structural_outputs = data.get("structural_outputs")
        aggregate_table = data.get("aggregate_table")
        rowwise_template = data.get("rowwise_template")
        rowwise_columns = data.get("rowwise_columns")
        system_messages = data.get("system_messages")
        condition = data.get("condition")
        return cls(
            message,
            config,
            return_history,
            structural_outputs,
            aggregate_table,
            rowwise_template,
            rowwise_columns,
            system_messages,
            cacheable,
            condition,
        )

    def _state_signature(self) -> Hashable | None:
        return (
            tuple(self.config.to_dict().items()),
            self.return_history,
            _structural_outputs_signature(self.structural_outputs),
            (
                tuple(tuple(sorted(item.items())) for item in self.aggregate_table)
                if self.aggregate_table
                else None
            ),
            self.rowwise_template,
            (
                tuple(tuple(sorted(item.items())) for item in self.rowwise_columns)
                if self.rowwise_columns
                else None
            ),
            tuple(self.system_messages) if self.system_messages else None,
            self.cacheable,
        )

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> PrefixType:
        op_id = self.id
        op_id = sliced_op_map.get(op_id, op_id)
        history = (
            self.get_input_prefix_template(input_templates)
            if self.return_history
            else ()
        )
        return history + (Placeholder(op_id),)

    def get_input_prefix_template(
        self, input_templates: dict[str, PrefixType]
    ) -> MessagePrefixType:
        return to_message_prefix(input_templates[self.messages.id])

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return input_slices[self.messages.id]


def llm_chat(
    messages: list[OpMessage] | Op,
    config: GenerationConfig | None = None,
    return_history: bool = False,
    cacheable: bool = False,
) -> LLMChatOp:
    return LLMChatOp(
        messages=messages,
        config=config,
        return_history=return_history,
        cacheable=cacheable,
    )


@Op.registry.register("LLMVisionOp")
class LLMVisionOp(LLMChatOp):
    """LLMChatOp variant that consumes an image input via embedding."""

    image_source: str
    image_path: str

    def __init__(
        self,
        messages: list[OpMessage] | Op,
        image_source: str,
        image_path: str = "images",
        image_source_op: Op | None = None,
        config: GenerationConfig | None = None,
        return_history: bool = False,
        rowwise_template: str | None = None,
        rowwise_columns: list[dict[str, Any]] | None = None,
        system_messages: list[str] | None = None,
        cacheable: bool = False,
    ) -> None:
        super().__init__(
            messages=messages,
            config=config,
            return_history=return_history,
            rowwise_template=rowwise_template,
            rowwise_columns=rowwise_columns,
            system_messages=system_messages,
            cacheable=cacheable,
        )
        if not image_source:
            raise ValueError("LLMVisionOp requires image_source")
        self.image_source = image_source
        self.image_path = image_path
        if image_source_op is not None:
            # Ensure the image source is part of the graph traversal.
            self.inputs.append(image_source_op)

    def _serialize(self) -> dict[str, Any]:
        payload = super()._serialize()
        payload["image_source"] = self.image_source
        payload["image_path"] = self.image_path
        return payload

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "LLMVisionOp":
        message = other_ops[data["messages"]]
        config = GenerationConfig(**data["config"])
        return_history = data["return_history"]
        cacheable = data["cacheable"]
        image_source = data.get("image_source", "")
        image_path = data.get("image_path", "images")
        rowwise_template = data.get("rowwise_template")
        rowwise_columns = data.get("rowwise_columns")
        system_messages = data.get("system_messages")
        image_source_op = other_ops.get(image_source)
        return cls(
            message,
            image_source=image_source,
            image_path=image_path,
            image_source_op=image_source_op,
            config=config,
            return_history=return_history,
            rowwise_template=rowwise_template,
            rowwise_columns=rowwise_columns,
            system_messages=system_messages,
            cacheable=cacheable,
        )

    def _state_signature(self) -> Hashable | None:
        return (
            super()._state_signature(),
            self.image_source,
            self.image_path,
        )


def llm_vision(
    messages: list[OpMessage] | Op,
    image_source: str,
    image_path: str = "images",
    image_source_op: Op | None = None,
    config: GenerationConfig | None = None,
    return_history: bool = False,
    rowwise_template: str | None = None,
    rowwise_columns: list[dict[str, Any]] | None = None,
    system_messages: list[str] | None = None,
    cacheable: bool = False,
) -> LLMVisionOp:
    return LLMVisionOp(
        messages,
        image_source=image_source,
        image_path=image_path,
        image_source_op=image_source_op,
        config=config,
        return_history=return_history,
        rowwise_template=rowwise_template,
        rowwise_columns=rowwise_columns,
        system_messages=system_messages,
        cacheable=cacheable,
    )


@Op.registry.register("ImageGenerationOp")
class ImageGenerationOp(LLMOp):
    """Op for generating images from text content."""

    def __init__(
        self,
        content: str | Op,
        config: GenerationConfig | None = None,
        cacheable: bool = False,
    ) -> None:
        content_op = content if isinstance(content, Op) else DataOp(data=[content])
        super().__init__([content_op], config, cacheable)

    @property
    def content(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        return dict(
            content=self.content.id,
            config=self.config.to_dict(),
            cacheable=self.cacheable,
        )

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "ImageGenerationOp":
        config = GenerationConfig(**data["config"])
        return cls(other_ops[data["content"]], config, data["cacheable"])

    def _state_signature(self) -> Hashable | None:
        return (tuple(self.config.to_dict().items()), self.cacheable)

    def get_prefix_template(
        self, input_templates: dict[str, PrefixType], sliced_op_map: dict[str, str]
    ) -> TextPrefixType:
        op_id = self.id
        op_id = sliced_op_map.get(op_id, op_id)
        return (Placeholder(op_id),)

    def get_input_prefix_template(
        self, input_templates: dict[str, PrefixType]
    ) -> TextPrefixType:
        return to_text_prefix(input_templates[self.content.id])

    def get_input_slice(self, data_size: int, input_slices: dict[str, Slice]) -> Slice:
        return input_slices[self.content.id]


def image_generation(
    content: str | Op,
    config: GenerationConfig | None = None,
    cacheable: bool = False,
) -> ImageGenerationOp:
    return ImageGenerationOp(content, config, cacheable)
