from dataclasses import dataclass
from typing import Any, ClassVar

from lumilake_server.common import GenerationConfig
from lumilake_server.ops.data_ops import DataOp, MessageOp, OpMessage
from lumilake_server.ops.ops import Op, RowShape


@dataclass(frozen=True)
class RowwiseMessages:
    """Locates one chat-message list per row at ``path`` under the chat op's
    producing ``rowwise_messages_op`` input; an absent path reads that
    producer's whole output."""

    path: str | None = None

    def __post_init__(self) -> None:
        if self.path is not None and (
            not isinstance(self.path, str) or not self.path.strip()
        ):
            raise ValueError("RowwiseMessages path must be a non-empty string")

    def to_dict(self, node: str) -> dict[str, str]:
        payload = {"node": node}
        if self.path is not None:
            payload["path"] = self.path
        return payload


def _refuse_fields_beside_rowwise_messages(
    op_kind: str,
    producer: Op,
    aggregate_table: list[dict[str, Any]] | None,
    rowwise_template: str | None,
    rowwise_columns: list[dict[str, Any]] | None,
    system_messages: list[str] | None,
    condition: dict[str, str] | None,
    return_history: bool,
) -> None:
    """Refuse a field the column would make moot: the column supplies each
    row's whole conversation."""
    beside: list[tuple[str, Any]] = [
        ("aggregate_table", aggregate_table),
        ("rowwise_template", rowwise_template),
        ("rowwise_columns", rowwise_columns),
        ("system_messages", system_messages),
        ("condition", condition),
        ("return_history", return_history or None),
    ]
    for field, value in beside:
        if value is None:
            continue
        raise ValueError(
            f"{op_kind} combines rowwise_messages with {field};"
            f" the column '{producer.id}' supplies the whole conversation, so"
            " the field beside it would be ignored. Drop it, or drop the"
            " messages column."
        )


class LLMOp(Op):
    config: GenerationConfig
    cacheable: bool
    # Whether this op validates its own output; only a text LLMChatOp does.
    validates_output: ClassVar[bool] = False

    def __init__(
        self,
        inputs: list["Op"] | None,
        config: GenerationConfig | None,
        cacheable: bool,
    ) -> None:
        super().__init__(inputs)
        self.config = config or GenerationConfig.from_env()
        self.cacheable = cacheable
        if self.config.output_enforcement is not None and not self.validates_output:
            raise ValueError(
                f"{type(self).__name__} does not support config.output_enforcement;"
                " use it only on LLMChatOp."
            )

    def _serialize(self) -> dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "LLMOp":
        raise NotImplementedError()


@Op.registry.register("LLMChatOp")
class LLMChatOp(LLMOp):
    validates_output: ClassVar[bool] = True
    config: GenerationConfig
    return_history: bool
    structural_outputs: dict[str, Any] | list[dict[str, Any]] | None
    aggregate_table: list[dict[str, Any]] | None
    rowwise_template: str | None
    rowwise_columns: list[dict[str, Any]] | None
    rowwise_messages: RowwiseMessages | None
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
        rowwise_messages: RowwiseMessages | None = None,
        rowwise_messages_op: Op | None = None,
        system_messages: list[str] | None = None,
        cacheable: bool = False,
        condition: dict[str, str] | None = None,
    ) -> None:
        messages = messages if isinstance(messages, Op) else MessageOp(messages)
        inputs: list[Op] = [messages]
        if rowwise_messages is not None:
            if rowwise_messages_op is None:
                raise ValueError(
                    f"{type(self).__name__} rowwise_messages requires the"
                    " producing op"
                )
            _refuse_fields_beside_rowwise_messages(
                type(self).__name__,
                rowwise_messages_op,
                aggregate_table,
                rowwise_template,
                rowwise_columns,
                system_messages,
                condition,
                return_history,
            )
            if not isinstance(messages, MessageOp) or messages.messages:
                raise ValueError(
                    f"{type(self).__name__} combines rowwise_messages with"
                    " messages; the column supplies each row's whole"
                    " conversation, so a message beside it would be ignored."
                )
            # Keep the column's producer in the graph traversal.
            inputs.append(rowwise_messages_op)
        super().__init__(inputs, config, cacheable)
        self.return_history = return_history
        self.structural_outputs = structural_outputs
        self.aggregate_table = aggregate_table
        self.rowwise_template = rowwise_template
        self.rowwise_columns = rowwise_columns
        self.rowwise_messages = rowwise_messages
        self.system_messages = system_messages
        self.condition = condition

    @property
    def messages(self) -> Op:
        return self.inputs[0]

    @property
    def rowwise_messages_op(self) -> Op | None:
        """The input the column reads each row's whole conversation from."""
        return self.inputs[1] if self.rowwise_messages is not None else None

    @property
    def row_shape(self) -> RowShape:
        """A row template groups a table's rows into one item; any other chat
        emits one value per row."""
        return RowShape.GROUPED if self.rowwise_template else RowShape.PER_ROW

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
        if self.rowwise_messages is not None:
            payload["rowwise_messages"] = self.rowwise_messages.to_dict(
                self.inputs[1].id
            )
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
        column_data = data.get("rowwise_messages")
        rowwise_messages: RowwiseMessages | None = None
        rowwise_messages_op: Op | None = None
        if column_data is not None:
            rowwise_messages = RowwiseMessages(column_data.get("path"))
            rowwise_messages_op = other_ops[column_data["node"]]
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
            rowwise_messages,
            rowwise_messages_op,
            system_messages,
            cacheable,
            condition,
        )


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


@Op.registry.register("LLMCompletionOp")
class LLMCompletionOp(LLMOp):
    """Generate from rendered string prompts without a chat template."""

    def __init__(
        self,
        prompt: str | list[str] | Op,
        config: GenerationConfig | None = None,
        cacheable: bool = False,
    ) -> None:
        prompt_op = (
            prompt if isinstance(prompt, Op) else DataOp(data=_as_prompt_list(prompt))
        )
        super().__init__([prompt_op], config, cacheable)
        if self.config.api is not None:
            raise ValueError("LLMCompletionOp does not support API mode")
        if self.config.chat_template_kwargs is not None:
            raise ValueError(
                "LLMCompletionOp does not apply the chat template, so it"
                " does not support chat_template_kwargs"
            )

    @property
    def prompt(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt.id,
            "config": self.config.to_dict(),
            "cacheable": self.cacheable,
        }

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "LLMCompletionOp":
        return cls(
            prompt=other_ops[data["prompt"]],
            config=GenerationConfig(**data["config"]),
            cacheable=data["cacheable"],
        )


def _as_prompt_list(prompt: str | list[str]) -> list[str]:
    values = [prompt] if isinstance(prompt, str) else list(prompt)
    if not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError("LLMCompletionOp prompt must contain non-empty strings")
    return values


def llm_completion(
    prompt: str | list[str] | Op,
    config: GenerationConfig | None = None,
    cacheable: bool = False,
) -> LLMCompletionOp:
    """Run rendered string prompts through raw local generation."""
    return LLMCompletionOp(prompt=prompt, config=config, cacheable=cacheable)


@Op.registry.register("LLMVisionOp")
class LLMVisionOp(LLMChatOp):
    """LLMChatOp variant that consumes an image input via embedding."""

    validates_output: ClassVar[bool] = False

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


def image_generation(
    content: str | Op,
    config: GenerationConfig | None = None,
    cacheable: bool = False,
) -> ImageGenerationOp:
    return ImageGenerationOp(content, config, cacheable)
