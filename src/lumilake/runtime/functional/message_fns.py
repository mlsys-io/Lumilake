from collections.abc import AsyncGenerator

from lumilake.runtime.data import Data, DataType, MessageData, MessageList
from lumilake.runtime.functional.fns import FnInput


class MessageFnInput(FnInput):
    NAME: str = "message"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        roles: list[str],
        message_refs: list[int | str],
        inputs: list[Data],
    ):
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=inputs,
            kwargs={},
            task_id=op_id,
        )
        self.roles = roles
        self.message_refs = message_refs

    @property
    def output_type(self) -> DataType:
        return DataType.MESSAGE

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        if len(self.roles) != len(self.message_refs):
            raise ValueError("Inconsistent number of roles and messages")

        # Find reference args and message data to broadcast.
        messages: list[MessageData] = []
        to_broadcast: list[MessageData] = []
        ref_num_contents, ref_arg = 0, None
        for role, ref in zip(self.roles, self.message_refs):
            if isinstance(ref, int):
                arg = self.args[ref]
                message_content = arg.as_text()
                if ref_arg is None:
                    # Set reference arg.
                    ref_num_contents = len(message_content)
                    ref_arg = arg
                elif len(message_content) != ref_num_contents:
                    # Consistency check.
                    raise ValueError("Inconsistent number of message contents")
                message = MessageData(role=role, content=message_content)
            else:
                # Broadcastable message data.
                message = MessageData(role=role, content=[ref])
                to_broadcast.append(message)
            messages.append(message)

        if ref_arg is None:
            raise ValueError("There must be at least one message with data.")

        ref_indices = ref_arg.indices
        # Broadcast the message contents of broadcastable messages.
        for message in to_broadcast:
            message.content *= ref_num_contents
        # Check consistency of indices.
        for arg in self.args:
            if arg.indices != ref_indices:
                raise ValueError("Inconsistent data ordering")
        yield Data.message(MessageList(messages), ref_indices)


class AppendMessageFnInput(FnInput):
    NAME: str = "append-message"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        messages: Data,
        content: str | Data,
        role: str,
    ) -> None:
        self.role = role
        self.content_str: str | None
        args = [messages]
        if isinstance(content, str):
            self.content_str = content
        else:
            self.content_str = None
            args.append(content)
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=args,
            kwargs={},
            task_id=op_id,
        )

    @property
    def output_type(self) -> DataType:
        return DataType.MESSAGE

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        # Separate text vs Data branches with distinct locals for mypy
        if self.content_str is None:
            messages_arg = self.args[0]
            content_data: Data = self.args[1]
            content_str: str | None = None
        else:
            messages_arg = self.args[0]
            content_data = None  # type: ignore[assignment]
            content_str = self.content_str

        messages = messages_arg.as_message().copy()

        if content_str is not None:
            role = self.role
            new_message_content = [content_str]
        else:
            # content_data is a Data instance here
            assert content_data is not None
            if content_data.is_text():
                role = self.role
                new_message_content = content_data.as_text()
            else:
                assert content_data.is_message()
                if len(content_data) != 1:
                    raise ValueError("Content must be a single message.")
                new_message = content_data.as_message().get(0)
                assert new_message is not None
                role = new_message.role if self.role is None else self.role
                new_message_content = new_message.content

        # This broadcasts the message content internally.
        messages.append(MessageData(role=role, content=new_message_content))

        yield messages_arg.into_message(messages)


class LastMessageFnInput(FnInput):
    NAME: str = "last-message"

    def __init__(
        self,
        request_id: str,
        op_id: str,
        is_eager: bool,
        ref_count: int,
        max_iter: int | None,
        messages: Data,
    ) -> None:
        super().__init__(
            request_id=request_id,
            op_id=op_id,
            is_eager=is_eager,
            ref_count=ref_count,
            max_iter=max_iter,
            args=[messages],
            kwargs={},
            task_id=op_id,
        )

    @property
    def output_type(self) -> DataType:
        return DataType.TEXT

    async def _run(self) -> AsyncGenerator[Data | None, None]:
        data = self.args[0]
        messages = data.as_message()
        last_message = messages.get(-1)
        if last_message is None:
            yield data.into_empty(DataType.TEXT)
        else:
            yield data.into_text(last_message.content)
