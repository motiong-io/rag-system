# aiwrappifymodels

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field

from .utils import CustomDateTime, PageParam


class ChatRoleEnum(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"


class MessageForLLM(BaseModel):
    role: ChatRoleEnum
    content: str
    # name: Optional[str] = None

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)  # 调用父类的 model_dump
        data["role"] = data["role"].value  # 将 role 转换为字符串

        return data


class MessageCreate(BaseModel):
    content: str
    digest_names: list[str] = Field(default_factory=list)

    @property
    def digest_ids(self) -> list[str]:
        return [digest_name.split("/")[-1] for digest_name in self.digest_names]


class MessagePageParam(PageParam):
    pass


class Message(BaseModel):
    id: uuid.UUID
    interaction_id: uuid.UUID
    content: str
    digest_ids: list[uuid.UUID]
    create_time: CustomDateTime
    creator: str

    @computed_field
    @property
    def name(self) -> str:
        return f"interactions/{self.interaction_id}/messages/{self.id}"

    @computed_field
    @property
    def digest_names(self) -> list[str]:
        return [f"digests/{digest_id}" for digest_id in self.digest_ids]
