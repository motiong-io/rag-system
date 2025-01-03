from pydantic import BaseModel
from aiwrappifymodels.message import ChatRoleEnum


class MessageForLLM(BaseModel):
    role: ChatRoleEnum
    content: str
    # name: Optional[str] = None

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)  # 调用父类的 model_dump
        data["role"] = data["role"].value  # 将 role 转换为字符串

        return data