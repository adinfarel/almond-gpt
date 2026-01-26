from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str = Field(..., examples="What is protein?")
    max_new_tokens: int = Field(default=50, ge=1, le=256)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
