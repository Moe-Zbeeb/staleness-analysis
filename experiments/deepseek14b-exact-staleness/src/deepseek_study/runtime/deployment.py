import json
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, model_validator


class RemoteInference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trainer_host: str
    router_url: str
    worker_urls: list[str]
    hardware: list[dict]

    @model_validator(mode="after")
    def validate_endpoints(self):
        if not self.trainer_host or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for c in self.trainer_host
        ):
            raise ValueError("Invalid trainer hostname")
        if len(set(self.worker_urls)) != len(self.worker_urls) or not self.worker_urls:
            raise ValueError("Inference endpoints must be unique and nonempty")
        for url in [self.router_url, *self.worker_urls]:
            parsed = urlparse(url)
            if parsed.scheme != "http" or not parsed.hostname or not parsed.port or parsed.path != "/v1":
                raise ValueError("Inference endpoints must be explicit HTTP host:port/v1 URLs")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Unexpected inference URL components")
        return self

    def validate_study(self, study):
        if study.inference_tensor_parallel != 1 or len(self.worker_urls) != study.inference_gpus:
            raise ValueError("Remote inference requires one registered endpoint per inference GPU")

    @classmethod
    def read(cls, path):
        return cls.model_validate(json.loads(Path(path).read_text()))
