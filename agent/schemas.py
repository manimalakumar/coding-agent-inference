"""Validated experiment inputs, with permissions controlled by the host."""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Budgets(StrictModel):
    seconds: int = Field(default=600, gt=0)
    model_calls: int = Field(default=12, gt=0)
    generated_tokens: int = Field(default=6000, gt=0)
    repair_rounds: int = Field(default=2, ge=0)
    tool_seconds: int = Field(default=90, gt=0)
    response_tokens: int = Field(default=1500, gt=0)


class Resource(StrictModel):
    name: str
    # Names resolve beneath a trusted resources directory; no arbitrary host mounts.
    resource: str
    target: str
    mode: Literal["ro", "rw"] = "ro"

    @model_validator(mode="after")
    def safe_mount(self) -> "Resource":
        source = PurePosixPath(self.resource)
        target = PurePosixPath(self.target)
        if source.is_absolute() or ".." in source.parts or not source.parts:
            raise ValueError("resource must be relative to trusted resources")
        if not target.is_absolute() or ".." in target.parts:
            raise ValueError("mount target must be absolute and normalized")
        if target.parts[:2] != ("/", "resources"):
            raise ValueError("additional resources must mount under /resources")
        if self.mode != "ro":
            raise ValueError("additional resources are read-only in schema v1")
        return self


class SandboxConfig(StrictModel):
    schema_version: Literal[1, 2, 3, 4] = 1
    test_policy: Literal["legacy", "read_only_existing_new_test_files"] = "legacy"
    stopping_protocol: Literal["legacy", "submit_priority_v1"] = "legacy"
    prompt_config: str = Field(
        default="mini-swe.yaml", pattern=r"^[a-zA-Z0-9_-]+\.yaml$"
    )
    workflow_label: str | None = None
    workflow_instructions: tuple[str, ...] = ()
    file_writing_tool: bool = False
    network: Literal["none"] = "none"
    python: Literal["3.14.6"] = "3.14.6"
    base_image: str = "python:3.14.6-slim-bookworm"
    cpus: float = Field(default=2, gt=0)
    memory_mb: int = Field(default=2048, ge=256)
    pids_limit: int = Field(default=128, ge=16)
    writable: tuple[str, ...]
    resources: tuple[Resource, ...] = ()
    budgets: Budgets = Budgets()
    tool_output_chars: int = Field(default=16000, gt=0)

    @model_validator(mode="after")
    def safe_paths(self) -> "SandboxConfig":
        seen: list[PurePosixPath] = []
        for value in self.writable:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("writable paths must be relative source directories")
            if any(p == ".git" or p.startswith(".git/") for p in path.parts):
                raise ValueError("Git metadata is prohibited")
            if any(
                path == old or path in old.parents or old in path.parents
                for old in seen
            ):
                raise ValueError("writable mounts must not overlap")
            seen.append(path)
        targets = [r.target for r in self.resources]
        if len(set(targets)) != len(targets):
            raise ValueError("duplicate resource mount target")
        return self


class ModelConfig(StrictModel):
    model: str = "qwen/qwen3-14b"
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    temperature: float = 0
    provider: dict = Field(default_factory=lambda: {"require_parameters": True})
    reasoning: dict | None = Field(default_factory=lambda: {"enabled": False})


class Check(StrictModel):
    name: str
    command: str


class Task(StrictModel):
    schema_version: Literal[1] = 1
    task_id: str
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    package: str
    instructions: str
    source_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    development_checks: tuple[Check, ...]
    reproduction_command: str = ""
    reproduction_script: str | None = None
    expected_failure: str = ""
    runtime_versions: dict[str, str] = Field(default_factory=dict)
    workspace_imports: tuple[str, ...] = ()

    @model_validator(mode="after")
    def safe_script(self) -> "Task":
        if self.reproduction_script is not None:
            path = PurePosixPath(self.reproduction_script)
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
                raise ValueError("reproduction script must be a task-local filename")
            if not path.name.endswith(".py"):
                raise ValueError("reproduction script must be a Python file")
            if self.reproduction_command != "python /task/" + path.name:
                raise ValueError("reproduction command must execute the mounted script")
        return self


class Submission(StrictModel):
    classification: Literal[
        "reproduced", "not_reproduced", "blocked_by_environment", "needs_clarification"
    ]
    explanation: str
