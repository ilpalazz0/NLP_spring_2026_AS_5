from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
