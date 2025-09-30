from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FedHooks:
    def on_client_init(self, client_id: int, model: Any) -> None:
        pass

    def on_post_update(self, client_id: int, step: int, model: Any, metrics: Dict[str, float]) -> None:
        pass

    def should_sync(self, global_step: int) -> bool:
        return False

    def pull_global(self) -> Optional[Dict[str, Any]]:
        return None

    def push_client(self, client_id: int, payload: Dict[str, Any]) -> None:
        pass
