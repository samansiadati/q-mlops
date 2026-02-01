# qmlops/core/quantum_run.py
from dataclasses import dataclass, asdict
from typing import Dict, Any
import uuid
import time

@dataclass
class QuantumRun:
    run_id: str
    backend: str
    model_type: str
    qubits: int
    circuit_depth: int
    shots: int
    seed: int
    metadata: Dict[str, Any]

    @classmethod
    def create(cls, **kwargs):
        return cls(
            run_id=str(uuid.uuid4())[:8],
            metadata={
                "created_at": time.time(),
                "status": "initialized"
            },
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
