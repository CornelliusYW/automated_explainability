import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Any

@dataclass
class ExplainabilityReport:
    timestamp: datetime
    model_version: str
    global_feature_importance: Dict[str, float]
    local_explanations: Dict[int, Dict[str, float]] = field(default_factory=dict)
    summary_explanation: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)
