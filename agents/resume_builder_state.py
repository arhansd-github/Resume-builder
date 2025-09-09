from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime, timezone

from pydantic import Field

from pyagenity.state import AgentState
from pyagenity.utils import Message
from pyagenity.utils.constants import START
from pyagenity.state.execution_state import ExecutionState as ExecMeta


class ResumeBuilderState(AgentState):
    """Custom state container for the resume builder."""
    jd_summary: Optional[str] = None
    resume_sections: Dict[str, Any] = None
    section_objects: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    current_section: Optional[str] = None
    section_done: Dict[str, bool] = Field(default_factory=lambda: {s: False for s in ["skills", "experiences", "education", "projects", "summary", "contact", "certificates", "publications", "languages", "recommendations", "custom"]})
    context: List[Message] = Field(default_factory=list)
    context_summary: Optional[str] = None
    execution_meta: ExecMeta = Field(default_factory=lambda: ExecMeta(current_node=START))
    recommended_answers: Dict[str, List[str]] = Field(default_factory=dict)  # Stores answers to recommended questions by section
    
    # NEW: Add routing attempt counter to prevent infinite loops
    routing_attempts: int = Field(default=0)

    def make_message(self, role: str, content: str) -> Message:
        msg_dict = {
            "message_id": str(uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            return Message(**msg_dict)
        except Exception as e:  # pragma: no cover - defensive
            print(f"Warning: Failed to create Message: {e}. Using fallback object.")
            m = Message.__new__(Message)
            for k, v in msg_dict.items():
                setattr(m, k, v)
            return m