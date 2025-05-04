"""
Data models for agent communication messages.
"""

from enum import Enum
from typing import Any, Optional
from datetime import datetime
import uuid


class Performative(Enum):
    """
    Standard performatives for agent messages (FIPA-inspired).
    """

    INFORM = "inform"
    REQUEST = "request"
    QUERY = "query"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"
    SUBSCRIBE = "subscribe"
    # Add others as needed, e.g., CFP, REFUSE, FAILURE, NOT_UNDERSTOOD


class AgentMessage:
    """
    Structured message class with FIPA-like fields for agent communication.
    Uses standard Python dataclass features if possible, or manual __init__.
    """

    def __init__(
        self,
        performative: Performative,
        sender: str,
        receiver: str,
        content: Any,
        ontology: str = "retail-general",
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,  # Added explicit message ID
        reply_with: Optional[str] = None,
        in_reply_to: Optional[str] = None,
        timestamp: Optional[datetime] = None,  # Allow passing timestamp
    ):
        if not isinstance(performative, Performative):
            raise TypeError(
                f"performative must be a Performative enum member, not {type(performative)}"
            )

        self.performative = performative
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.ontology = ontology
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.message_id = message_id or str(
            uuid.uuid4()
        )  # Generate unique ID if not provided
        self.timestamp = timestamp or datetime.now()  # Use provided or generate now
        self.reply_with = reply_with  # Identifier for expected replies
        self.in_reply_to = in_reply_to  # Correlates reply to a previous message_id

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the message to a dictionary representation.
        """
        return {
            "performative": self.performative.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,  # Assumes content is serializable
            "ontology": self.ontology,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "reply_with": self.reply_with,
            "in_reply_to": self.in_reply_to,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMessage":
        """
        Create an AgentMessage from a dictionary.
        Performs basic validation.
        """
        # Validate required fields
        required_fields = [
            "performative",
            "sender",
            "receiver",
            "content",
            "conversation_id",
            "message_id",
            "timestamp",
        ]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field in message data: {field}")

        try:
            performative = Performative(data["performative"])
        except ValueError:
            raise ValueError(f"Invalid performative value: {data['performative']}")

        try:
            timestamp = datetime.fromisoformat(data["timestamp"])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid timestamp format: {data['timestamp']}")

        return cls(
            performative=performative,
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            ontology=data.get(
                "ontology", "retail-general"
            ),  # Provide default if missing
            conversation_id=data["conversation_id"],
            message_id=data["message_id"],
            timestamp=timestamp,
            reply_with=data.get("reply_with"),
            in_reply_to=data.get("in_reply_to"),
        )

    def create_reply(
        self,
        performative: Performative,
        content: Any,
        sender: Optional[str] = None,  # Allow overriding sender if needed
    ) -> "AgentMessage":
        """
        Create a reply message directed back to the original sender.
        Sets conversation_id and in_reply_to fields appropriately.
        """
        if not self.sender:
            raise ValueError("Cannot create reply, original sender is unknown.")

        reply_sender = (
            sender if sender is not None else self.receiver
        )  # The recipient of the original msg replies
        if not reply_sender:
            raise ValueError("Cannot determine sender for the reply message.")

        # The new message ID for the reply itself
        reply_message_id = str(uuid.uuid4())

        # The 'reply_with' field of the original message becomes the 'in_reply_to' of the reply
        # If 'reply_with' wasn't set, use the original message's ID as a fallback for correlation
        in_reply_to_id = self.reply_with or self.message_id

        return AgentMessage(
            performative=performative,
            sender=reply_sender,
            receiver=self.sender,  # Reply goes to the original sender
            content=content,
            ontology=self.ontology,
            conversation_id=self.conversation_id,  # Keep the same conversation
            message_id=reply_message_id,
            in_reply_to=in_reply_to_id,  # Link to the original message (or its reply tag)
            # reply_with can be set if further replies are expected to this reply
        )

    def __repr__(self) -> str:
        return (
            f"AgentMessage(performative={self.performative.name}, "
            f"sender='{self.sender}', receiver='{self.receiver}', "
            f"conv_id='{self.conversation_id[:8]}...', msg_id='{self.message_id[:8]}...', "
            f"in_reply_to='{self.in_reply_to[:8] if self.in_reply_to else None}...', "
            f"content={type(self.content).__name__})"
        )
