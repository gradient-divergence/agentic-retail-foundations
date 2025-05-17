import pytest
from models.messaging import AgentMessage, Performative


def test_to_dict_and_from_dict_roundtrip():
    msg = AgentMessage(
        performative=Performative.REQUEST,
        sender="A",
        receiver="B",
        content={"foo": 1},
    )
    data = msg.to_dict()
    reconstructed = AgentMessage.from_dict(data)
    assert reconstructed.performative == msg.performative
    assert reconstructed.sender == "A"
    assert reconstructed.receiver == "B"
    assert reconstructed.content == {"foo": 1}
    assert reconstructed.conversation_id == msg.conversation_id
    assert reconstructed.message_id == msg.message_id


def test_create_reply():
    msg = AgentMessage(
        performative=Performative.REQUEST,
        sender="agent1",
        receiver="agent2",
        content="ping",
    )
    reply = msg.create_reply(Performative.ACCEPT, content="pong")
    assert reply.sender == "agent2"
    assert reply.receiver == "agent1"
    assert reply.conversation_id == msg.conversation_id
    assert reply.in_reply_to == msg.message_id
    assert reply.performative == Performative.ACCEPT
    assert reply.content == "pong"


def test_from_dict_missing_field():
    data = {
        "performative": Performative.INFORM.value,
        "sender": "a",
        # receiver missing
        "content": "hi",
        "conversation_id": "1",
        "message_id": "2",
        "timestamp": "2024-01-01T00:00:00",
    }
    with pytest.raises(ValueError):
        AgentMessage.from_dict(data)
