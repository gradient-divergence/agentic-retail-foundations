import pytest


def test_gateway_hashes_and_verifies_demo_passwords():
    from demos.api_gateway_demo import FAKE_AGENTS_DB, verify_password

    password_hash = FAKE_AGENTS_DB["inventory-agent-1"]["hashed_password"]
    assert verify_password("password123", password_hash)
    assert not verify_password("wrong-password", password_hash)


@pytest.mark.parametrize("password", ["x" * 73, "ü" * 37])
def test_gateway_rejects_passwords_over_bcrypt_byte_limit(password):
    from demos.api_gateway_demo import FAKE_AGENTS_DB, verify_password

    assert not verify_password(password, FAKE_AGENTS_DB["inventory-agent-1"]["hashed_password"])


def test_response_connection_exclusions_are_independent_of_request(monkeypatch):
    import asyncio

    import httpx
    from fastapi import BackgroundTasks, Request

    from demos import api_gateway_book_demo as demo

    async def respond(request):
        assert "x-request-only" not in request.headers
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b"ok"),
            headers={"Connection": "X-Response-Only", "X-Response-Only": "drop", "X-Request-Only": "keep"},
        )

    factory = httpx.AsyncClient
    monkeypatch.setattr(demo.httpx, "AsyncClient", lambda: factory(transport=httpx.MockTransport(respond)))
    request = Request(
        dict(
            type="http",
            method="GET",
            path="/items",
            query_string=b"",
            headers=[(b"connection", b"X-Request-Only"), (b"x-request-only", b"drop")],
        )
    )
    request._body = b""
    response = asyncio.run(
        demo.proxy_request(
            request,
            "product-service",
            "/items",
            demo.Agent(agent_id="trusted", agent_name="agent", roles=["read"]),
            BackgroundTasks(),
        )
    )
    assert response.headers["x-request-only"] == "keep"
    assert "x-response-only" not in response.headers
