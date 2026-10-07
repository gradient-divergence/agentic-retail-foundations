def test_gateway_hashes_and_verifies_demo_passwords():
    from demos.api_gateway_demo import FAKE_AGENTS_DB, pwd_context

    password_hash = FAKE_AGENTS_DB["inventory-agent-1"]["hashed_password"]
    assert pwd_context.verify("password123", password_hash)
    assert not pwd_context.verify("wrong-password", password_hash)
