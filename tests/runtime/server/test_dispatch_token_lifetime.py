from support.runtime_server import make_server


def test_release_request_workflows_clears_dispatch_token() -> None:
    """The secret-lifetime guarantee: once a request's workflows are released
    (the completion path), the dispatch token and API credential must be gone
    from the in-process store. A credential that outlives its request is the
    bug this mechanism exists to remove."""
    server = make_server()
    server.runtime_manager.set_dispatch_token("req-1", "runtime-token")
    server.runtime_manager.set_api_credential("req-1", "Bearer caller-key")

    assert server.runtime_manager.get_dispatch_token("req-1") == "runtime-token"
    assert server.runtime_manager.get_api_credential("req-1") == "Bearer caller-key"

    server.release_request_workflows("req-1")

    assert server.runtime_manager.get_dispatch_token("req-1") is None
    assert server.runtime_manager.get_api_credential("req-1") is None
