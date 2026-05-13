from lumid_hooks import PrincipalContext

from lumilake_hook import UsageRow

TOKENS: dict[str, PrincipalContext] = {
    "demo-admin": PrincipalContext(
        principal_id="alice",
        org_id="demo",
        external_id="alice@example.com",
        principal_type="admin",
        scopes=["admin", "data"],
    ),
    "demo-user": PrincipalContext(
        principal_id="bob",
        org_id="demo",
        external_id="bob@example.com",
        principal_type="user",
        scopes=["user", "data"],
    ),
}

BLOCKED_PRINCIPALS: set[str] = set()
OWNERSHIP: dict[tuple[str, str], str] = {}
USAGE_LEDGER: list[UsageRow] = []
