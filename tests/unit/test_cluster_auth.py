"""
Unit tests for Secure Cluster Authentication (HMAC-SHA256 worker join).
Tests cover token lifecycle, HMAC verification, clock skew, replay
protection, revocation, and audit logging.
"""
import time
from backend.security.cluster_auth import ClusterAuthManager, JoinToken


def test_create_join_token():
    """Join token creation returns a valid, non-expired token."""
    auth = ClusterAuthManager(master_secret="test-secret", require_auth=True)
    token = auth.create_join_token(ttl_seconds=3600)
    assert isinstance(token, str)
    assert len(token) > 20
    # Token should be tracked internally
    assert token in auth._tokens
    assert auth._tokens[token].is_valid


def test_hmac_signature_verification():
    """Valid HMAC signature authorizes the worker."""
    auth = ClusterAuthManager(master_secret="test-secret", require_auth=True)
    token = auth.create_join_token(ttl_seconds=3600)
    worker_id = "worker-test-1"
    ts = time.time()
    sig = auth.compute_hmac(worker_id, token, ts)

    result = auth.verify_worker_join(
        worker_id=worker_id,
        token=token,
        signature=sig,
        timestamp=ts,
        source_address="10.0.0.5:50052",
    )
    assert result["authorized"] is True
    assert result["reason"] == "verified"
    assert auth.is_worker_authorized(worker_id)


def test_invalid_signature_rejected():
    """Invalid HMAC signature is rejected."""
    auth = ClusterAuthManager(master_secret="test-secret", require_auth=True)
    token = auth.create_join_token(ttl_seconds=3600)
    ts = time.time()

    result = auth.verify_worker_join(
        worker_id="worker-bad",
        token=token,
        signature="invalid_signature_hex",
        timestamp=ts,
    )
    assert result["authorized"] is False
    assert result["reason"] == "invalid_signature"


def test_expired_token_rejected():
    """Expired join token is rejected."""
    auth = ClusterAuthManager(master_secret="test-secret", require_auth=True)
    token = auth.create_join_token(ttl_seconds=0.001)  # Expires almost immediately
    time.sleep(0.01)  # Wait for expiry

    ts = time.time()
    sig = auth.compute_hmac("worker-x", token, ts)
    result = auth.verify_worker_join(
        worker_id="worker-x",
        token=token,
        signature=sig,
        timestamp=ts,
    )
    assert result["authorized"] is False
    assert result["reason"] == "token_expired"


def test_single_use_token():
    """Join token can only be used once (replay protection)."""
    auth = ClusterAuthManager(master_secret="test-secret", require_auth=True)
    token = auth.create_join_token(ttl_seconds=3600)
    ts = time.time()

    # First use succeeds
    sig1 = auth.compute_hmac("worker-1", token, ts)
    r1 = auth.verify_worker_join("worker-1", token, sig1, ts)
    assert r1["authorized"] is True

    # Second use with same token fails
    sig2 = auth.compute_hmac("worker-2", token, ts)
    r2 = auth.verify_worker_join("worker-2", token, sig2, ts)
    assert r2["authorized"] is False
    assert r2["reason"] == "token_already_used"


def test_unknown_token_rejected():
    """Completely unknown token is rejected."""
    auth = ClusterAuthManager(master_secret="test-secret", require_auth=True)
    ts = time.time()
    sig = auth.compute_hmac("worker-x", "fake-token-abc", ts)
    result = auth.verify_worker_join("worker-x", "fake-token-abc", sig, ts)
    assert result["authorized"] is False
    assert result["reason"] == "unknown_token"


def test_clock_skew_rejection():
    """Requests with excessive clock skew are rejected."""
    auth = ClusterAuthManager(master_secret="test-secret", require_auth=True)
    token = auth.create_join_token(ttl_seconds=3600)
    old_ts = time.time() - 600  # 10 minutes in the past
    sig = auth.compute_hmac("worker-x", token, old_ts)

    result = auth.verify_worker_join(
        worker_id="worker-x",
        token=token,
        signature=sig,
        timestamp=old_ts,
        clock_skew_tolerance=300.0,  # 5 minutes
    )
    assert result["authorized"] is False
    assert result["reason"] == "clock_skew"


def test_token_revocation():
    """Revoked token cannot be used to join."""
    auth = ClusterAuthManager(master_secret="test-secret", require_auth=True)
    token = auth.create_join_token(ttl_seconds=3600)
    assert auth.revoke_token(token) is True

    ts = time.time()
    sig = auth.compute_hmac("worker-x", token, ts)
    result = auth.verify_worker_join("worker-x", token, sig, ts)
    assert result["authorized"] is False
    assert result["reason"] == "token_already_used"


def test_auth_disabled_bypasses_checks():
    """When require_auth=False, all workers are auto-authorized."""
    auth = ClusterAuthManager(require_auth=False)
    result = auth.verify_worker_join("any-worker", "any-token", "any-sig", time.time())
    assert result["authorized"] is True
    assert result["reason"] == "auth_disabled"
    assert auth.is_worker_authorized("any-worker")


def test_audit_log_records_events():
    """Audit log correctly records join and rejection events."""
    auth = ClusterAuthManager(master_secret="test-secret", require_auth=True)
    token = auth.create_join_token()
    ts = time.time()

    # Attempt with bad sig (rejected)
    auth.verify_worker_join("bad-worker", token, "wrong-sig", ts)
    # Attempt with correct sig (accepted)
    sig = auth.compute_hmac("good-worker", token, ts)
    auth.verify_worker_join("good-worker", token, sig, ts)

    log = auth.get_audit_log()
    event_types = [e["event_type"] for e in log]
    assert "token_created" in event_types
    assert "join_rejected" in event_types
    assert "join_accepted" in event_types


def test_deauthorize_worker():
    """Deauthorized workers lose their authorized status."""
    auth = ClusterAuthManager(master_secret="test-secret", require_auth=True)
    token = auth.create_join_token()
    ts = time.time()
    sig = auth.compute_hmac("worker-to-evict", token, ts)
    auth.verify_worker_join("worker-to-evict", token, sig, ts)
    assert auth.is_worker_authorized("worker-to-evict")

    auth.deauthorize_worker("worker-to-evict")
    assert not auth.is_worker_authorized("worker-to-evict")


def test_auth_stats():
    """Stats reflect current authentication system state."""
    auth = ClusterAuthManager(master_secret="test-secret", require_auth=True)
    auth.create_join_token()
    auth.create_join_token()
    stats = auth.stats()
    assert stats["active_tokens"] == 2
    assert stats["require_auth"] is True
    assert stats["authorized_workers"] == 0
