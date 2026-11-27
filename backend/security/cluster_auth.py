"""
Secure cluster authentication and worker join authorization.
Implements HMAC-SHA256 token-based mutual authentication between Master
Controller and Worker nodes, plus join-token lifecycle management.

Features:
- Cluster-wide shared secret derivation from a configurable master key.
- Time-limited join tokens with configurable TTL and single-use enforcement.
- HMAC challenge-response verification for worker registration handshake.
- Token revocation and audit log of join/reject events.

Maintained by the Distributed Memory Grid team.
"""
import hashlib
import hmac
import secrets
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger("DMG-ClusterAuth")


@dataclass
class JoinToken:
    """Represents a time-limited, single-use token that authorizes a worker to join."""
    token: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    used: bool = False
    used_by: Optional[str] = None
    used_at: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def is_valid(self) -> bool:
        return not self.used and not self.is_expired


@dataclass
class AuthEvent:
    """Audit log entry for join/reject events."""
    timestamp: float
    event_type: str  # "join_accepted", "join_rejected", "token_created", "token_revoked"
    worker_id: Optional[str] = None
    token_prefix: Optional[str] = None  # First 8 chars for audit without exposing full token
    reason: Optional[str] = None
    source_address: Optional[str] = None


class ClusterAuthManager:
    """
    Manages HMAC-based cluster authentication.

    Workflow:
    1. Admin generates a join token via `create_join_token()`.
    2. Worker presents join token + its worker_id to `verify_worker_join()`.
    3. Master verifies HMAC signature and token validity.
    4. On success, worker is authorized and token is marked as used.
    """

    def __init__(
        self,
        master_secret: str = "",
        token_ttl_seconds: float = 3600.0,
        max_active_tokens: int = 100,
        require_auth: bool = True,
    ):
        """
        Args:
            master_secret: Cluster-wide secret key. If empty, one is auto-generated.
            token_ttl_seconds: Default lifetime for newly created join tokens.
            max_active_tokens: Maximum number of active (unused, unexpired) tokens.
            require_auth: If False, authentication is bypassed (dev/test mode).
        """
        self.master_secret = master_secret or secrets.token_hex(32)
        self.token_ttl_seconds = token_ttl_seconds
        self.max_active_tokens = max_active_tokens
        self.require_auth = require_auth

        self._tokens: Dict[str, JoinToken] = {}
        self._audit_log: List[AuthEvent] = []
        self._authorized_workers: Dict[str, float] = {}  # worker_id -> authorized_at

    def _derive_key(self, context: str = "cluster-join") -> bytes:
        """Derive an HMAC signing key from the master secret and a context string."""
        return hashlib.sha256(
            f"{self.master_secret}:{context}".encode("utf-8")
        ).digest()

    def create_join_token(self, ttl_seconds: Optional[float] = None) -> str:
        """
        Generate a new cryptographically random join token.

        Returns:
            The raw token string to be distributed to the worker out-of-band.
        """
        # Enforce active token limit
        active_count = sum(1 for t in self._tokens.values() if t.is_valid)
        if active_count >= self.max_active_tokens:
            self._cleanup_expired_tokens()
            active_count = sum(1 for t in self._tokens.values() if t.is_valid)
            if active_count >= self.max_active_tokens:
                raise RuntimeError(
                    f"Maximum active token limit ({self.max_active_tokens}) reached."
                )

        ttl = ttl_seconds if ttl_seconds is not None else self.token_ttl_seconds
        raw_token = secrets.token_urlsafe(32)
        now = time.time()

        join_token = JoinToken(
            token=raw_token,
            created_at=now,
            expires_at=now + ttl,
        )
        self._tokens[raw_token] = join_token

        self._audit_log.append(AuthEvent(
            timestamp=now,
            event_type="token_created",
            token_prefix=raw_token[:8],
        ))
        logger.info(f"Join token created (prefix: {raw_token[:8]}..., TTL: {ttl}s)")
        return raw_token

    def compute_hmac(self, worker_id: str, token: str, timestamp: float) -> str:
        """
        Compute HMAC-SHA256 signature for a worker join request.
        The worker computes this on its side and sends it along with the request.
        """
        key = self._derive_key("cluster-join")
        message = f"{worker_id}:{token}:{timestamp}".encode("utf-8")
        return hmac.new(key, message, hashlib.sha256).hexdigest()

    def verify_worker_join(
        self,
        worker_id: str,
        token: str,
        signature: str,
        timestamp: float,
        source_address: str = "unknown",
        clock_skew_tolerance: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Verify a worker's join request using HMAC challenge-response.

        Args:
            worker_id: The ID the worker wants to register as.
            token: The join token presented by the worker.
            signature: HMAC-SHA256 signature computed by the worker.
            timestamp: Unix timestamp from the worker's request.
            source_address: Network address of the requesting worker.
            clock_skew_tolerance: Maximum allowed clock difference in seconds.

        Returns:
            Dict with 'authorized' (bool) and 'reason' (str).
        """
        if not self.require_auth:
            self._authorized_workers[worker_id] = time.time()
            return {"authorized": True, "reason": "auth_disabled"}

        now = time.time()

        # 1. Clock skew check
        if abs(now - timestamp) > clock_skew_tolerance:
            self._record_rejection(worker_id, token, source_address, "clock_skew")
            return {"authorized": False, "reason": "clock_skew"}

        # 2. Token existence and validity
        join_token = self._tokens.get(token)
        if join_token is None:
            self._record_rejection(worker_id, token, source_address, "unknown_token")
            return {"authorized": False, "reason": "unknown_token"}

        if join_token.used:
            self._record_rejection(worker_id, token, source_address, "token_already_used")
            return {"authorized": False, "reason": "token_already_used"}

        if join_token.is_expired:
            self._record_rejection(worker_id, token, source_address, "token_expired")
            return {"authorized": False, "reason": "token_expired"}

        # 3. HMAC signature verification
        expected_sig = self.compute_hmac(worker_id, token, timestamp)
        if not hmac.compare_digest(signature, expected_sig):
            self._record_rejection(worker_id, token, source_address, "invalid_signature")
            return {"authorized": False, "reason": "invalid_signature"}

        # 4. All checks passed — authorize and consume token
        join_token.used = True
        join_token.used_by = worker_id
        join_token.used_at = now
        self._authorized_workers[worker_id] = now

        self._audit_log.append(AuthEvent(
            timestamp=now,
            event_type="join_accepted",
            worker_id=worker_id,
            token_prefix=token[:8],
            source_address=source_address,
        ))
        logger.info(
            f"Worker '{worker_id}' authorized from {source_address} "
            f"(token prefix: {token[:8]}...)"
        )
        return {"authorized": True, "reason": "verified"}

    def is_worker_authorized(self, worker_id: str) -> bool:
        """Check if a worker has been previously authorized."""
        if not self.require_auth:
            return True
        return worker_id in self._authorized_workers

    def revoke_token(self, token: str) -> bool:
        """Revoke an active join token so it can no longer be used."""
        join_token = self._tokens.get(token)
        if join_token is None:
            return False
        join_token.used = True
        join_token.used_by = "__revoked__"
        join_token.used_at = time.time()

        self._audit_log.append(AuthEvent(
            timestamp=time.time(),
            event_type="token_revoked",
            token_prefix=token[:8],
        ))
        logger.info(f"Join token revoked (prefix: {token[:8]}...)")
        return True

    def deauthorize_worker(self, worker_id: str) -> bool:
        """Remove a worker's authorization (e.g., after eviction)."""
        if worker_id in self._authorized_workers:
            del self._authorized_workers[worker_id]
            logger.info(f"Worker '{worker_id}' deauthorized.")
            return True
        return False

    def _record_rejection(
        self,
        worker_id: str,
        token: str,
        source_address: str,
        reason: str,
    ):
        """Log a rejected join attempt."""
        self._audit_log.append(AuthEvent(
            timestamp=time.time(),
            event_type="join_rejected",
            worker_id=worker_id,
            token_prefix=token[:8] if token else None,
            reason=reason,
            source_address=source_address,
        ))
        logger.warning(
            f"Join REJECTED for worker '{worker_id}' from {source_address}: {reason}"
        )

    def _cleanup_expired_tokens(self):
        """Remove expired and used tokens from the store."""
        to_remove = [
            k for k, v in self._tokens.items()
            if v.is_expired or v.used
        ]
        for k in to_remove:
            del self._tokens[k]

    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent audit events."""
        entries = list(reversed(self._audit_log))[:limit]
        return [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "worker_id": e.worker_id,
                "token_prefix": e.token_prefix,
                "reason": e.reason,
                "source_address": e.source_address,
            }
            for e in entries
        ]

    def stats(self) -> Dict[str, Any]:
        """Return authentication system statistics."""
        active_tokens = sum(1 for t in self._tokens.values() if t.is_valid)
        used_tokens = sum(1 for t in self._tokens.values() if t.used)
        expired_tokens = sum(
            1 for t in self._tokens.values() if t.is_expired and not t.used
        )
        return {
            "require_auth": self.require_auth,
            "active_tokens": active_tokens,
            "used_tokens": used_tokens,
            "expired_tokens": expired_tokens,
            "authorized_workers": len(self._authorized_workers),
            "total_audit_events": len(self._audit_log),
        }
