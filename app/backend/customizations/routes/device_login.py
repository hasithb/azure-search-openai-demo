"""CUSTOM: Device code login flow for email-first authentication.

Users enter their email on the splash screen, then receive a device code
to enter at microsoft.com/devicelogin. The backend manages the MSAL device
code flow and polls for completion. A session token is issued on success
so the browser is remembered across visits without re-authentication.

Security notes (reviewed against RFC 8628 and Microsoft Entra docs):
- Per-IP rate limiting on /start to prevent brute-force and DoS (RFC 8628 §5.1)
- Bounded in-memory stores with TTL-based eviction (OWASP A05)
- Uniform error responses to prevent flow enumeration
- Public client (no client secret) per RFC 8628 §5.6
- Tenanted authority (not /common) per Microsoft device code requirements
"""

import asyncio
import logging
import secrets
import time
import threading

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from msal import PublicClientApplication
from quart import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

device_login_bp = Blueprint("device_login", __name__)

# --- Bounded stores with TTL ---
# Maximum limits to prevent memory exhaustion (OWASP A05)
MAX_PENDING_FLOWS = 100
MAX_COMPLETED_FLOWS = 200
MAX_SESSIONS = 500
FLOW_TTL_SECONDS = 900  # 15 min (matches Entra device code lifetime)
COMPLETED_FLOW_TTL_SECONDS = 300  # 5 min for frontend to pick up result
SESSION_TTL_SECONDS = 90 * 24 * 3600  # 90 days (refresh tokens last ~90 days)

_pending_flows: OrderedDict[str, dict] = OrderedDict()
_completed_flows: OrderedDict[str, dict] = OrderedDict()
_sessions: OrderedDict[str, dict] = OrderedDict()
_store_lock = threading.Lock()

# Per-IP rate limiting for /start endpoint (RFC 8628 §5.1)
_rate_limit: dict[str, list[float]] = {}  # ip -> list of timestamps
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 5  # max /start requests per IP per window

_executor = ThreadPoolExecutor(max_workers=4)

# Module-level config set by configure()
_config: dict = {}


def _cleanup_stale_entries():
    """Remove expired entries from all in-memory stores."""
    now = time.time()
    with _store_lock:
        # Clean pending flows older than TTL
        stale_pending = [fid for fid, v in _pending_flows.items() if now - v.get("started", 0) > FLOW_TTL_SECONDS]
        for fid in stale_pending:
            _pending_flows.pop(fid, None)

        # Clean completed flows older than TTL
        stale_completed = [fid for fid, v in _completed_flows.items() if now - v.get("completed_at", 0) > COMPLETED_FLOW_TTL_SECONDS]
        for fid in stale_completed:
            _completed_flows.pop(fid, None)

        # Clean sessions older than TTL (even if refresh token may still be valid,
        # we bound memory; user will re-authenticate)
        stale_sessions = [sid for sid, v in _sessions.items() if now - v.get("created_at", 0) > SESSION_TTL_SECONDS]
        for sid in stale_sessions:
            _sessions.pop(sid, None)

        # Clean rate limit entries
        stale_ips = [ip for ip, timestamps in _rate_limit.items() if all(now - t > RATE_LIMIT_WINDOW for t in timestamps)]
        for ip in stale_ips:
            _rate_limit.pop(ip, None)


def _check_rate_limit(ip: str) -> bool:
    """Return True if the request should be rate-limited (denied)."""
    now = time.time()
    with _store_lock:
        timestamps = _rate_limit.get(ip, [])
        # Remove timestamps outside the window
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            _rate_limit[ip] = timestamps
            return True
        timestamps.append(now)
        _rate_limit[ip] = timestamps
    return False


def configure_device_login(client_app_id: str, authority: str, server_app_id: str):
    """Called from app.py during setup to provide Entra ID credentials."""
    _config["client_app_id"] = client_app_id
    _config["authority"] = authority
    _config["scopes"] = [f"api://{server_app_id}/access_as_user"]
    logger.info("Device login configured for client %s", client_app_id)


def _evict_oldest_if_needed(store: OrderedDict, max_size: int):
    """Remove oldest entries from an OrderedDict to stay within max_size."""
    while len(store) > max_size:
        store.popitem(last=False)


def _run_device_flow(flow_id: str, pca: PublicClientApplication, flow: dict):
    """Background thread: blocks on MSAL polling until user completes auth."""
    try:
        result = pca.acquire_token_by_device_flow(flow)
        if "access_token" in result:
            session_token = secrets.token_urlsafe(48)
            accounts = pca.get_accounts()
            now = time.time()
            with _store_lock:
                _evict_oldest_if_needed(_sessions, MAX_SESSIONS - 1)
                _sessions[session_token] = {
                    "access_token": result["access_token"],
                    "expires_at": now + result.get("expires_in", 3600),
                    "username": result.get("id_token_claims", {}).get("preferred_username", ""),
                    "name": result.get("id_token_claims", {}).get("name", ""),
                    "pca": pca,
                    "account": accounts[0] if accounts else None,
                    "created_at": now,
                }
                _evict_oldest_if_needed(_completed_flows, MAX_COMPLETED_FLOWS - 1)
                _completed_flows[flow_id] = {
                    "status": "complete",
                    "session_token": session_token,
                    "access_token": result["access_token"],
                    "expires_in": result.get("expires_in", 3600),
                    "username": result.get("id_token_claims", {}).get("preferred_username", ""),
                    "name": result.get("id_token_claims", {}).get("name", ""),
                    "completed_at": now,
                }
            logger.info("Device code flow %s succeeded for %s",
                        flow_id, result.get("id_token_claims", {}).get("preferred_username", "?"))
        else:
            with _store_lock:
                _evict_oldest_if_needed(_completed_flows, MAX_COMPLETED_FLOWS - 1)
                _completed_flows[flow_id] = {
                    "status": "error",
                    "error": result.get("error_description", "Authentication failed"),
                    "completed_at": time.time(),
                }
            logger.warning("Device code flow %s failed: %s", flow_id, result.get("error"))
    except Exception:
        with _store_lock:
            _evict_oldest_if_needed(_completed_flows, MAX_COMPLETED_FLOWS - 1)
            _completed_flows[flow_id] = {
                "status": "error",
                "error": "An unexpected error occurred during authentication.",
                "completed_at": time.time(),
            }
        logger.exception("Device code flow %s exception", flow_id)
    finally:
        with _store_lock:
            _pending_flows.pop(flow_id, None)


@device_login_bp.route("/api/device_login/start", methods=["POST"])
async def start_device_login():
    """Initiate a device code flow. Returns user_code + verification_uri."""
    if not _config:
        return jsonify({"error": "Device login not configured"}), 500

    # Run stale-entry cleanup on each start request
    _cleanup_stale_entries()

    # Rate limit per IP (RFC 8628 §5.1)
    client_ip = request.remote_addr or "unknown"
    if _check_rate_limit(client_ip):
        return jsonify({"error": "Too many requests. Please wait a minute and try again."}), 429

    # Enforce maximum concurrent pending flows (prevent memory exhaustion)
    with _store_lock:
        if len(_pending_flows) >= MAX_PENDING_FLOWS:
            return jsonify({"error": "Service is busy. Please try again in a few minutes."}), 503

    data = await request.get_json(silent=True) or {}
    email = data.get("email", "").strip()

    pca = PublicClientApplication(
        client_id=_config["client_app_id"],
        authority=_config["authority"],
    )
    flow = await asyncio.to_thread(pca.initiate_device_flow, scopes=_config["scopes"])
    if "error" in flow:
        logger.error("initiate_device_flow error: %s", flow.get("error_description"))
        return jsonify({"error": flow.get("error_description", "Failed to start device flow")}), 400

    flow_id = str(uuid4())
    with _store_lock:
        _evict_oldest_if_needed(_pending_flows, MAX_PENDING_FLOWS - 1)
        _pending_flows[flow_id] = {"email": email, "started": time.time()}

    # Start the blocking MSAL poller in a background thread
    _executor.submit(_run_device_flow, flow_id, pca, flow)

    return jsonify({
        "flow_id": flow_id,
        "user_code": flow["user_code"],
        "verification_uri": flow["verification_uri"],
        "message": flow.get("message", ""),
        "expires_in": flow.get("expires_in", 900),
        "interval": flow.get("interval", 5),  # RFC 8628 §3.2: tell frontend the min polling interval
    })


@device_login_bp.route("/api/device_login/poll", methods=["POST"])
async def poll_device_login():
    """Check whether the device code flow has completed."""
    data = await request.get_json(silent=True) or {}
    flow_id = data.get("flow_id", "")

    with _store_lock:
        completed = _completed_flows.pop(flow_id, None)

    if completed:
        if completed["status"] == "error":
            return jsonify(completed), 400
        return jsonify(completed)

    # Return "pending" for both genuinely pending AND unknown flow IDs
    # to prevent flow enumeration (OWASP info disclosure)
    return jsonify({"status": "pending"})


@device_login_bp.route("/api/device_login/refresh", methods=["POST"])
async def refresh_device_login():
    """Use a stored session token to silently get a fresh access token."""
    data = await request.get_json(silent=True) or {}
    session_token = data.get("session_token", "")

    with _store_lock:
        session = _sessions.get(session_token)
    if not session:
        return jsonify({"status": "expired"}), 401

    # If current access token is still valid, return it
    if session["expires_at"] > time.time() + 60:
        return jsonify({
            "status": "valid",
            "access_token": session["access_token"],
            "username": session["username"],
            "name": session["name"],
        })

    # Try silent token acquisition using MSAL cache
    pca = session.get("pca")
    account = session.get("account")
    if pca and account:
        try:
            result = await asyncio.to_thread(
                pca.acquire_token_silent,
                _config["scopes"],
                account=account,
            )
            if result and "access_token" in result:
                session["access_token"] = result["access_token"]
                session["expires_at"] = time.time() + result.get("expires_in", 3600)
                return jsonify({
                    "status": "valid",
                    "access_token": result["access_token"],
                    "username": session["username"],
                    "name": session["name"],
                })
        except Exception:
            logger.exception("Silent token refresh failed for session")

    # Refresh token expired or silent acquisition failed
    with _store_lock:
        _sessions.pop(session_token, None)
    return jsonify({"status": "expired"}), 401
