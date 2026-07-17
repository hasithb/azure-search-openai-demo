import json
import os
from pathlib import Path

from msal import PublicClientApplication

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".azure" / "cpr-rag" / ".env"

for raw in ENV_PATH.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    os.environ[key] = value

tenant_id = os.environ["AZURE_AUTH_TENANT_ID"]
client_id = os.environ["AZURE_CLIENT_APP_ID"]
server_app_id = os.environ["AZURE_SERVER_APP_ID"]
authority = f"https://login.microsoftonline.com/{tenant_id}"
scopes = [f"api://{server_app_id}/access_as_user"]

app = PublicClientApplication(client_id=client_id, authority=authority)
flow = app.initiate_device_flow(scopes=scopes)
if "user_code" not in flow:
    raise SystemExit(json.dumps(flow, indent=2))

print(flow["message"], flush=True)
result = app.acquire_token_by_device_flow(flow)
print(json.dumps(result, indent=2), flush=True)
