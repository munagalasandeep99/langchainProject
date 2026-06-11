import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import boto3

CONFIG_BUCKET = "internal-config-bucket"
CONFIG_KEY    = "config.json"

LOG_DIR = "/var/log/app_log/"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

os.makedirs(LOG_DIR, exist_ok=True)

# Fail fast if the log file cannot be opened — no further details given
with open(LOG_FILE, "a"):
    pass


def _imdsv2_token() -> str:
    """Obtain a short-lived IMDSv2 session token (required when IMDSv1 is disabled)."""
    resp = requests.put(
        "http://169.254.169.254/latest/api/token",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        timeout=2,
    )
    resp.raise_for_status()
    return resp.text


def get_instance_ip() -> str:
    """Fetch the instance private IP via IMDSv2."""
    try:
        token = _imdsv2_token()
        r = requests.get(
            "http://169.254.169.254/latest/meta-data/local-ipv4",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=2,
        )
        return r.text
    except Exception as e:
        print(f"Could not fetch instance IP: {e}")
        return "UNKNOWN"


def get_candidate_id() -> str:
    """Fetch the Candidate-ID tag via IMDSv2 + boto3."""
    try:
        token = _imdsv2_token()
        headers = {"X-aws-ec2-metadata-token": token}

        instance_id = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers=headers,
            timeout=2,
        ).text

        region = requests.get(
            "http://169.254.169.254/latest/dynamic/instance-identity/document",
            headers=headers,
            timeout=2,
        ).json().get("region")

        ec2 = boto3.client("ec2", region_name=region)
        tags = ec2.describe_tags(Filters=[
            {"Name": "resource-id", "Values": [instance_id]},
            {"Name": "key",         "Values": ["Candidate-ID"]},
        ])
        return tags["Tags"][0]["Value"] if tags["Tags"] else "UNKNOWN"
    except Exception as e:
        print(f"Could not fetch Candidate-ID: {e}")
        return "UNKNOWN"


def get_app_config() -> dict:
    """Fetch assessment config from S3 at startup."""
    token   = _imdsv2_token()
    region  = requests.get(
        "http://169.254.169.254/latest/dynamic/instance-identity/document",
        headers={"X-aws-ec2-metadata-token": token},
        timeout=2,
    ).json().get("region")
    s3  = boto3.client("s3", region_name=region)
    obj = s3.get_object(Bucket=CONFIG_BUCKET, Key=CONFIG_KEY)
    return json.loads(obj["Body"].read().decode())


# Fetch once at startup – avoids a metadata + EC2 API call on every request
INSTANCE_IP   = get_instance_ip()
CANDIDATE_ID  = get_candidate_id()
APP_CONFIG    = get_app_config()
ASSESSMENT_ID = APP_CONFIG["assessment_id"]


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Zetwerk Assessment</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f0f4f8;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    .card {{
      background: #ffffff;
      border-radius: 16px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.10);
      padding: 48px 56px;
      max-width: 480px;
      width: 100%;
      text-align: center;
    }}

    .logo {{
      font-size: 2rem;
      font-weight: 800;
      color: #1a1a2e;
      letter-spacing: -0.5px;
      margin-bottom: 4px;
    }}

    .logo span {{ color: #e63946; }}

    .subtitle {{
      font-size: 0.95rem;
      color: #6b7280;
      margin-bottom: 36px;
    }}

    .info-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 0;
      border-bottom: 1px solid #f1f3f5;
    }}

    .info-row:last-of-type {{ border-bottom: none; }}

    .label {{
      font-size: 0.82rem;
      font-weight: 600;
      color: #9ca3af;
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }}

    .value {{
      font-size: 0.97rem;
      font-weight: 500;
      color: #111827;
    }}

    .badge {{
      margin-top: 32px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #ecfdf5;
      color: #065f46;
      font-size: 0.82rem;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 999px;
    }}

    .dot {{
      width: 8px; height: 8px;
      background: #10b981;
      border-radius: 50%;
      animation: pulse 1.5s infinite;
    }}

    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50%       {{ opacity: 0.4; }}
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">Zetwerk</div>
    <div class="subtitle">Candidate Assessment Environment</div>

    <div class="info-row">
      <span class="label">Assessment ID</span>
      <span class="value">{assessment_id}</span>
    </div>
    <div class="info-row">
      <span class="label">Candidate ID</span>
      <span class="value">{candidate_id}</span>
    </div>
    <div class="info-row">
      <span class="label">Instance IP</span>
      <span class="value">{instance_ip}</span>
    </div>

    <div class="badge">
      <div class="dot"></div>
      Environment is live
    </div>
  </div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        page = HTML_PAGE.format(assessment_id=ASSESSMENT_ID, candidate_id=CANDIDATE_ID, instance_ip=INSTANCE_IP)

        log_entry = f"candidate_id={CANDIDATE_ID} instance_ip={INSTANCE_IP} path={self.path}"
        with open(LOG_FILE, "a") as f:
            f.write(log_entry + "\n")

        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode())

    def log_message(self, fmt, *args):
        # Suppress default per-request stderr noise; real logging goes to LOG_FILE
        pass


def run(server_class=HTTPServer, handler_class=Handler, port=8000):
    server_address = ("127.0.0.1", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting Zetwerk Assessment app on port {port}")
    print(f"Instance IP  : {INSTANCE_IP}")
    print(f"Candidate ID : {CANDIDATE_ID}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()

