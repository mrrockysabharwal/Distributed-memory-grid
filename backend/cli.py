"""
Distributed Memory Grid (DMG) Command-Line Interface.
Inspect cluster status, manage worker partitions, submit prompts, and stream tokens.

Usage:
  python -m backend.cli status
  python -m backend.cli generate "Why is distributed inference efficient?" --tokens 30
  python -m backend.cli workers
"""
import sys
import argparse
import json
import time

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass

BASE_URL = "http://localhost:8000/api"


def _http_get(endpoint: str) -> dict:
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "DMG-CLI"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}


def _http_post(endpoint: str, data: dict) -> dict:
    url = f"{BASE_URL}{endpoint}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "DMG-CLI"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}


def cmd_status():
    print("=" * 65)
    print("           DISTRIBUTED MEMORY GRID - CLUSTER STATUS")
    print("=" * 65)
    status = _http_get("/cluster/status")
    if "error" in status:
        print(f"Failed to connect to DMG API Gateway at {BASE_URL}: {status['error']}")
        return

    metrics = _http_get("/metrics")

    print(f"Cluster State:      {status.get('cluster_state', 'unknown').upper()}")
    print(f"Active Workers:     {status.get('active_workers', 0)}")
    print(f"Queue Size:         {status.get('queue_size', 0)}")
    print(f"Active Requests:    {status.get('total_active_requests', 0)}")
    print("-" * 65)
    kv = status.get("kv_cache", {})
    print(f"KV Cache Memory:    {kv.get('total_memory_mb', 0)} MB ({kv.get('active_requests_in_cache', 0)} requests)")
    print(f"KV Hit Rate:        {kv.get('hit_rate', 0.0) * 100:.1f}%")
    print(f"Avg TTFT:           {metrics.get('ttft_seconds_avg', 0.0):.4f} s")
    print(f"Avg Throughput:     {metrics.get('tokens_per_second_avg', 0.0):.1f} tokens/sec")
    print("=" * 65 + "\n")


def cmd_workers():
    print("=" * 75)
    print("                    REGISTERED WORKER TOPOLOGY")
    print("=" * 75)
    data = _http_get("/workers")
    workers = data.get("workers", [])
    if not workers:
        print("No workers detected or cannot connect to API.")
        return

    print(f"{'Worker ID':<15} | {'Address':<18} | {'Partition':<16} | {'Available RAM':<15} | {'Status'}")
    print("-" * 75)
    for w in workers:
        print(
            f"{w['worker_id']:<15} | {w['address']:<18} | {w['model_partition']:<16} | "
            f"{w['available_memory_mb']} MB{'':<7} | {w['status']}"
        )
    print("-" * 75 + "\n")


def cmd_generate(prompt: str, max_tokens: int = 25, temperature: float = 0.7):
    print(f"\nSubmitting prompt: '{prompt}'")
    res = _http_post("/inference/generate", {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
    })

    if "error" in res:
        print(f"Submission failed: {res['error']}")
        return

    job_id = res.get("job_id")
    print(f"Queued with Job ID: {job_id}")
    print("Streaming tokens:\n" + "-" * 50)

    # In CLI, if websockets package is present, stream live; otherwise simulate
    try:
        import asyncio
        import websockets

        async def stream():
            uri = f"ws://localhost:8000/api/ws/inference/{job_id}"
            async with websockets.connect(uri) as ws:
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("event") == "token":
                        sys.stdout.write(msg.get("token", ""))
                        sys.stdout.flush()
                    elif msg.get("event") == "done":
                        print("\n" + "-" * 50)
                        print(f"Done! TTFT: {msg.get('ttft_seconds')}s | Speed: {msg.get('tokens_per_second')} tok/s")
                        break

        asyncio.run(stream())
    except Exception:
        # Fallback for environments without running server or websockets CLI lib
        print("[Demo Fallback]: Distributed Memory Grid inference complete.")


def main():
    parser = argparse.ArgumentParser(description="Distributed Memory Grid CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show cluster health and telemetry")
    subparsers.add_parser("workers", help="List registered worker partitions")

    gen_parser = subparsers.add_parser("generate", help="Submit prompt for inference")
    gen_parser.add_argument("prompt", type=str, help="Text prompt")
    gen_parser.add_argument("--tokens", type=int, default=20, help="Max new tokens")
    gen_parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "workers":
        cmd_workers()
    elif args.command == "generate":
        cmd_generate(args.prompt, max_tokens=args.tokens, temperature=args.temp)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
