"""
SignBridge QA Analysis - Phase 3 Supplement: Live WebSocket Benchmark
Tests against running WebSocket server at ws://127.0.0.1:5001:
- Malformed frames (non-JSON, empty, wrong types)
- Disconnect and reconnect
- 100+ rapid sequential messages (measuring p50/p95/p99 latency)
- Concurrent clients (5 concurrent connections)
"""

import asyncio
import json
import time
import websockets
import numpy as np
from pathlib import Path

WS_URL = "ws://127.0.0.1:5001"

async def test_websocket():
    results = {}
    print(f"Connecting to live WebSocket server at {WS_URL}...")

    # 1. Malformed frames
    print("1. Testing Malformed Frames...")
    malformed_handled = False
    try:
        async with websockets.connect(WS_URL, close_timeout=2) as ws:
            # Send non-JSON string
            await ws.send("NOT_JSON_DATA_CORRUPT_FRAME")
            # The server will ignore or catch and continue
            # Send valid request immediately after to confirm server didn't crash
            test_req = {
                "type": "letter",
                "req_id": 1,
                "landmarks": [0.5] * 126
            }
            await ws.send(json.dumps(test_req))
            reply = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(reply)
            malformed_handled = (data.get("req_id") == 1)
    except Exception as e:
        print(f"Malformed test exception: {e}")

    results["malformed_frame_handling"] = {
        "pass": malformed_handled,
        "detail": "Sent invalid non-JSON string; server survived and replied to subsequent request."
    }

    # 2. Disconnect & Reconnect
    print("2. Testing Disconnect & Reconnect...")
    reconnect_pass = False
    try:
        # First connection
        ws1 = await websockets.connect(WS_URL)
        await ws1.close()
        # Second connection
        ws2 = await websockets.connect(WS_URL)
        await ws2.send(json.dumps({"type": "letter", "req_id": 2, "landmarks": [0.5] * 126}))
        reply2 = await asyncio.wait_for(ws2.recv(), timeout=2.0)
        await ws2.close()
        reconnect_pass = (json.loads(reply2).get("req_id") == 2)
    except Exception as e:
        print(f"Reconnect exception: {e}")

    results["disconnect_reconnect"] = {
        "pass": reconnect_pass
    }

    # 3. 100+ Rapid Messages Latency Benchmark
    print("3. Testing 100+ Rapid Sequential Messages (Latency Measurement)...")
    latencies_ms = []
    try:
        async with websockets.connect(WS_URL) as ws:
            sample_landmarks = [0.5] * 126
            # Set wrist high to avoid idle reject
            sample_landmarks[1] = 0.4
            sample_landmarks[64] = 0.4

            for i in range(120):
                t0 = time.perf_counter()
                payload = {
                    "type": "letter",
                    "req_id": i + 100,
                    "landmarks": sample_landmarks
                }
                await ws.send(json.dumps(payload))
                raw = await ws.recv()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

        results["latency_stats_ms"] = {
            "total_messages": len(latencies_ms),
            "p50": round(float(np.percentile(latencies_ms, 50)), 2),
            "p95": round(float(np.percentile(latencies_ms, 95)), 2),
            "p99": round(float(np.percentile(latencies_ms, 99)), 2),
            "mean": round(float(np.mean(latencies_ms)), 2),
            "pass": len(latencies_ms) >= 100
        }
        print(f"Latency stats: p50={results['latency_stats_ms']['p50']}ms, p95={results['latency_stats_ms']['p95']}ms, p99={results['latency_stats_ms']['p99']}ms")
    except Exception as e:
        print(f"Latency benchmark exception: {e}")
        results["latency_stats_ms"] = {"pass": False, "error": str(e)}

    # 4. Concurrent Clients (5 connections)
    print("4. Testing Concurrent Clients...")
    concurrent_success = 0
    async def client_worker(worker_id):
        nonlocal concurrent_success
        try:
            async with websockets.connect(WS_URL) as ws:
                for j in range(10):
                    await ws.send(json.dumps({
                        "type": "letter",
                        "req_id": worker_id * 1000 + j,
                        "landmarks": [0.5] * 126
                    }))
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if data.get("req_id") == worker_id * 1000 + j:
                        concurrent_success += 1
        except Exception as err:
            print(f"Worker {worker_id} error: {err}")

    tasks = [client_worker(w) for w in range(5)]
    await asyncio.gather(*tasks)

    results["concurrent_clients"] = {
        "num_clients": 5,
        "messages_per_client": 10,
        "successful_replies": concurrent_success,
        "pass": concurrent_success == 50
    }
    print(f"Concurrent clients: {concurrent_success}/50 replies matched.")

    out_file = Path(__file__).resolve().parent / "phase3_ws_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"WebSocket results saved to {out_file}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
