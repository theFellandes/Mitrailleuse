#!/usr/bin/env python
import argparse, base64, json, shutil, sys, time, grpc, pathlib
from mitrailleuse import mitrailleuse_pb2 as pb
from mitrailleuse import mitrailleuse_pb2_grpc as stubs


def main():
    ap = argparse.ArgumentParser(description="Create an OpenAI batch, wait for it, download results")
    ap.add_argument("--server", default="localhost:50051")
    ap.add_argument("--user",   required=True)
    ap.add_argument("--api", default="openai", choices=["openai", "deepseek"])
    ap.add_argument("--jsonl",  required=True, help="Batch JSONL file")
    ap.add_argument("--out",    default="results.jsonl")
    args = ap.parse_args()

    chan = grpc.insecure_channel(args.server)
    stub = stubs.MitrailleuseServiceStub(chan)

    # --- 1. CreateBatch (file is base64‑encoded inside the message) ---
    if args.api == "deepseek":
        sys.exit("DeepSeek API does not support batch jobs yet.")

    file_b64 = base64.b64encode(pathlib.Path(args.jsonl).read_bytes()).decode()
    batch = stub.CreateBatch(pb.BatchRequest(
        user_id=args.user,
        jsonl_file=file_b64
    ))
    job = batch.job_id
    print("🚀  Batch created:", job)

    # --- 2. Poll status ---
    while True:
        stat = stub.CheckBatchStatus(pb.BatchStatusRequest(job_id=job))
        print("⏱️   status:", stat.status, end="\r")
        if stat.status in ("completed", "failed"):
            print()
            break
        time.sleep(5)   # seconds

    if stat.status != "completed":
        print("❌  Batch ended with status", stat.status)
        sys.exit(1)

    # --- 3. Stream results ---
    out = pathlib.Path(args.out)
    with out.open("wb") as sink:
        for line in stub.DownloadBatchResults(pb.BatchResultRequest(job_id=job)):
            sink.write(line.json_line.encode("utf-8") + b"\n")
    print("✅  Saved", out)


if __name__ == "__main__":
    main()
