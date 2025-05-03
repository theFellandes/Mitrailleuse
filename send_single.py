#!/usr/bin/env python
import argparse, json, grpc
from mitrailleuse import mitrailleuse_pb2 as pb
from mitrailleuse import mitrailleuse_pb2_grpc as stubs


def main():
    ap = argparse.ArgumentParser(description="Send one OpenAI request")
    ap.add_argument("--server", default="localhost:50051")
    ap.add_argument("--user",   required=True, help="same user_id you used in CreateTask")
    ap.add_argument("--api",    default="openai", choices=["openai", "deepseek", "deepl"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--temp",   type=float, default=0.7)
    ap.add_argument("--messages", required=True,
                    help="JSON file containing list[dict] of chat messages")
    args = ap.parse_args()

    with open(args.messages, encoding="utf-8") as f:
        msgs = json.load(f)

    chan = grpc.insecure_channel(args.server)
    stub = stubs.MitrailleuseServiceStub(chan)
    resp = stub.SendSingle(pb.SingleRequest(
        user_id   = args.user,
        api_name=args.api,
        messages  = json.dumps(msgs),  # adjust if proto has repeated Message
        model     = args.model or ("deepseek-chat" if args.api == "deepseek" else "gpt-4o-mini"),
        temperature = args.temp
    ))
    print("💬  Assistant:", resp.content)


if __name__ == "__main__":
    main()
