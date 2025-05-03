import argparse, json, grpc, pathlib
from mitrailleuse import mitrailleuse_pb2 as pb
from mitrailleuse import mitrailleuse_pb2_grpc as stubs


def main():
    ap = argparse.ArgumentParser(description="Create a Mitrailleuse task")
    # TODO: Edit base_cfg dynamically.
    base_cfg = json.loads(pathlib.Path("mitrailleuse/config/config.json").read_text())
    ap.add_argument("--server", default="localhost:50051", help="host:port of gRPC server")
    ap.add_argument("--user",   help="user_id (folder name under tasks/)", default="owx123456")
    ap.add_argument("--api",    default="openai", choices=["openai", "deepseek"],
                    help="openai or deepseek")
    ap.add_argument("--name",   help="task_name, e.g. few_shot_demo", default="few_shot_demo")
    ap.add_argument("--config", help="JSON file with your task config", default=base_cfg)
    args = ap.parse_args()


    chan  = grpc.insecure_channel(args.server)
    stub  = stubs.MitrailleuseServiceStub(chan)
    resp  = stub.CreateTask(pb.CreateTaskRequest(
        user_id    = args.user,
        api_name   = args.api,
        task_name  = args.name,
        config_json= json.dumps(base_cfg, ensure_ascii=False)       # string field
    ))
    print("✅  Task folder:", resp.task_folder)


if __name__ == "__main__":
    main()
