from mitrailleuse.config.config import Config, GeneralConfig, Sampling, GeneralLoggingConfig, DBConfig, SimilarityCheck

def test_config_roundtrip(tmp_path):
    cfg = Config(
        task_name="test_task",
        user_id="user1",
        general=GeneralConfig(
            verbose=True,
            sampling=Sampling(enable_sampling=False, sample_size=1),
            logs=GeneralLoggingConfig(
                log_file="log.log", log_to_file=False, log_to_db=False, log_buffer_size=10
            ),
            multiprocessing_enabled=False,
            num_processes=1,
            db=DBConfig(postgres={}),
            similarity_check=SimilarityCheck()
        )
    )
    path = tmp_path / "config.json"
    Config.write(cfg, path)
    cfg2 = Config.read(path)
    assert cfg2.task_name == "test_task"