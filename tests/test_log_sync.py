from grasp_grpo.log_sync import stage_files


def test_stage_keeps_small_logs_and_tails_console(tmp_path):
    run = tmp_path / "stage3"
    (run / "logs").mkdir(parents=True)
    (run / "checkpoint-grpo-step-000050").mkdir()
    (run / "best").mkdir()
    (run / "metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    (run / "eval_history.json").write_text("[]", encoding="utf-8")
    (run / "logs" / "gpu_metrics.csv").write_text("timestamp,index\n", encoding="utf-8")
    (run / "logs" / "console.log").write_text("".join(f"line {i}\n" for i in range(500)), encoding="utf-8")
    (run / "checkpoint-grpo-step-000050" / "adapter_model.safetensors").write_bytes(b"x")
    (run / "checkpoint-grpo-step-000050" / "trainer_state.json").write_text("{}", encoding="utf-8")
    (run / "best" / "meta.json").write_text("{}", encoding="utf-8")

    staging = tmp_path / "staging"
    stage_files(run, staging, tail_lines=100)

    files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    assert files == ["eval_history.json", "logs/console_tail.log", "logs/gpu_metrics.csv", "metrics.jsonl"]
    tail = (staging / "logs" / "console_tail.log").read_text(encoding="utf-8").splitlines()
    assert len(tail) == 100 and tail[-1] == "line 499"
