import json
from types import SimpleNamespace

import pytest

from scripts import training_time_limit as limit


def test_time_limit_option_preserves_training_arguments():
    forwarded, hours = limit.consume_time_limit(["--from-scratch", "--max-training-hours", "3", "--num_envs", "8192"])
    assert hours == 3
    assert forwarded == ["--from-scratch", "--num_envs", "8192"]
    assert limit.consume_time_limit(["--max-training-hours=0.5"])[1] == 0.5
    assert limit.consume_time_limit([]) == ([], None)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_time_limit_rejects_invalid_duration(value):
    with pytest.raises(SystemExit):
        limit.consume_time_limit(["--max-training-hours=" + value])


@pytest.mark.parametrize("hours, expected", [(1 / 3600, "time_limit_reached"), (1, "iterations_complete")])
def test_time_limit_saves_completed_iteration_and_restores_logger(tmp_path, monkeypatch, hours, expected):
    now = [0.0]
    monkeypatch.setattr(limit.time, "monotonic", lambda: now[0])
    calls = []

    class Runner:
        def learn(self, count, init_at_random_ep_len=False):
            for iteration in range(count):
                self.current_learning_iteration = iteration
                now[0] += 0.6
                self.logger.log(it=iteration)
            self.save(str(tmp_path / f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

        def save(self, path):
            calls.append(("save", path, self.current_learning_iteration))

    logger = SimpleNamespace(log_dir=str(tmp_path), log=lambda **kw: calls.append(("log", kw)),
                             stop_logging_writer=lambda: calls.append(("close",)))
    original_log = logger.log
    runner = Runner()
    runner.logger = logger
    runner.current_learning_iteration = 0
    limit.install_training_time_limit(Runner, hours)
    runner.learn(5)
    status = json.loads((tmp_path / "training_time_limit.json").read_text())
    assert status["status"] == expected
    assert status["iteration"] == (1 if expected == "time_limit_reached" else 4)
    assert len([call for call in calls if call[0] == "save"]) == 1
    assert calls[-1] == ("close",)
    assert logger.log is original_log
