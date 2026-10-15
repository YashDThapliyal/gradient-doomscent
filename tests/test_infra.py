from omegaconf import OmegaConf

from doomscent.checkpoints import list_checkpoints, parse_steps
from doomscent.config import load_config
from doomscent.metrics import column_name
from doomscent.plots.common import fmt_steps
from doomscent.plots.data import run_group
from doomscent.runs import RunPaths
from doomscent.train import checkpoint_interval, config_diff, sf_argv


def test_parse_steps():
    assert parse_steps("2.5M") == 2_500_000
    assert parse_steps("500k") == 500_000
    assert parse_steps("0") == 0
    assert parse_steps("1200000") == 1_200_000


def test_fmt_steps():
    assert [fmt_steps(x) for x in (0, 500_000, 1_000_000, 2_500_000, 20_000_000)] == [
        "0",
        "500K",
        "1M",
        "2.5M",
        "20M",
    ]


def test_metric_columns():
    heads = ["turn", "attack"]
    assert column_name("train/entropy_head1", heads) == "policy_entropy_attack"
    assert column_name("policy_stats/avg_act_turn_left_fast", heads) == "action_hist_turn_left_fast"
    assert column_name("policy_stats/avg_kills", heads) == "kills_mean"
    assert column_name("policy_stats/avg_kills_min", heads) is None
    assert column_name("perf/_fps", heads) == "fps"


def test_run_group():
    assert run_group("v0_defend_center_s12") == "v0_defend_center"
    assert run_group("v0_smoke") == "v0_smoke"


def test_sf_argv_from_config():
    cfg = load_config("v0_defend_center_s1.yaml")
    args = sf_argv(cfg, RunPaths.for_run("x"))
    assert "--env_frameskip=4" in args and "--batch_size=2048" in args
    assert "--use_rnn=True" in args and "--rnn_type=gru" in args
    assert args[args.index("--wandb_tags") + 1 : args.index("--wandb_tags") + 3] == [
        "v0",
        "defend_the_center",
    ]
    assert checkpoint_interval(cfg) == 100_000


def test_config_diff_flags_learning_changes_only():
    a = OmegaConf.create({"reward": {"kill": 1.0}, "train": {"wandb": {"mode": "offline"}}})
    b = OmegaConf.create({"reward": {"kill": 2.0}, "train": {"wandb": {"mode": "online"}}})
    assert config_diff(a, b) == ["reward.kill: 1.0 -> 2.0"]


def test_list_checkpoints_by_boundary(tmp_path):
    root = tmp_path / "r"
    cfg = load_config("v0_defend_center_s1.yaml")
    paths = RunPaths("r", root)
    paths.milestone_dir.mkdir(parents=True)
    OmegaConf.save(cfg, paths.config)
    for name in (
        "checkpoint_000000000_0.pth",
        "checkpoint_000000015_122880.pth",
        "checkpoint_000000070_516096.pth",
        "checkpoint_000000071_524288.pth",
    ):
        (paths.milestone_dir / name).write_bytes(b"")
    cks = list_checkpoints(paths)
    assert [(c.step, c.regular) for c in cks] == [(0, True), (100_000, False), (500_000, True)]
    assert cks[2].env_steps == 516096 and cks[2].label == "step_0500000"


def test_resume_allows_eval_changes_but_not_learning_changes():
    a = OmegaConf.create({"record": {"crf": 20}, "eval": {"x": 1}, "train": {"gamma": 0.99}})
    b = OmegaConf.create({"record": {"crf": 30}, "eval": {"x": 2}, "train": {"gamma": 0.995}})
    assert config_diff(a, b) == ["train.gamma: 0.99 -> 0.995"]
