"""Open-source extract CLI: features, data, FeatureStore, court."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

import click


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _env_with_pythonpath() -> dict:
    env = os.environ.copy()
    parts = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
    if env.get("PYTHONPATH"):
        parts.insert(0, env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(parts)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_python_module(module: str, args: List[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    click.echo("Running: " + " ".join(cmd))
    return subprocess.run(cmd, env=_env_with_pythonpath(), cwd=str(PROJECT_ROOT)).returncode


def run_script(script_path: str, args: List[str]) -> int:
    script = PROJECT_ROOT / script_path
    cmd = [sys.executable, str(script), *args]
    click.echo("Running: " + " ".join(cmd))
    return subprocess.run(cmd, env=_env_with_pythonpath(), cwd=str(PROJECT_ROOT)).returncode


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Rule-research CLI (closed-bar features + court)."""


@cli.group()
def features():
    """List registered feature compute functions."""


@features.command("list")
@click.option("--all", "-a", "show_all", is_flag=True, help="Show names only (default)")
@click.option("--category", "-c", default=None, help="Filter by registry category")
@click.option("--search", "-s", default=None, help="Substring on the function name")
def features_list(show_all: bool, category: Optional[str], search: Optional[str]) -> None:
    from src.features.registry import ensure_features_registered, get_registry

    ensure_features_registered()
    names = get_registry().list_features(category)
    if search:
        q = search.lower()
        names = [n for n in names if q in n.lower()]
    for name in names:
        click.echo(name)
    if show_all:
        click.echo(f"({len(names)} functions)")


@features.command("count")
def features_count() -> None:
    from src.features.registry import ensure_features_registered, get_registry

    ensure_features_registered()
    reg = get_registry()
    names = reg.list_features()
    cats = Counter((reg._metadata.get(n) or {}).get("category", "default") for n in names)
    click.echo(f"total {len(names)}")
    for cat, n in cats.most_common():
        click.echo(f"  {cat}: {n}")


@cli.group()
def data():
    """Download and convert Binance aggTrades."""


def _data_download_impl(
    *,
    symbols: str,
    universe_config: Optional[str],
    universe_set: str,
    universe_groups: Optional[str],
    start_year: str,
    start_month: str,
    end_year: Optional[str],
    end_month: Optional[str],
    data_dir: str,
    parquet_dir: str,
) -> int:
    if universe_config:
        from src.data_tools.universe_config import load_universe_config

        cfg = load_universe_config(universe_config)
        groups = (
            [g.strip() for g in str(universe_groups).split(",") if g.strip()]
            if universe_groups
            else None
        )
        resolved = cfg.resolve_symbols_usdt(
            universe_set=str(universe_set), groups=groups
        )
        symbols = ",".join(resolved)
    args = [
        "--data-dir",
        data_dir,
        "--parquet-dir",
        parquet_dir,
        "--symbols",
        *[s for s in symbols.split(",") if s.strip()],
        "--start-year",
        str(start_year),
        "--start-month",
        str(start_month),
        "--yes",
    ]
    if end_year:
        args.extend(["--end-year", str(end_year)])
    if end_month:
        args.extend(["--end-month", str(end_month)])
    return run_script("src/data_tools/download_training_data.py", args)


@data.command("download")
@click.option("--symbols", "-s", default="BTCUSDT,ETHUSDT")
@click.option("--universe-config", default=None)
@click.option("--universe-set", default="starter_a")
@click.option("--universe-groups", default=None)
@click.option("--start-year", default="2023")
@click.option("--start-month", default="1")
@click.option("--end-year", default=None)
@click.option("--end-month", default=None)
@click.option("--data-dir", default="data/agg_data")
@click.option("--parquet-dir", default="data/parquet_data")
@click.option("--docker/--no-docker", default=False, help="Ignored; extract runs locally")
def data_download(
    symbols,
    universe_config,
    universe_set,
    universe_groups,
    start_year,
    start_month,
    end_year,
    end_month,
    data_dir,
    parquet_dir,
    docker,
):
    """Download Binance monthly aggTrades."""
    sys.exit(
        _data_download_impl(
            symbols=symbols,
            universe_config=universe_config,
            universe_set=universe_set,
            universe_groups=universe_groups,
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
            data_dir=data_dir,
            parquet_dir=parquet_dir,
        )
    )


@data.command("convert")
@click.option("--input-dir", default=None)
@click.option("--output-dir", default=None)
@click.option("--pattern", default=None)
@click.option("--symbols", default=None)
@click.option("--force", is_flag=True)
@click.option("--docker/--no-docker", default=False, help="Ignored; extract runs locally")
def data_convert(input_dir, output_dir, pattern, symbols, force, docker):
    """Convert aggTrade ZIPs to parquet."""
    args: List[str] = []
    if pattern:
        args.extend(["--pattern", pattern])
    if symbols:
        args.extend(["--symbols", symbols])
    if input_dir:
        args.extend(["--input-dir", input_dir])
    if output_dir:
        args.extend(["--output-dir", output_dir])
    if force:
        args.append("--force")
    sys.exit(run_python_module("src.data_tools.zip_to_parquet", args))


@data.command("download-funding-rate")
@click.option("--symbols", "-s", default="BTCUSDT,ETHUSDT")
@click.option("--start-year", default="2023")
@click.option("--start-month", default="1")
@click.option("--end-year", default=None)
@click.option("--end-month", default=None)
@click.option("--docker/--no-docker", default=False, help="Ignored")
def data_download_funding_rate(
    symbols, start_year, start_month, end_year, end_month, docker
):
    args = [
        "--symbols",
        symbols,
        "--start-year",
        start_year,
        "--start-month",
        start_month,
    ]
    if end_year:
        args.extend(["--end-year", end_year])
    if end_month:
        args.extend(["--end-month", end_month])
    sys.exit(run_script("src/data_tools/download_funding_rate.py", args))


@data.command("download-open-interest")
@click.option("--symbols", "-s", default="BTCUSDT,ETHUSDT")
@click.option("--start-year", default="2023")
@click.option("--start-month", default="1")
@click.option("--end-year", default=None)
@click.option("--end-month", default=None)
@click.option("--docker/--no-docker", default=False, help="Ignored")
def data_download_open_interest(
    symbols, start_year, start_month, end_year, end_month, docker
):
    args = [
        "--symbols",
        symbols,
        "--start-year",
        start_year,
        "--start-month",
        start_month,
    ]
    if end_year:
        args.extend(["--end-year", end_year])
    if end_month:
        args.extend(["--end-month", end_month])
    sys.exit(run_script("src/data_tools/download_open_interest.py", args))


@data.command("download-ashare")
@click.option(
    "--symbols",
    "-s",
    default="000300.SH",
    help="Comma-separated A-share codes (e.g. 000300.SH,600519)",
)
@click.option("--years", default=5, show_default=True, type=int)
@click.option("--start-date", default=None, help="YYYY-MM-DD (overrides --years)")
@click.option("--end-date", default=None, help="YYYY-MM-DD")
@click.option("--output-dir", default="data/ashare/daily", show_default=True)
@click.option("--no-resume", is_flag=True, help="Re-download even if parquet exists")
@click.option(
    "--universe",
    default=None,
    help="listed,delisted,or listed,delisted — full SH/SZ A-share tape via baostock",
)
@click.option(
    "--backend",
    type=click.Choice(["akshare", "baostock", "sina"]),
    default="akshare",
    show_default=True,
)
@click.option("--workers", default=3, show_default=True, type=int)
@click.option(
    "--basic-path",
    default="data/ashare/stock_basic/stock_basic.parquet",
    show_default=True,
)
def data_download_ashare(
    symbols,
    years,
    start_date,
    end_date,
    output_dir,
    no_resume,
    universe,
    backend,
    workers,
    basic_path,
):
    """Download A-share / index daily OHLCV (court examples only)."""
    if universe:
        from src.data_tools.ashare_baostock import fetch_stock_basic, universe_symbols

        import pandas as pd

        basic_fp = Path(basic_path)
        if basic_fp.is_file():
            basic = pd.read_parquet(basic_fp)
        else:
            basic = fetch_stock_basic(basic_fp)
        start = start_date or None
        if start is None:
            from datetime import date, timedelta

            start = (date.today() - timedelta(days=int(years) * 365)).isoformat()
        syms = universe_symbols(basic, include=universe)
        engine = backend if backend != "akshare" else "sina"
        if engine == "baostock":
            from src.data_tools.ashare_baostock import download_ashare_universe

            stats = download_ashare_universe(
                syms,
                output_dir=output_dir,
                start_date=start,
                end_date=end_date,
                resume=not no_resume,
                workers=int(workers),
            )
        else:
            from src.data_tools.ashare_downloader import download_ashare_universe_sina

            stats = download_ashare_universe_sina(
                syms,
                output_dir=output_dir,
                start_date=start,
                end_date=end_date,
                resume=not no_resume,
                workers=int(workers),
            )
    else:
        from src.data_tools.ashare_downloader import download_ashare_daily

        if backend == "baostock":
            click.echo("baostock backend requires --universe; use akshare for --symbols")
            sys.exit(2)
        syms = [s.strip() for s in str(symbols).split(",") if s.strip()]
        stats = download_ashare_daily(
            syms,
            output_dir=output_dir,
            years=int(years),
            start_date=start_date,
            end_date=end_date,
            resume=not no_resume,
        )
    click.echo(
        f"ashare daily: total={stats['total']} success={stats['success']} "
        f"skipped={stats['skipped']} failed={stats['failed']} "
        f"elapsed={stats['elapsed_sec']}s → {stats['output_dir']}"
    )
    if stats["failed"]:
        click.echo(f"failed symbols: {stats['failed_symbols'][:20]}")
        if len(stats["failed_symbols"]) > 20:
            click.echo(f"  … +{len(stats['failed_symbols']) - 20} more")
        if not universe:
            sys.exit(1)


@data.command("pipeline")
@click.pass_context
@click.option("--symbols", "-s", default="BTCUSDT,ETHUSDT")
@click.option("--start-year", default="2023")
@click.option("--start-month", default="1")
@click.option("--end-year", default=None)
@click.option("--end-month", default=None)
@click.option("--docker/--no-docker", default=False, help="Ignored")
def data_pipeline(
    ctx, symbols, start_year, start_month, end_year, end_month, docker
):
    """Download then convert."""
    ctx.invoke(
        data_download,
        symbols=symbols,
        universe_config=None,
        universe_set="starter_a",
        universe_groups=None,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        data_dir="data/agg_data",
        parquet_dir="data/parquet_data",
        docker=False,
    )
    ctx.invoke(
        data_convert,
        input_dir=None,
        output_dir=None,
        pattern=None,
        symbols=symbols,
        force=False,
        docker=False,
    )


@cli.group("feature-store")
def feature_store():
    """Build the monthly partitioned FeatureStore."""


@feature_store.command("build")
@click.option("--config", "-c", required=True)
@click.option("--symbols", "-s", default=None)
@click.option("--timeframe", "-t", required=True)
@click.option("--data-path", default="data/parquet_data")
@click.option("--start-date", default=None)
@click.option("--end-date", default=None)
@click.option("--root", "feature_store_root", default="feature_store")
@click.option("--layer", default=None)
@click.option("--docker/--no-docker", default=False, help="Ignored")
def feature_store_build(
    config,
    symbols,
    timeframe,
    data_path,
    start_date,
    end_date,
    feature_store_root,
    layer,
    docker,
):
    args = [
        "--config",
        config,
        "--timeframe",
        timeframe,
        "--data-path",
        data_path,
        "--root",
        feature_store_root,
    ]
    if symbols:
        args.extend(["--symbols", symbols])
    if start_date:
        args.extend(["--start-date", start_date])
    if end_date:
        args.extend(["--end-date", end_date])
    if layer:
        args.extend(["--layer", layer])
    sys.exit(run_script("scripts/build_feature_store_from_config.py", args))


@cli.group()
def research():
    """Court: index / harness / run / close. Humans --declare verdicts."""


def _research_forward(module: str, argv: list) -> None:
    import importlib

    mod = importlib.import_module(f"scripts.research.{module}")
    sys.exit(mod.main(argv))


_RESEARCH_CTX = {"ignore_unknown_options": True}


@research.command("index", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_index(args):
    _research_forward("index", list(args))


@research.command("harness", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_harness(args):
    _research_forward("harness", list(args))


@research.command("close", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_close(args):
    _research_forward("close", list(args))


@research.command("run", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_run(args):
    _research_forward("run", list(args))


@research.command("replay", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_replay(args):
    _research_forward("replay", list(args))


@research.command("standardize", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_standardize(args):
    _research_forward("standardize", list(args))


@research.command("scorecard", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_scorecard(args):
    _research_forward("scorecard", list(args))


@research.command("review", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_review(args):
    _research_forward("review", list(args))


@research.command("stale", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_stale(args):
    _research_forward("stale", list(args))


@research.command("validate", context_settings=_RESEARCH_CTX)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def research_validate(args):
    _research_forward("validate", list(args))


@research.command("init")
@click.argument("topic")
@click.option("--strategy", default="ma_cross")
@click.option("--layers", default="regime")
@click.option("--segment", default="recent_6m_oos")
@click.option("--force", is_flag=True)
def research_init(topic, strategy, layers, segment, force):
    from scripts.research.init_experiment import init_experiment

    try:
        out = init_experiment(
            topic, strategy=strategy, layers=layers, segment=segment, force=force
        )
    except FileExistsError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(3)
    click.echo(f"created {out}")


@cli.command("lab")
@click.option("--port", "-p", type=int, default=8008, show_default=True)
@click.option("--bind", default="127.0.0.1", show_default=True)
@click.option("--reload", is_flag=True, help="Reload on code changes (dev)")
def lab_cmd(port: int, bind: str, reload: bool) -> None:
    """Local court Lab: experiments, Q&A, results browse. No auxiliary trading."""
    from cli.i18n import t

    host_label = bind if bind not in ("0.0.0.0", "::") else "localhost"
    click.echo(t("cli.lab.banner"))
    click.echo(t("cli.bind", bind=bind, port=port))
    click.echo(t("cli.lab.rd", host=host_label, port=port))
    click.echo(t("cli.lab.qa", host=host_label, port=port))
    click.echo(t("cli.lab.browse", host=host_label, port=port))
    click.echo(t("cli.stop"))
    import uvicorn

    if reload:
        uvicorn.run("src.lab.app:app", host=bind, port=int(port), reload=True)
        return
    from src.lab.app import app

    uvicorn.run(app, host=bind, port=int(port))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
