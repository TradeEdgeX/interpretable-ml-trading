import yaml

from scripts.update_market_cap import (
    _coingecko_headers,
    _coingecko_effective_url,
    _infer_base_symbol,
    _load_universe_symbols,
    _write_yaml,
)


def test_infer_base_symbol_strips_quote():
    assert _infer_base_symbol("BTCUSDT") == "BTC"
    assert _infer_base_symbol("ETHUSDC") == "ETH"
    assert _infer_base_symbol("SOLUSD") == "SOL"
    assert _infer_base_symbol("BNB") == "BNB"


def test_load_universe_symbols_builds_usdt_symbols(tmp_path):
    uni = {
        "quote": "USDT",
        "universe_sets": {
            "starter_a": {
                "groups": {
                    "highcap": ["BTC", "ETH"],
                    "meme": ["DOGE"],
                }
            }
        },
    }
    p = tmp_path / "u.yaml"
    p.write_text(yaml.safe_dump(uni), encoding="utf-8")
    quote, symbols = _load_universe_symbols(str(p), universe_set="starter_a")
    assert quote == "USDT"
    assert set(symbols) == {"BTCUSDT", "ETHUSDT", "DOGEUSDT"}


def test_write_yaml_keeps_order_and_writes(tmp_path):
    p = tmp_path / "c.yaml"
    obj = {"provider": "coingecko", "symbols": {"BTCUSDT": {"coingecko_id": "bitcoin"}}}
    _write_yaml(p, obj)
    loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert loaded["provider"] == "coingecko"
    assert loaded["symbols"]["BTCUSDT"]["coingecko_id"] == "bitcoin"


def test_coingecko_headers_demo_vs_pro():
    assert _coingecko_headers(None) is None
    assert _coingecko_headers("CG-abc") == {"x-cg-demo-api-key": "CG-abc"}
    assert _coingecko_headers("not-demo") == {"x-cg-pro-api-key": "not-demo"}
    assert _coingecko_headers("CG-abc", force_pro=True) == {
        "x-cg-pro-api-key": "CG-abc"
    }


def test_coingecko_effective_url_switches_to_pro_host_for_pro_key():
    u = "https://api.coingecko.com/api/v3/coins/bitcoin"
    assert _coingecko_effective_url(u, "CG-abc") == u
    assert _coingecko_effective_url(u, "pro_key_123").startswith(
        "https://pro-api.coingecko.com/api/v3/"
    )
