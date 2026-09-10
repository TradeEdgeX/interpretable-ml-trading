from __future__ import annotations


def compute_next_aligned_delay_seconds(*, now_ns: int, interval_minutes: int) -> int:
    """
    Compute delay seconds until the next check aligned to multiples of interval_minutes.

    - now_ns: current time in ns (unix epoch)
    - interval_minutes: e.g. 10 => align to :00/:10/:20/:30/:40/:50
    """
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be > 0")

    now_sec = int(now_ns // 1_000_000_000)
    current_min = (now_sec // 60) % 60
    sec_in_min = now_sec % 60

    next_check_min = ((current_min // interval_minutes) + 1) * interval_minutes
    if next_check_min >= 60:
        # to next hour
        delay_sec = (60 - current_min) * 60 - sec_in_min
    else:
        delay_sec = (next_check_min - current_min) * 60 - sec_in_min

    if delay_sec <= 0:
        delay_sec += interval_minutes * 60
    return int(delay_sec)
