"""The crash breaker must fingerprint the CAUSE, not the exit code.

A plain crash is booked from the reaped status alone: ``pid N exited with code 1``.
That label says nothing about WHY, so before this file existed every exit-1 crash
in a detection pass hashed to one fingerprint, three of them made the pass
"systemic", and every remaining crash tripped the breaker at ONE failure with
``sticky`` set — five healthy cards went ``ready`` -> ``blocked`` on 2026-10-27
over two environmental faults plus one already-answered ``needs_input`` question.

The fingerprint now folds in a normalised class of the worker's own last output
when (and only when) the error text is that synthetic exit-status label.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _dead_exit1_worker(conn, tid: str, pid: int, cause: str) -> None:
    """Claim ``tid`` for a worker that exited 1 and left ``cause`` as its last output."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(
            f"{cause}\n\nResume this session with:\n  hermes --resume x\n\n"
            f"{KANBAN_WORKER_EXIT_TRAILER}1\n"
        )


def _gave_up_events(conn, tid: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up' ORDER BY id",
        (tid,),
    ).fetchall()
    return [kb._json_dict(r["payload"]) for r in rows]


# Four unrelated causes, in the shape this board actually recorded on 2026-10-27:
# the harness's refusal sentence is repeated into every worker's tail, so the
# card-specific diagnostic sits far past the first 80 characters of the booked
# error text — where the old fingerprint stopped reading.
REFUSAL = (
    "Kanban owner refused: Gateway disconnected; turn outcome is unknown. Resume the "
    "printed session ID to check whether it completed or was interrupted; do not resend "
    "the work."
)
OWNER_REFUSED = REFUSAL + " The owner RPC never answered the receipt request."
INSTALL_FENCE = REFUSAL + " The install fence found an update still running; run hermes update."
NEEDS_INPUT = REFUSAL + " The worker had posted its needs_input question in comment 175."
CONNECTION_FAILED = REFUSAL + " ws://127.0.0.1:8760/ws refused three consecutive attempts."


def test_three_distinct_causes_leave_a_fourth_crash_out_of_the_systemic_path(kanban_home):
    """Three unrelated exit-1 crashes in one pass must not declare the fourth "systemic".

    Before the fix the fingerprint was ``error_text[:80]`` — 49 characters of
    synthetic label plus the shared refusal sentence — so all four crashed into
    ONE bucket, ``fp_counts`` reached 4, and every card was booked sticky at
    ``failure_limit=1``. Now the worker's own cause line is part of the key, so
    each crash is an ordinary failure: one count, no ``sticky``, card stays
    retryable — and only a repeat reaches ``DEFAULT_FAILURE_LIMIT``.
    """
    causes = [OWNER_REFUSED, CONNECTION_FAILED, INSTALL_FENCE, NEEDS_INPUT]
    with kbc.connect() as conn:
        tids = [kb.create_task(conn, title=f"c{i}", assignee="a") for i in range(4)]
        for i, (tid, cause) in enumerate(zip(tids, causes)):
            _dead_exit1_worker(conn, tid, 73000 + i, cause)

        # One detection pass sees all four dead workers.
        kbd.detect_crashed_workers(conn)

        for tid, cause in zip(tids, causes):
            task = kb.get_task(conn, tid)
            assert task.status == "ready", f"{tid} was held for an operator: {causes!r}"
            assert task.consecutive_failures == 1
            # The cause really was captured, so the fingerprint had something to group on.
            assert cause[:40] in (task.last_failure_error or "")
            # No ``gave_up`` event at all: the breaker did not trip, let alone stickily.
            assert _gave_up_events(conn, tid) == []

        # The fourth card is an ordinary failure, so it trips at the configured
        # limit — not at one failure. One more distinct-cause crash reaches it.
        last = tids[-1]
        _dead_exit1_worker(conn, last, 73100, "kanban_block: needs_input on this card — try again")
        kbd.detect_crashed_workers(conn)
        task = kb.get_task(conn, last)
        assert task.status == "blocked"
        assert task.consecutive_failures == kbd.DEFAULT_FAILURE_LIMIT
        events = _gave_up_events(conn, last)
        assert len(events) == 1
        assert events[0]["effective_limit"] == kbd.DEFAULT_FAILURE_LIMIT


def test_three_identical_causes_still_trip_stickily_at_one_failure(kanban_home):
    """The fast path survives: a genuinely repeated cause holds the card at 1 failure.

    The three crashed workers differ only in the volatile bits (pid, session id,
    timestamp, absolute path), which normalisation must strip so one real cause
    keeps one bucket.
    """
    causes = [
        "Kanban owner refused: Gateway disconnected while reading /home/hermes/.hermes/logs/a.log "
        "for session_id=aaaa1111 at 1793095922",
        "Kanban owner refused: Gateway disconnected while reading /var/tmp/b.log "
        "for session_id=bbbb2222 at 1793096000",
        "Kanban owner refused: Gateway disconnected while reading /tmp/c.log "
        "for session_id=cccc3333 at 1793096500",
    ]
    with kbc.connect() as conn:
        tids = [kb.create_task(conn, title=f"s{i}", assignee="a") for i in range(3)]
        for i, (tid, cause) in enumerate(zip(tids, causes)):
            _dead_exit1_worker(conn, tid, 74000 + i, cause)

        kbd.detect_crashed_workers(conn)

        for tid in tids:
            task = kb.get_task(conn, tid)
            assert task.status == "blocked"
            assert task.consecutive_failures == 1
            events = _gave_up_events(conn, tid)
            assert len(events) == 1
            assert events[0]["sticky"] is True
            assert events[0]["effective_limit"] == 1


@pytest.mark.parametrize(
    "a, b, same",
    [
        # Same cause, different pid/session id/timestamp/path -> one bucket.
        (
            "pid 11 exited with code 1 Worker's last output: 'failed to read "
            "/home/hermes/.hermes/state.db session_id=abc123 at 1793095922'",
            "pid 99 exited with code 1 Worker's last output: 'failed to read "
            "/tmp/x/state.db session_id=zzz999 at 1793096000'",
            True,
        ),
        # Different causes hid behind the same 186-char refusal boilerplate ->
        # different buckets (the old 80-char window could not tell them apart).
        (
            "pid 11 exited with code 1 Worker's last output: '" + REFUSAL
            + " The install fence found an update still running.'",
            "pid 11 exited with code 1 Worker's last output: '" + REFUSAL
            + " The worker's needs_input question sits in comment 175.'",
            False,
        ),
        # A non-synthetic error keeps its original fingerprint.
        ("provider authentication failed", "provider authentication failed", True),
        ("provider authentication failed", "provider rate limited", False),
    ],
)
def test_error_fingerprint_groups_by_cause(a, b, same):
    assert (kbd._error_fingerprint(a) == kbd._error_fingerprint(b)) is same


def test_synthetic_label_without_worker_output_keeps_the_old_fingerprint():
    """No worker output to fold in: fall back to the bare normalised label, unchanged."""
    assert kbd._error_fingerprint("pid 42 exited with code 1") == "pid n exited with code 1"
    assert kbd._error_fingerprint("pid 42 killed by signal 9") == "pid n killed by signal 9"
