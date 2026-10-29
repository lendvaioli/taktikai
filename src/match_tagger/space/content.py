"""What the Space shows: aggregates of one match and one accepted report, as pydantic models.

Only aggregates leave the machine: per-team counts and shares, ball losses per third and lane,
the pressing proxy, shot and xG totals, and the strongest pass-network links with mean player
positions. No per-event lists (no shots, passes or ball losses one by one), no raw data.
"""

from pathlib import Path

from pydantic import BaseModel

from match_tagger.events.schema import Team
from match_tagger.metrics.context import TEAMS, team_labels_hu
from match_tagger.metrics.stage import MatchMetrics
from match_tagger.metrics.tactical import PassNetwork, Pressing, ShotMap
from match_tagger.report.runs import RunRecord
from match_tagger.report.sample_note import RELIABILITY_HU, duration_hu, sample_note


class SpaceConfig(BaseModel):
    """Contents of configs/space.yaml."""

    match_id: str
    run_id: str
    title_hu: str
    out_dir: Path = Path("space")
    logo_light: str
    logo_dark: str
    top_edges: int = 10
    repo_url: str
    report_note_hu: str = ""


class TeamRow(BaseModel):
    """One team's aggregates for the metrics table."""

    name_hu: str
    passes: int
    completion_share: float | None
    ball_losses: int
    losses_by_third: dict[str, int]
    pressing: float | None
    opponent_passes: int | None
    opponent_ball_losses: int | None
    shots: int | None
    shot_goals: int | None
    xg_total: float | None


class NetworkView(BaseModel):
    """The strongest links of one team's pass network (aggregates only)."""

    team_hu: str
    nodes: dict[str, tuple[float, float, int]]  # player -> (mean x, mean y, passes)
    edges: list[tuple[str, str, int]]


class SpaceContent(BaseModel):
    """Everything the page shows."""

    title_hu: str
    teams_hu: dict[Team, str]
    duration_hu: str
    reliability_hu: str
    sample_note_hu: str
    attribution: str
    rows: dict[Team, TeamRow]
    networks: dict[Team, NetworkView]
    report: dict
    run: dict


def team_row(m: MatchMetrics, team: Team) -> TeamRow:
    """Aggregates of one team; unavailable metrics stay None."""
    s, losses = m.summary[team], m.ball_losses_by_zone[team]
    press, shots = m.ppda[team], m.shot_map[team]
    pressing = press if isinstance(press, Pressing) else None
    shot_map = shots if isinstance(shots, ShotMap) else None
    return TeamRow(
        name_hu=team_labels_hu(m.context)[team],
        passes=s.passes,
        completion_share=s.pass_completion_share,
        ball_losses=s.ball_losses,
        losses_by_third={k: v.count for k, v in losses.thirds.items()},
        pressing=pressing.value if pressing else None,
        opponent_passes=pressing.opponent_passes if pressing else None,
        opponent_ball_losses=pressing.opponent_ball_losses if pressing else None,
        shots=shot_map.shots if shot_map else None,
        shot_goals=shot_map.goals if shot_map else None,
        xg_total=shot_map.xg_total if shot_map else None,
    )


def network_view(m: MatchMetrics, team: Team, top: int) -> NetworkView | None:
    """Top links between players that both have a mean position; None without a player network."""
    net = m.pass_network[team]
    if not isinstance(net, PassNetwork) or net.node != "player":
        return None
    nodes = {
        n.id: (n.mean_x_m, n.mean_y_m, n.passes)
        for n in net.nodes
        if n.mean_x_m is not None and n.mean_y_m is not None
    }
    edges = [
        (e.source, e.target, e.count) for e in net.edges if e.source in nodes and e.target in nodes
    ]
    used = {p for s, t, _ in edges[:top] for p in (s, t)}
    return NetworkView(
        team_hu=team_labels_hu(m.context)[team],
        nodes={p: xy for p, xy in nodes.items() if p in used},
        edges=edges[:top],
    )


def build_content(m: MatchMetrics, record: RunRecord, cfg: SpaceConfig) -> SpaceContent:
    """Page content; refuses a match without StatsBomb attribution or a rejected report."""
    if m.context.source != "statsbomb" or not m.context.attribution:
        raise ValueError("the Space shows StatsBomb reference data only, with attribution")
    if record.status != "ok" or record.report is None or record.match_id != m.match_id:
        raise ValueError(f"run {record.run_id} is not an accepted report of {m.match_id}")
    networks = {t: v for t in TEAMS if (v := network_view(m, t, cfg.top_edges))}
    return SpaceContent(
        title_hu=cfg.title_hu,
        teams_hu=team_labels_hu(m.context),
        duration_hu=duration_hu(m.sample.duration_sec),
        reliability_hu=RELIABILITY_HU[m.sample.reliability],
        sample_note_hu=sample_note(m.sample),
        attribution=m.context.attribution,
        rows={t: team_row(m, t) for t in TEAMS},
        networks=networks,
        report=record.report,
        run=record.model_dump(
            mode="json", include={"run_id", "backend", "model", "prompt_version", "steps"}
        ),
    )
