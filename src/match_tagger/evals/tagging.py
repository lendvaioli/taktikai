"""Pure state transitions of the tagger app: add, undo and delete events of one clip."""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from match_tagger.evals.golden import GoldenClip
from match_tagger.events.schema import Event, EventType, Team


class TagAction(BaseModel):
    """A message from the tagger component. ``nonce`` makes repeated messages idempotent."""

    action: Literal["add", "undo", "delete"]
    nonce: str
    type: EventType | None = None
    team: Team | None = None
    t_sec: float | None = None
    id: str | None = None


def apply_action(
    clip: GoldenClip, action: TagAction, history: list[str], new_id: Callable[[], str]
) -> tuple[GoldenClip, list[str]]:
    """Return the updated clip and undo history (ids added this session, oldest first).

    ``add`` appends an event, ``undo`` removes the most recently added event that still
    exists, ``delete`` removes one event by id. Unknown ids and empty undo are no-ops.
    """
    events, history = list(clip.events), list(history)
    if action.action == "add":
        if action.type is None or action.team is None or action.t_sec is None:
            raise ValueError("add needs type, team and t_sec")
        event = Event(id=new_id(), t_sec=round(action.t_sec, 2), type=action.type,
                      team=action.team)  # fmt: skip
        events.append(event)
        history.append(event.id)
    elif action.action == "undo":
        existing = {e.id for e in events}
        while history and history[-1] not in existing:
            history.pop()
        if history:
            removed = history.pop()
            events = [e for e in events if e.id != removed]
    else:
        events = [e for e in events if e.id != action.id]
        history = [h for h in history if h != action.id]
    events.sort(key=lambda e: (e.t_sec, e.id))
    return clip.model_copy(update={"events": events}), history
