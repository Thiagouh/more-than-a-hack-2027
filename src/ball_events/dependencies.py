"""Explicit transitive support without retaining raw coordinate histories."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Dependencies:
    start: int
    end: int
    available: int
    points: frozenset = frozenset()
    hints: frozenset = frozenset()
    events: frozenset = frozenset()

    @classmethod
    def point(cls, p, ball=False):
        key = f'{p["producer"]}:ball' if ball else p['track_id']
        return cls(p['support_start_frame'], p['support_end_frame'], p['available_at_frame'],
                   frozenset([f'{key}@{p["frame"]}']))

    @classmethod
    def hint(cls, h):
        # A phase may span a period boundary; only its evidence availability is carried.
        return cls(h['start_frame'], h['start_frame']+1, h['available_at_frame'], hints=frozenset([h['hint_id']]))

    @classmethod
    def parent(cls, event):
        d = event['dependencies']
        return cls(d['support_start_frame'], d['support_end_frame'], event['available_at_frame'],
                   frozenset(d['point_refs']), frozenset(d['hint_ids']), frozenset(d['event_ids']) | {event['event_id']})

    def union(self, *others):
        all_deps = (self, *others)
        return Dependencies(min(d.start for d in all_deps), max(d.end for d in all_deps),
                            max(d.available for d in all_deps),
                            frozenset().union(*(d.points for d in all_deps)),
                            frozenset().union(*(d.hints for d in all_deps)),
                            frozenset().union(*(d.events for d in all_deps)))

    def as_dict(self):
        return dict(support_start_frame=self.start, support_end_frame=self.end, available_at_frame=self.available,
                    point_refs=sorted(self.points), hint_ids=sorted(self.hints), event_ids=sorted(self.events))


def combine(points=(), ball=(), deps=(), frame=0):
    seed = Dependencies(frame, frame+1, frame)
    return seed.union(*(Dependencies.point(p) for p in points),
                      *(Dependencies.point(p, True) for p in ball), *deps)
