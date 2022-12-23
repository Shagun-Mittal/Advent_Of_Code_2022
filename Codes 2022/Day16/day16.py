"""
Day 16: Proboscidea Volcanium
"""

import heapq
import itertools
import re
from collections import defaultdict
from dataclasses import dataclass

PATTERN = re.compile(
    r"Valve (\w+) has flow rate=(\d+); "
    r"(?:tunnel leads to valve|tunnels lead to valves) (\w+(?:, \w+)*)"
)

SAMPLE_INPUT = ["Valve TZ has flow rate=0; tunnels lead to valves ZJ, DM",
"Valve LH has flow rate=0; tunnels lead to valves FP, IS",
"Valve AA has flow rate=0; tunnels lead to valves XU, JH, CD, WY, HK",
"Valve GP has flow rate=0; tunnels lead to valves BO, KL",
"Valve GN has flow rate=0; tunnels lead to valves QO, FP",
"Valve QO has flow rate=0; tunnels lead to valves CA, GN",
"Valve JT has flow rate=22; tunnel leads to valve BL",
"Valve DF has flow rate=0; tunnels lead to valves BO, HK",
"Valve UM has flow rate=0; tunnels lead to valves OS, LE",
"Valve KJ has flow rate=0; tunnels lead to valves YF, UK",
"Valve UX has flow rate=23; tunnels lead to valves WM, ZI",
"Valve ZI has flow rate=0; tunnels lead to valves UX, AR",
"Valve YF has flow rate=0; tunnels lead to valves KJ, EK",
"Valve SX has flow rate=0; tunnels lead to valves DM, CD",
"Valve KZ has flow rate=0; tunnels lead to valves FR, LE",
"Valve IH has flow rate=0; tunnels lead to valves DM, IE",
"Valve EL has flow rate=0; tunnels lead to valves WQ, BO",
"Valve CD has flow rate=0; tunnels lead to valves AA, SX",
"Valve OR has flow rate=0; tunnels lead to valves FP, IR",
"Valve EK has flow rate=19; tunnels lead to valves YF, LK",
"Valve UE has flow rate=0; tunnels lead to valves FP, LG",
"Valve WQ has flow rate=0; tunnels lead to valves EL, DM",
"Valve XI has flow rate=0; tunnels lead to valves YH, DM",
"Valve GO has flow rate=0; tunnels lead to valves BO, CQ",
"Valve IR has flow rate=0; tunnels lead to valves ZJ, OR",
"Valve WY has flow rate=0; tunnels lead to valves UI, AA",
"Valve JH has flow rate=0; tunnels lead to valves AA, CA",
"Valve WM has flow rate=0; tunnels lead to valves UX, YH",
"Valve OS has flow rate=0; tunnels lead to valves UM, CA",
"Valve AE has flow rate=0; tunnels lead to valves FP, YH",
"Valve LG has flow rate=0; tunnels lead to valves UE, LE",
"Valve IS has flow rate=0; tunnels lead to valves LH, AR",
"Valve XU has flow rate=0; tunnels lead to valves AA, TU",
"Valve KL has flow rate=0; tunnels lead to valves GP, TU",
"Valve LV has flow rate=0; tunnels lead to valves UK, TU",
"Valve UI has flow rate=0; tunnels lead to valves ZJ, WY",
"Valve IL has flow rate=0; tunnels lead to valves GW, LK",
"Valve XY has flow rate=0; tunnels lead to valves AZ, CA",
"Valve JF has flow rate=15; tunnels lead to valves FR, BK",
"Valve UK has flow rate=18; tunnels lead to valves LV, KJ",
"Valve CA has flow rate=13; tunnels lead to valves JH, XY, QO, BK, OS",
"Valve BL has flow rate=0; tunnels lead to valves JT, GW",
"Valve GW has flow rate=16; tunnels lead to valves IL, BL",
"Valve CQ has flow rate=0; tunnels lead to valves ZJ, GO",
"Valve HK has flow rate=0; tunnels lead to valves DF, AA",
"Valve BO has flow rate=4; tunnels lead to valves GO, GP, EL, DF",
"Valve TU has flow rate=11; tunnels lead to valves XU, IE, KL, LV",
"Valve AZ has flow rate=0; tunnels lead to valves ZJ, XY",
"Valve FP has flow rate=5; tunnels lead to valves GN, AE, UE, LH, OR",
"Valve LE has flow rate=14; tunnels lead to valves KZ, LG, UM",
"Valve IE has flow rate=0; tunnels lead to valves IH, TU",
"Valve NZ has flow rate=0; tunnels lead to valves YH, AR",
"Valve DM has flow rate=3; tunnels lead to valves WQ, IH, TZ, SX, XI",
"Valve YH has flow rate=21; tunnels lead to valves WM, NZ, AE, XI",
"Valve BK has flow rate=0; tunnels lead to valves JF, CA",
"Valve LK has flow rate=0; tunnels lead to valves EK, IL",
"Valve AR has flow rate=20; tunnels lead to valves IS, NZ, ZI",
"Valve ZJ has flow rate=9; tunnels lead to valves IR, AZ, TZ, UI, CQ",
"Valve FR has flow rate=0; tunnels lead to valves JF, KZ",
]


def _parse(lines):
    return {
        (match := re.match(PATTERN, line)).group(1): (
            int(match.group(2)),
            match.group(3).split(", "),
        )
        for line in lines
    }


def _distances(adj):
    keys, distances = set(), defaultdict(lambda: float("inf"))
    for src, dsts in adj:
        keys.add(src)
        distances[src, src] = 0
        for dst, weight in dsts:
            keys.add(dst)
            distances[dst, dst] = 0
            distances[src, dst] = weight
    for mid in keys:
        for src in keys:
            for dst in keys:
                distance = distances[src, mid] + distances[mid, dst]
                if distance < distances[src, dst]:
                    distances[src, dst] = distance
    return distances


@dataclass(order=True, frozen=True)
class _State:
    rooms: tuple[tuple[str, int]]
    valves: frozenset[str]
    flow: int
    total: int
    time: int


def _solve(lines, num_agents, total_time):
    # pylint: disable=too-many-branches,too-many-nested-blocks,too-many-locals
    graph = _parse(lines)
    distances = _distances(
        (src, ((dst, 1) for dst in dsts)) for src, (_, dsts) in graph.items()
    )
    seen, max_seen = set(), 0
    heap = [
        (
            0,
            _State(
                rooms=(("AA", 0),) * num_agents,
                valves=frozenset(src for src, (flow, _) in graph.items() if flow > 0),
                flow=0,
                total=0,
                time=total_time,
            ),
        )
    ]

    while heap:
        estimate, state = heapq.heappop(heap)
        estimate = -estimate
        if state in seen:
            continue
        seen.add(state)
        potential = estimate + sum(
            max(
                (
                    graph[valve][0] * (state.time - delta - 1)
                    for room, age in state.rooms
                    if (delta := distances[room, valve] - age) in range(state.time)
                ),
                default=0,
            )
            for valve in state.valves
        )
        if estimate > max_seen:
            max_seen = estimate
        if potential < max_seen:
            continue

        moves_by_time = defaultdict(lambda: defaultdict(list))
        for valve in state.valves:
            for i, (room, age) in enumerate(state.rooms):
                delta = distances[room, valve] - age
                if delta in range(state.time):
                    moves_by_time[delta][i].append(valve)
        if not moves_by_time:
            continue

        for delta, moves_by_agent in moves_by_time.items():
            for size in range(1, num_agents + 1):
                for combo in itertools.combinations(moves_by_agent.items(), size):
                    indices = [i for i, _ in combo]
                    for valves in itertools.product(*(valves for _, valves in combo)):
                        if len(set(valves)) != size:
                            continue
                        new_rooms = [
                            (room, age + delta + 1) for room, age in state.rooms
                        ]
                        for i, valve in zip(indices, valves):
                            new_rooms[i] = valve, 0
                        rate = sum(graph[valve][0] for valve in valves)
                        new_state = _State(
                            rooms=tuple(sorted(new_rooms)),
                            valves=state.valves - set(valve for valve in valves),
                            flow=state.flow + rate,
                            total=state.total + state.flow * (delta + 1),
                            time=state.time - delta - 1,
                        )
                        heapq.heappush(
                            heap, (-estimate - rate * new_state.time, new_state)
                        )

    return max_seen


def part1(lines):
    
    
    
    
    return _solve(lines, num_agents=1, total_time=30)


def part2(lines):
    """
    >>> part2(SAMPLE_INPUT)
    1707
    """
    return _solve(lines, num_agents=2, total_time=26)


parts = (part1, part2)
print(part1(SAMPLE_INPUT))
print(part2(SAMPLE_INPUT))