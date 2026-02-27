import os
import math
import heapq
import pickle
import bisect
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from pyproj import Transformer

def time_to_sec(t: str) -> int:
    h, m, s = str(t).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)
def sec_to_hhmm(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    return f"{h:02d}:{m:02d}"

def yyyymmdd(date_str: str) -> int:
    return int(str(date_str).replace("-", ""))

def normalize_name(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

def load_communes(shp_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path).to_crs(epsg=2154)
    cent = gdf.geometry.centroid
    gdf["cx"] = cent.x
    gdf["cy"] = cent.y
    gdf["nom_norm"] = gdf["nom"].map(normalize_name)
    return gdf[["insee", "nom", "nom_norm", "cx", "cy", "geometry"]].copy()

def pick_commune_by_name(gdf: gpd.GeoDataFrame, name: str) -> Tuple[str, str]:
    key = normalize_name(name)
    hits = gdf[gdf["nom_norm"] == key][["insee", "nom"]]
    if hits.empty:
        hits2 = gdf[gdf["nom_norm"].str.contains(key, na=False)][["insee", "nom"]]
        if hits2.empty:
            raise ValueError(f"Aucune commune trouvée pour: {name!r}")
        row = hits2.iloc[0]
        return str(row["insee"]), str(row["nom"])
    row = hits.iloc[0]
    return str(row["insee"]), str(row["nom"])

def read_gtfs_csv(gtfs_dir: str, name: str, usecols=None) -> pd.DataFrame:
    path = os.path.join(gtfs_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier manquant: {path}")
    return pd.read_csv(path, dtype=str, usecols=usecols, low_memory=False)

def active_service_ids(gtfs_dir: str, date_yyyymmdd: int) -> Set[str]:
    active: Set[str] = set()

    cal_path = os.path.join(gtfs_dir, "calendar.txt")
    cal_dates_path = os.path.join(gtfs_dir, "calendar_dates.txt")

    if os.path.exists(cal_path):
        cal = read_gtfs_csv(
            gtfs_dir,
            "calendar.txt",
            usecols=[
                "service_id",
                "monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday",
                "start_date", "end_date",
            ],
        )

        d = pd.to_datetime(str(date_yyyymmdd), format="%Y%m%d")
        wd = d.weekday()
        wd_col = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][wd]

        cal["start_date_i"] = pd.to_numeric(cal["start_date"], errors="coerce")
        cal["end_date_i"] = pd.to_numeric(cal["end_date"], errors="coerce")
        cal[wd_col] = pd.to_numeric(cal[wd_col], errors="coerce")

        base = cal[
            (cal["start_date_i"] <= date_yyyymmdd)
            & (cal["end_date_i"] >= date_yyyymmdd)
            & (cal[wd_col] == 1)
        ]["service_id"].astype(str)

        active = set(base.tolist())

    if os.path.exists(cal_dates_path):
        ex = read_gtfs_csv(gtfs_dir, "calendar_dates.txt", usecols=["service_id", "date", "exception_type"])
        ex["date_i"] = pd.to_numeric(ex["date"], errors="coerce")
        ex = ex[ex["date_i"] == date_yyyymmdd]

        for _, row in ex.iterrows():
            sid = str(row["service_id"])
            et = int(row["exception_type"])
            if et == 1:
                active.add(sid)
            elif et == 2:
                active.discard(sid)
    if not active:
        trips = read_gtfs_csv(gtfs_dir, "trips.txt", usecols=["service_id"])
        active = set(trips["service_id"].astype(str).unique().tolist())

    return active

def build_stops_xy_2154(gtfs_dir: str) -> pd.DataFrame:
    stops = read_gtfs_csv(
        gtfs_dir,
        "stops.txt",
        usecols=["stop_id", "stop_lat", "stop_lon", "stop_name", "parent_station", "location_type"],
    )
    stops["stop_id"] = stops["stop_id"].astype(str)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    xs, ys = tr.transform(
        stops["stop_lon"].astype(float).to_numpy(),
        stops["stop_lat"].astype(float).to_numpy()
    )
    out = pd.DataFrame(
        {
            "stop_id": stops["stop_id"].to_numpy(),
            "x": xs,
            "y": ys,
            "stop_name": stops["stop_name"].astype(str).to_numpy(),
            "parent_station": stops.get("parent_station", pd.Series([None] * len(stops))).astype(str).to_numpy(),
            "location_type": stops.get("location_type", pd.Series(["0"] * len(stops))).astype(str).to_numpy(),
        }
    )
    return out

def load_trip_stops(gtfs_dir: str, trip_id: str) -> pd.DataFrame:
    st = read_gtfs_csv(
        gtfs_dir,
        "stop_times.txt",
        usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
    )
    st = st[st["trip_id"].astype(str) == str(trip_id)].copy()
    if st.empty:
        return st
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce").fillna(0).astype(int)
    st.sort_values("stop_sequence", inplace=True)
    return st

def pretty_trip_calling_pattern_segment(
    gtfs_dir: str,
    trip_id: str,
    stop_name: Dict[str, str],
    from_stop_id: Optional[str],
    to_stop_id: Optional[str],
    limit: Optional[int] = None
) -> str:

    st = load_trip_stops(gtfs_dir, trip_id)
    if st.empty:
        return "  (impossible de charger les arrêts du trip)"
    if not from_stop_id or not to_stop_id:
        return pretty_trip_calling_pattern_full(gtfs_dir, trip_id, stop_name, limit=limit)

    from_stop_id = str(from_stop_id)
    to_stop_id = str(to_stop_id)
    seq_from = st.loc[st["stop_id"].astype(str) == from_stop_id, "stop_sequence"]
    seq_to = st.loc[st["stop_id"].astype(str) == to_stop_id, "stop_sequence"]

    if seq_from.empty or seq_to.empty:
        return pretty_trip_calling_pattern_full(gtfs_dir, trip_id, stop_name, limit=limit)

    a = int(seq_from.min())
    b = int(seq_to.max())
    if b < a:
        a, b = b, a

    seg = st[(st["stop_sequence"] >= a) & (st["stop_sequence"] <= b)].copy()
    rows = seg.to_dict("records")

    if limit is not None and len(rows) > limit:
        rows = rows[:limit]
        truncated = True
    else:
        truncated = False

    out = []
    for r in rows:
        sid = str(r["stop_id"])
        nm = stop_name.get(sid, sid)
        arr = str(r.get("arrival_time") or "")
        dep = str(r.get("departure_time") or "")
        arr_hm = arr[:5] if ":" in arr else arr
        dep_hm = dep[:5] if ":" in dep else dep
        out.append(f"  - {nm} ({arr_hm}/{dep_hm})")

    if truncated:
        out.append("  ... (tronqué)")
    return "\n".join(out)

def pretty_trip_calling_pattern_full(
    gtfs_dir: str,
    trip_id: str,
    stop_name: Dict[str, str],
    limit: Optional[int] = None
) -> str:
    st = load_trip_stops(gtfs_dir, trip_id)
    if st.empty:
        return "  (impossible de charger les arrêts du trip)"
    rows = st.to_dict("records")

    if limit is not None and len(rows) > limit:
        rows = rows[:limit]
        truncated = True
    else:
        truncated = False

    out = []
    for r in rows:
        sid = str(r["stop_id"])
        nm = stop_name.get(sid, sid)
        arr = str(r.get("arrival_time") or "")
        dep = str(r.get("departure_time") or "")
        arr_hm = arr[:5] if ":" in arr else arr
        dep_hm = dep[:5] if ":" in dep else dep
        out.append(f"  - {nm} ({arr_hm}/{dep_hm})")

    if truncated:
        out.append("  ... (tronqué)")
    return "\n".join(out)

def commune_access_stops(
    communes_gdf: gpd.GeoDataFrame,
    insee: str,
    stops_xy: pd.DataFrame,
    k: int = 20,
    max_radius_m: float = 50_000,
    access_speed_mps: float = 1.3,
    prefer_stop_points: bool = True,
) -> List[Tuple[str, int]]:
    row = communes_gdf[communes_gdf["insee"].astype(str) == str(insee)].iloc[0]
    cx, cy = float(row["cx"]), float(row["cy"])

    df = stops_xy
    if prefer_stop_points and "location_type" in df.columns:
        cand = df[df["location_type"].astype(str) == "0"]
        if len(cand) > 0:
            df = cand

    coords = df[["x", "y"]].to_numpy(float)
    ids = df["stop_id"].astype(str).to_numpy()
    tree = cKDTree(coords)

    kk = min(k, len(ids))
    dists, idxs = tree.query([[cx, cy]], k=kk)

    out: List[Tuple[str, int]] = []
    for dist, j in zip(dists[0], idxs[0]):
        if dist > max_radius_m:
            continue
        sec = int(dist / access_speed_mps)
        out.append((ids[j], sec))
    return out

@dataclass(frozen=True)
class Connection:
    dep: int
    arr: int
    u: str
    v: str
    trip_id: str
    route_id: str

def build_connections_for_date(
    gtfs_dir: str, date_yyyymmdd: int
) -> Tuple[List[Connection], Dict[str, str], Dict[str, str], Dict[str, str]]:
    services = active_service_ids(gtfs_dir, date_yyyymmdd)

    trips_path = os.path.join(gtfs_dir, "trips.txt")
    trips_all = pd.read_csv(trips_path, dtype=str, low_memory=False)

    has_short = "trip_short_name" in trips_all.columns
    has_headsign = "trip_headsign" in trips_all.columns

    needed = ["trip_id", "route_id", "service_id"]
    if has_short:
        needed.append("trip_short_name")
    if has_headsign:
        needed.append("trip_headsign")

    trips = trips_all[needed].copy()
    trips["service_id"] = trips["service_id"].astype(str)
    trips = trips[trips["service_id"].isin(services)]

    trip_to_route = dict(zip(trips["trip_id"].astype(str), trips["route_id"].astype(str)))
    valid_trips = set(trip_to_route.keys())

    trip_label: Dict[str, str] = {}
    for _, r in trips.iterrows():
        tid = str(r["trip_id"])
        label = tid
        if has_short:
            sn = str(r.get("trip_short_name") or "")
            if sn and sn != "nan":
                label = sn
        if label == tid and has_headsign:
            hs = str(r.get("trip_headsign") or "")
            if hs and hs != "nan":
                label = hs
        trip_label[tid] = label

    routes = read_gtfs_csv(gtfs_dir, "routes.txt", usecols=["route_id", "route_short_name", "route_long_name"])
    route_label: Dict[str, str] = {}
    for _, r in routes.iterrows():
        rid = str(r["route_id"])
        s = str(r.get("route_short_name") or "")
        l = str(r.get("route_long_name") or "")
        route_label[rid] = (s if s and s != "nan" else l) or rid

    stops = read_gtfs_csv(gtfs_dir, "stops.txt", usecols=["stop_id", "stop_name"])
    stop_name = dict(zip(stops["stop_id"].astype(str), stops["stop_name"].astype(str)))

    st = read_gtfs_csv(
        gtfs_dir,
        "stop_times.txt",
        usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
    )
    st["trip_id"] = st["trip_id"].astype(str)
    st = st[st["trip_id"].isin(valid_trips)].copy()
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce").fillna(0).astype(int)
    st.sort_values(["trip_id", "stop_sequence"], inplace=True)

    conns: List[Connection] = []
    for trip_id, grp in st.groupby("trip_id", sort=False):
        tid = str(trip_id)
        rid = trip_to_route.get(tid)
        if rid is None:
            continue

        rows = grp.to_dict("records")
        if len(rows) < 2:
            continue

        for a, b in zip(rows, rows[1:]):
            u = str(a["stop_id"])
            v = str(b["stop_id"])

            at_dep = a.get("departure_time")
            bt_arr = b.get("arrival_time")
            if pd.isna(at_dep) or pd.isna(bt_arr):
                continue

            try:
                dep = time_to_sec(str(at_dep))
                arr = time_to_sec(str(bt_arr))
            except Exception:
                continue

            if arr < dep:
                continue

            conns.append(Connection(dep=dep, arr=arr, u=u, v=v, trip_id=tid, route_id=str(rid)))

    conns.sort(key=lambda c: c.dep)
    return conns, stop_name, route_label, trip_label

def build_footpaths(
    gtfs_dir: str,
    stops_xy_2154: pd.DataFrame,
    default_parent_transfer_sec: int = 300,
    max_walk_m: float = 350.0,
    walk_speed_mps: float = 1.3,
) -> Dict[str, List[Tuple[str, int]]]:
    foot: Dict[str, List[Tuple[str, int]]] = {}

    transfers_path = os.path.join(gtfs_dir, "transfers.txt")
    if os.path.exists(transfers_path):
        tr = read_gtfs_csv(gtfs_dir, "transfers.txt", usecols=["from_stop_id", "to_stop_id", "min_transfer_time"])
        for _, r in tr.iterrows():
            u = str(r["from_stop_id"])
            v = str(r["to_stop_id"])
            t = r.get("min_transfer_time")
            sec = int(t) if pd.notna(t) and str(t).strip() != "" else default_parent_transfer_sec
            foot.setdefault(u, []).append((v, sec))
        return foot

    stops = read_gtfs_csv(gtfs_dir, "stops.txt", usecols=["stop_id", "parent_station"])
    stops["stop_id"] = stops["stop_id"].astype(str)

    for parent, grp in stops.dropna(subset=["parent_station"]).groupby("parent_station"):
        ids = grp["stop_id"].tolist()
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i != j:
                    foot.setdefault(ids[i], []).append((ids[j], default_parent_transfer_sec))
    coords = stops_xy_2154[["x", "y"]].to_numpy(float)
    stop_ids = stops_xy_2154["stop_id"].astype(str).to_numpy()
    tree = cKDTree(coords)

    pairs = tree.query_pairs(r=max_walk_m)
    for i, j in pairs:
        u = stop_ids[i]
        v = stop_ids[j]
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        dist = math.hypot(dx, dy)
        sec = int(dist / walk_speed_mps)
        foot.setdefault(u, []).append((v, sec))
        foot.setdefault(v, []).append((u, sec))

    return foot

@dataclass
class Pred:
    prev_stop: Optional[str]
    via_kind: str
    trip_id: Optional[str]
    route_id: Optional[str]
    dep: Optional[int]
    arr: Optional[int]

def relax_footpaths(
    earliest: Dict[str, int],
    pred: Dict[str, Pred],
    foot: Dict[str, List[Tuple[str, int]]],
    start_stop: str,
):
    pq = [(earliest[start_stop], start_stop)]
    seen = set()
    while pq:
        t, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        for v, w in foot.get(u, []):
            nt = t + w
            if nt < earliest.get(v, 10**18):
                earliest[v] = nt
                pred[v] = Pred(prev_stop=u, via_kind="walk", trip_id=None, route_id=None, dep=t, arr=nt)
                heapq.heappush(pq, (nt, v))

def csa_earliest_arrival(
    conns: List[Connection],
    foot: Dict[str, List[Tuple[str, int]]],
    origin_stops: List[Tuple[str, int]],
    departure_sec: int,
    start_index: int = 0,
) -> Tuple[Dict[str, int], Dict[str, Pred]]:
    INF = 10**18
    earliest: Dict[str, int] = {}
    pred: Dict[str, Pred] = {}

    for sid, access in origin_stops:
        t0 = departure_sec + access
        if t0 < earliest.get(sid, INF):
            earliest[sid] = t0
            pred[sid] = Pred(prev_stop=None, via_kind="walk", trip_id=None, route_id=None, dep=departure_sec, arr=t0)
            relax_footpaths(earliest, pred, foot, sid)

    for c in conns[start_index:]:
        tu = earliest.get(c.u, INF)
        if tu <= c.dep:
            if c.arr < earliest.get(c.v, INF):
                earliest[c.v] = c.arr
                pred[c.v] = Pred(prev_stop=c.u, via_kind="train", trip_id=c.trip_id, route_id=c.route_id, dep=c.dep, arr=c.arr)
                relax_footpaths(earliest, pred, foot, c.v)

    return earliest, pred

def reconstruct(pred: Dict[str, Pred], dest: str) -> List[Tuple[str, Pred]]:
    path: List[Tuple[str, Pred]] = []
    cur = dest
    while cur in pred:
        p = pred[cur]
        path.append((cur, p))
        if p.prev_stop is None:
            break
        cur = p.prev_stop
    path.reverse()
    return path

def count_transfers(path: List[Tuple[str, Pred]]) -> int:
    last_trip = None
    transfers = 0
    for _, p in path:
        if p.via_kind != "train":
            continue
        if last_trip is None:
            last_trip = p.trip_id
        elif p.trip_id != last_trip:
            transfers += 1
            last_trip = p.trip_id
    return transfers
def pretty_itinerary(
    gtfs_dir: str,
    path: List[Tuple[str, Pred]],
    stop_name: Dict[str, str],
    route_label: Dict[str, str],
    trip_label: Dict[str, str],
) -> str:
    lines: List[str] = []
    i = 0
    while i < len(path):
        stop, p = path[i]

        if p.via_kind == "walk":
            if p.prev_stop is not None:
                a = stop_name.get(p.prev_stop, p.prev_stop)
                b = stop_name.get(stop, stop)
                mins = int((p.arr - p.dep) // 60) if p.arr is not None and p.dep is not None else 0
                lines.append(f"- Correspondance / marche: {a} -> {b} (+{mins} min)")
            i += 1
            continue

        trip = p.trip_id
        rid = p.route_id
        start_stop = p.prev_stop
        start_dep = p.dep
        end_stop = stop
        end_arr = p.arr

        j = i + 1
        while j < len(path) and path[j][1].via_kind == "train" and path[j][1].trip_id == trip:
            end_stop = path[j][0]
            end_arr = path[j][1].arr
            j += 1

        A = stop_name.get(start_stop, start_stop)
        B = stop_name.get(end_stop, end_stop)
        rlab = route_label.get(rid, rid)
        tlab = trip_label.get(trip, trip)

        lines.append(f"- Train ({rlab} / {tlab}): {A} {sec_to_hhmm(start_dep)} -> {B} {sec_to_hhmm(end_arr)}")
        lines.append("  Arrêts traversés (segment):")
        lines.append(pretty_trip_calling_pattern_segment(
            gtfs_dir,
            trip_id=trip,
            stop_name=stop_name,
            from_stop_id=start_stop,
            to_stop_id=end_stop,
            limit=None
        ))

        i = j

    return "\n".join(lines)

def best_of_day(
    conns: List[Connection],
    foot: Dict[str, List[Tuple[str, int]]],
    origin_stops: List[Tuple[str, int]],
    dest_stops_with_egress: List[Tuple[str, int]],
    start_hhmm: str = "06:00:00",
    end_hhmm: str = "20:00:00",
    step_minutes: int = 30,
):
    dep_list = [c.dep for c in conns]
    t0 = time_to_sec(start_hhmm)
    t1 = time_to_sec(end_hhmm)
    step = step_minutes * 60
    best = None
    best_path = None

    for dep in range(t0, t1 + 1, step):
        start_idx = bisect.bisect_left(dep_list, dep)

        earliest, pred = csa_earliest_arrival(
            conns,
            foot,
            origin_stops,
            departure_sec=dep,
            start_index=start_idx
        )

        best_arr = 10**18
        best_dest = None
        for sid, egress in dest_stops_with_egress:
            arr = earliest.get(sid, 10**18)
            if arr >= 10**18:
                continue
            arr_total = arr + egress
            if arr_total < best_arr:
                best_arr = arr_total
                best_dest = sid

        if best_dest is None:
            continue

        path = reconstruct(pred, best_dest)
        transfers = count_transfers(path)
        duration = best_arr - dep

        cand = (duration, transfers, best_arr, dep, best_dest, pred)
        if best is None or cand < best:
            best = cand
            best_path = path

    return best, best_path

def main():
    COMMUNES_SHP = "./data/communes-20220101.shp"
    GTFS_DIR = "./data/sncf_gtfs"
    DATE = "2026-02-26"
    FROM_COMMUNE = "Nevers"
    TO_COMMUNE = "Rouen"

    PROFILE_START = "06:00:00"
    PROFILE_END = "20:00:00"
    PROFILE_STEP_MIN = 30

    ACCESS_K_ORIGIN = 25
    ACCESS_K_DEST = 35
    ACCESS_RADIUS_M = 50_000

    communes = load_communes(COMMUNES_SHP)
    src_insee, src_nom = pick_commune_by_name(communes, FROM_COMMUNE)
    dst_insee, dst_nom = pick_commune_by_name(communes, TO_COMMUNE)

    stops_xy = build_stops_xy_2154(GTFS_DIR)

    foot_cache = os.path.join(GTFS_DIR, "footpaths.pkl")
    if os.path.exists(foot_cache):
        with open(foot_cache, "rb") as f:
            foot = pickle.load(f)
    else:
        foot = build_footpaths(GTFS_DIR, stops_xy)
        with open(foot_cache, "wb") as f:
            pickle.dump(foot, f)

    date_i = yyyymmdd(DATE)
    conn_cache = os.path.join(GTFS_DIR, f"connections_{date_i}.pkl")
    meta_cache = os.path.join(GTFS_DIR, f"conn_meta_{date_i}.pkl")

    if os.path.exists(conn_cache) and os.path.exists(meta_cache):
        with open(conn_cache, "rb") as f:
            conns = pickle.load(f)
        with open(meta_cache, "rb") as f:
            stop_name, route_label, trip_label = pickle.load(f)
    else:
        conns, stop_name, route_label, trip_label = build_connections_for_date(GTFS_DIR, date_i)
        with open(conn_cache, "wb") as f:
            pickle.dump(conns, f)
        with open(meta_cache, "wb") as f:
            pickle.dump((stop_name, route_label, trip_label), f)

    if not conns:
        print("Aucune connection trouvée. Vérifie DATE et calendar_dates.txt.")
        return

    origin = commune_access_stops(
        communes, src_insee, stops_xy,
        k=ACCESS_K_ORIGIN, max_radius_m=ACCESS_RADIUS_M
    )
    dest = commune_access_stops(
        communes, dst_insee, stops_xy,
        k=ACCESS_K_DEST, max_radius_m=ACCESS_RADIUS_M
    )
    dest_with_egress = dest

    if not origin:
        print(f"Aucun stop trouvé autour de {src_nom} (augmente ACCESS_RADIUS_M).")
        return
    if not dest:
        print(f"Aucun stop trouvé autour de {dst_nom} (augmente ACCESS_RADIUS_M).")
        return

    best, path = best_of_day(
        conns, foot,
        origin_stops=origin,
        dest_stops_with_egress=dest_with_egress,
        start_hhmm=PROFILE_START,
        end_hhmm=PROFILE_END,
        step_minutes=PROFILE_STEP_MIN,
    )

    if best is None or path is None:
        print("Aucun itinéraire trouvé sur cette journée.")
        print("Essaye: autre DATE, élargis ACCESS_RADIUS_M, ou step=60.")
        return

    duration, transfers, arr_abs, dep_abs, best_dest, _pred = best

    print(f"Départ : {src_nom} (commune)")
    print(f"Arrivée: {dst_nom} (commune)")
    print(f"Date   : {DATE}")
    print(f"Départ choisi : {sec_to_hhmm(dep_abs)}")
    print(f"Arrivée      : {sec_to_hhmm(arr_abs)}")
    print(f"Durée        : {int(duration // 60)} min")
    print(f"Correspondances (changement de trip) : {transfers}\n")

    print("Itinéraire RÉEL (trip_id + stop_times):")
    print(pretty_itinerary(GTFS_DIR, path, stop_name, route_label, trip_label))


if __name__ == "__main__":
    main()