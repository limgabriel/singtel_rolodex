import json
import math
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Stakeholder Rolodex", page_icon="🕸️", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_FILE = APP_DIR / "sample_kumu_stakeholder_map.xlsx"

CATEGORY_COLORS = {
    "Bilateral Channel": "#3B82F6",
    "Policy/Regulatory": "#8B5CF6",
    "Think Tank": "#06B6D4",
    "Private Capital": "#F59E0B",
    "Media": "#EC4899",
    "Business Council": "#10B981",
    "Sovereign Capital": "#6366F1",
    "Private/State Capital": "#B45309",
    "Capital Markets": "#14B8A6",
    "Self": "#EF4444",
    "Unknown": "#64748B",
}


def find_default_workbook():
    if DEFAULT_FILE.exists():
        return DEFAULT_FILE
    candidates = [p for p in APP_DIR.glob("*.xlsx") if not p.name.startswith("~$")]
    return candidates[0] if candidates else None


def safe(value, default=""):
    if pd.isna(value):
        return default
    return str(value).strip()


def safe_num(value, default=1.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@st.cache_data(show_spinner=False)
def load_workbook(file_obj):
    elements = pd.read_excel(file_obj, sheet_name="Elements")
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    connections = pd.read_excel(file_obj, sheet_name="Connections")
    return elements, connections


def normalize_data(elements, connections):
    e = elements.copy()
    c = connections.copy()

    element_cols = [
        "Label", "Type", "category", "country", "institution", "role", "notes",
        "Description", "relevance", "reachability", "Alignment", "relationship_status"
    ]
    connection_cols = ["From", "To", "Direction", "edge_type", "status", "evidence", "notes"]

    for col in element_cols:
        if col not in e.columns:
            e[col] = ""
    for col in connection_cols:
        if col not in c.columns:
            c[col] = ""

    for col in ["Type", "category", "country"]:
        e[col] = e[col].fillna("Unknown").astype(str)
    for col in ["edge_type", "status"]:
        c[col] = c[col].fillna("Unknown").astype(str)

    # Kumu exports can reference a node in Connections even when it is missing from Elements.
    known = set(e["Label"].dropna().astype(str))
    referenced = set(c["From"].dropna().astype(str)) | set(c["To"].dropna().astype(str))
    missing = sorted(x for x in referenced - known if x and x.lower() != "nan")
    if missing:
        inferred = []
        for label in missing:
            is_self = label.lower() in {"you", "self", "me"}
            inferred.append({
                "Label": label,
                "Type": "Person" if is_self else "Unknown",
                "category": "Self" if is_self else "Unknown",
                "country": "Unknown",
                "institution": "",
                "role": "",
                "notes": "",
                "Description": "",
                "relevance": 8 if is_self else 1,
                "reachability": 8 if is_self else 1,
                "Alignment": "",
                "relationship_status": "",
            })
        e = pd.concat([e, pd.DataFrame(inferred)], ignore_index=True)

    return e, c


def build_adjacency(labels, connections):
    adj = {x: set() for x in labels}
    for _, r in connections.iterrows():
        a, b = safe(r.get("From")), safe(r.get("To"))
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def bfs_distances(root, adjacency):
    distances = {root: 0}
    q = deque([root])
    while q:
        cur = q.popleft()
        for nxt in adjacency.get(cur, set()):
            if nxt not in distances:
                distances[nxt] = distances[cur] + 1
                q.append(nxt)
    return distances


def radial_positions(nodes_meta, connections, root):
    """Create a category-clustered radial layout around the selected root.

    Each category receives its own angular sector. Dense sectors use multiple
    staggered radial bands so nodes do not pile up on a single ring.
    """
    labels = [n["id"] for n in nodes_meta]
    if not labels:
        return {}
    if root not in labels:
        root = "You" if "You" in labels else labels[0]

    adjacency = build_adjacency(set(labels), connections)
    dist = bfs_distances(root, adjacency)

    layers = defaultdict(list)
    disconnected = []
    for n in nodes_meta:
        label = n["id"]
        if label in dist:
            layers[dist[label]].append(n)
        else:
            disconnected.append(n)

    positions = {root: (0, 0)}
    base_radii = {1: 360, 2: 690, 3: 980, 4: 1240}

    for depth in sorted(k for k in layers if k > 0):
        layer = layers[depth]
        if not layer:
            continue

        category_groups = defaultdict(list)
        for n in layer:
            category_groups[n.get("category", "Unknown")].append(n)
        for group in category_groups.values():
            group.sort(key=lambda n: n["label"])

        # Larger category gaps create visually distinct clusters. Small groups
        # still get enough angular room to avoid being squeezed between large ones.
        ordered_groups = sorted(category_groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        group_count = len(ordered_groups)
        gap = 0.13 if group_count <= 6 else 0.095
        total_gap = min(1.15, gap * group_count)
        usable = max(2.5, 2 * math.pi - total_gap)
        weights = [max(2.2, len(group) ** 0.82) for _, group in ordered_groups]
        weight_total = sum(weights)

        cursor = -math.pi / 2
        base_radius = base_radii.get(depth, 1240 + (depth - 4) * 290)

        for (category, group), weight in zip(ordered_groups, weights):
            group_span = usable * weight / weight_total
            count = len(group)

            # Dense clusters use multiple concentric bands instead of one ring.
            if depth == 1:
                per_band = 8
                band_step = 105
            elif depth == 2:
                per_band = 10
                band_step = 120
            else:
                per_band = 12
                band_step = 135
            band_count = max(1, math.ceil(count / per_band))

            for j, n in enumerate(group):
                band = j % band_count
                slot = j // band_count
                slots_in_band = math.ceil((count - band) / band_count)
                frac = (slot + 0.5) / max(1, slots_in_band)
                # Keep nodes away from the exact sector boundaries.
                angle = cursor + group_span * (0.08 + 0.84 * frac)
                radius = base_radius + band * band_step
                # Gentle alternating offset prevents perfectly aligned labels.
                radius += 18 if (slot + band) % 2 else -18
                positions[n["id"]] = (radius * math.cos(angle), radius * math.sin(angle))

            cursor += group_span + gap

    if disconnected:
        radius = max(base_radii.values()) + 350
        ordered = sorted(disconnected, key=lambda x: (x.get("category", "Unknown"), x["label"]))
        for i, n in enumerate(ordered):
            angle = 2 * math.pi * i / max(1, len(ordered))
            positions[n["id"]] = (radius * math.cos(angle), radius * math.sin(angle))

    return positions

def prepare_graph(elements, connections, countries, categories, entity_types, edge_types, statuses, root_name, scope_hops):
    e = elements.copy()
    c = connections.copy()

    e = e[
        e["country"].isin(countries)
        & e["category"].isin(categories)
        & e["Type"].isin(entity_types)
    ].copy()

    allowed_labels = set(e["Label"].dropna().astype(str))
    c = c[
        c["From"].astype(str).isin(allowed_labels)
        & c["To"].astype(str).isin(allowed_labels)
        & c["edge_type"].isin(edge_types)
        & c["status"].isin(statuses)
    ].copy()

    if root_name not in allowed_labels:
        root_name = "You" if "You" in allowed_labels else (next(iter(allowed_labels)) if allowed_labels else "")

    if scope_hops is not None and root_name:
        adjacency = build_adjacency(allowed_labels, c)
        visible = {root_name}
        frontier = {root_name}
        for _ in range(scope_hops):
            nxt = set()
            for n in frontier:
                nxt.update(adjacency.get(n, set()))
            nxt -= visible
            visible |= nxt
            frontier = nxt
        e = e[e["Label"].astype(str).isin(visible)].copy()
        visible_labels = set(e["Label"].astype(str))
        c = c[c["From"].astype(str).isin(visible_labels) & c["To"].astype(str).isin(visible_labels)].copy()

    # Node size is derived from the currently visible network, not a manual field.
    # Degree = number of distinct visible neighbours for each entity.
    visible_labels = set(e["Label"].dropna().astype(str))
    adjacency = build_adjacency(visible_labels, c)
    degree_map = {label: len(neighbours) for label, neighbours in adjacency.items()}
    non_self_degrees = [
        degree for label, degree in degree_map.items()
        if label.lower() not in {"you", "self", "me"}
    ]
    max_degree = max(non_self_degrees, default=max(degree_map.values(), default=1)) or 1

    nodes = []
    for _, row in e.iterrows():
        label = safe(row["Label"])
        category = safe(row.get("category"), "Unknown") or "Unknown"
        relevance = safe_num(row.get("relevance"), 1)
        reachability = safe_num(row.get("reachability"), 1)
        node_type = safe(row.get("Type"), "Unknown") or "Unknown"
        is_self = category == "Self" or label.lower() in {"you", "self", "me"}
        degree = degree_map.get(label, 0)
        # Compact square-root scaling keeps hubs noticeable without overwhelming the map.
        degree_ratio = min(1.0, degree / max_degree) if max_degree else 0.0
        node_size = 12.0 + 10.0 * math.sqrt(degree_ratio)
        if is_self:
            node_size = 17.0

        details = {
            "Type": node_type,
            "Category": category,
            "Country": safe(row.get("country")),
            "Institution": safe(row.get("institution")),
            "Role": safe(row.get("role")),
            "Relevance": relevance,
            "Reachability": reachability,
            "Alignment": safe(row.get("Alignment")),
            "Relationship status": safe(row.get("relationship_status")),
            "Network connections": degree,
            "Description": safe(row.get("Description")),
            "Notes": safe(row.get("notes")),
        }

        nodes.append({
            "id": label,
            "label": label,
            "category": category,
            "nodeType": node_type,
            "shape": "diamond" if is_self else ("box" if node_type.lower() == "organisation" else "dot"),
            "size": round(node_size, 1),
            "color": {
                "background": CATEGORY_COLORS.get(category, CATEGORY_COLORS["Unknown"]),
                "border": "#F8FAFC" if not is_self else "#FBBF24",
                "highlight": {"background": CATEGORY_COLORS.get(category, CATEGORY_COLORS["Unknown"]), "border": "#FFFFFF"},
            },
            "font": {"color": "#F8FAFC", "size": 13, "face": "Inter, Arial", "strokeWidth": 4, "strokeColor": "#0B1020"},
            "relevance": relevance,
            "degree": degree,
            "borderWidth": 4 if is_self else 1.5,
            "details": details,
        })

    positions = radial_positions(nodes, c, root_name)
    for n in nodes:
        x, y = positions.get(n["id"], (0, 0))
        n["x"] = round(x, 2)
        n["y"] = round(y, 2)
        n["fixed"] = False

    edges = []
    for i, (_, row) in enumerate(c.iterrows()):
        edge_type = safe(row.get("edge_type"), "Unknown") or "Unknown"
        status = safe(row.get("status"), "Unknown") or "Unknown"
        direction = safe(row.get("Direction")).lower()
        is_contact = edge_type == "My Contact"
        edges.append({
            "id": f"e{i}",
            "from": safe(row["From"]),
            "to": safe(row["To"]),
            "relationship": edge_type,
            "status": status,
            "arrows": {"to": {"enabled": direction == "directed", "scaleFactor": 0.55}},
            "dashes": status.lower() == "historical",
            "width": 1.8 if is_contact else 1.1,
            "color": {
                "color": "rgba(245,158,11,.38)" if is_contact else "rgba(100,116,139,.32)",
                "highlight": "#F8FAFC",
                "hover": "#CBD5E1",
                "inherit": False,
            },
            "smooth": {"enabled": True, "type": "continuous", "roundness": 0.08},
            "details": {
                "Relationship": edge_type,
                "Status": status,
                "Evidence": safe(row.get("evidence")),
                "Notes": safe(row.get("notes")),
            },
        })

    return nodes, edges, root_name


def graph_html(nodes, edges, root_name, jump_to, height=840):
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    root_json = json.dumps(root_name, ensure_ascii=False)
    jump_json = json.dumps(jump_to or "", ensure_ascii=False)
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ box-sizing:border-box; }}
  html, body {{ margin:0; padding:0; background:#090E19; font-family:Inter,Segoe UI,Arial,sans-serif; overflow:hidden; }}
  #wrap {{ position:relative; width:100%; height:{height}px; background:radial-gradient(circle at 45% 45%, #111C31 0%, #090E19 70%); border:1px solid #1E293B; border-radius:16px; overflow:hidden; }}
  #network {{ width:100%; height:100%; }}
  #toolbar {{ position:absolute; z-index:12; top:14px; left:14px; display:flex; gap:7px; }}
  .btn {{ background:rgba(15,23,42,.94); color:#E2E8F0; border:1px solid #334155; border-radius:9px; padding:8px 11px; cursor:pointer; font-size:12px; box-shadow:0 5px 18px rgba(0,0,0,.18); }}
  .btn:hover {{ background:#1E293B; border-color:#475569; }}
  #panel {{ position:absolute; z-index:11; top:14px; right:14px; width:340px; max-height:calc(100% - 28px); overflow:auto; background:rgba(10,16,29,.97); color:#E5E7EB; border:1px solid #334155; border-radius:15px; padding:0; box-shadow:0 16px 40px rgba(0,0,0,.45); display:none; }}
  #panelHead {{ padding:18px 18px 14px; border-bottom:1px solid #253247; position:sticky; top:0; background:rgba(10,16,29,.98); z-index:2; }}
  #panelTitle {{ margin:0; font-size:20px; color:white; line-height:1.2; }}
  #panelSubtitle {{ color:#94A3B8; margin-top:6px; font-size:12px; }}
  #panelBody {{ padding:15px 18px 18px; }}
  .sectionTitle {{ margin:17px 0 9px; color:#94A3B8; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.12em; }}
  .detailGrid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
  .tile {{ border:1px solid #263349; background:#101827; border-radius:9px; padding:8px 10px; min-height:53px; }}
  .k {{ color:#94A3B8; font-size:9px; text-transform:uppercase; letter-spacing:.06em; }}
  .v {{ color:#F8FAFC; font-size:12px; margin-top:4px; line-height:1.35; }}
  .relation {{ padding:9px 0; border-bottom:1px solid #1E293B; }}
  .relation:last-child {{ border-bottom:none; }}
  .relName {{ color:#F8FAFC; font-size:12px; font-weight:600; }}
  .relMeta {{ color:#94A3B8; font-size:10px; margin-top:3px; }}
  .badge {{ display:inline-block; border:1px solid #334155; background:#111827; color:#CBD5E1; border-radius:999px; padding:3px 7px; font-size:9px; margin:5px 4px 0 0; }}
  .score {{ display:flex; gap:3px; margin-top:6px; }}
  .dot {{ width:8px; height:8px; border-radius:50%; background:#334155; }}
  .dot.on {{ background:#F59E0B; }}
  .definitionBox {{ margin-top:10px; border:1px solid #263349; background:#0D1524; border-radius:10px; padding:10px 11px; }}
  .definitionRow {{ margin:7px 0; color:#CBD5E1; font-size:10px; line-height:1.45; }}
  .definitionRow b {{ color:#F8FAFC; }}
  #hint {{ position:absolute; left:16px; bottom:14px; color:#7F8EA3; font-size:10px; background:rgba(9,14,25,.78); padding:7px 10px; border-radius:8px; border:1px solid rgba(51,65,85,.5); }}
  #rootPill {{ position:absolute; z-index:10; top:16px; left:50%; transform:translateX(-50%); background:rgba(15,23,42,.86); border:1px solid #334155; color:#CBD5E1; border-radius:999px; padding:7px 11px; font-size:11px; pointer-events:none; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="toolbar">
    <button class="btn" id="fitBtn">Fit map</button>
    <button class="btn" id="resetBtn">Reset selection</button>
    <button class="btn" id="labelsBtn">Labels: Auto</button>
  </div>
  <div id="rootPill">Map centred on <strong id="rootLabel"></strong></div>
  <div id="network"></div>
  <aside id="panel">
    <div id="panelHead"><h3 id="panelTitle"></h3><div id="panelSubtitle"></div></div>
    <div id="panelBody"></div>
  </aside>
  <div id="hint">Scroll to zoom · drag to pan · nodes auto-separate · click a stakeholder to trace its full relationship branch</div>
</div>
<script>
  const originalNodes = {nodes_json};
  const originalEdges = {edges_json};
  const rootName = {root_json};
  const jumpTo = {jump_json};
  document.getElementById('rootLabel').textContent = rootName;

  const nodes = new vis.DataSet(originalNodes);
  const edges = new vis.DataSet(originalEdges);
  const container = document.getElementById('network');
  const options = {{
    autoResize:true,
    layout:{{improvedLayout:false}},
    interaction:{{hover:true, multiselect:false, navigationButtons:false, keyboard:true, tooltipDelay:220}},
    physics:{{
      enabled:true,
      solver:'barnesHut',
      barnesHut:{{
        gravitationalConstant:-5200,
        centralGravity:0.025,
        springLength:235,
        springConstant:0.018,
        damping:0.72,
        avoidOverlap:1
      }},
      stabilization:{{enabled:true,iterations:220,updateInterval:25,fit:false}}
    }},
    nodes:{{
      chosen:{{node:(values)=>{{values.borderWidth=4;}}, label:(values)=>{{values.size=16; values.mod='bold';}}}},
      shadow:{{enabled:true,size:8,x:0,y:2,color:'rgba(0,0,0,.38)'}},
      margin:10
    }},
    edges:{{selectionWidth:2, hoverWidth:1.5, chosen:true}}
  }};
  const network = new vis.Network(container, {{nodes,edges}}, options);
  let labelMode = 'auto';
  let selectedNodeId = null;
  let hoveredNodeId = null;
  let labelRefreshTimer = null;
  let activeTraceNodes = null;
  let activeTraceEdges = null;
  let collisionSettlePending = true;
  const hubName = originalNodes.some(n => n.id === 'You') ? 'You' : rootName;

  function buildAdjacency() {{
    const adj = new Map();
    originalNodes.forEach(n => adj.set(n.id, []));
    originalEdges.forEach(e => {{
      if (!adj.has(e.from)) adj.set(e.from, []);
      if (!adj.has(e.to)) adj.set(e.to, []);
      adj.get(e.from).push({{node:e.to, edge:e.id}});
      adj.get(e.to).push({{node:e.from, edge:e.id}});
    }});
    return adj;
  }}

  const graphAdjacency = buildAdjacency();

  function traceRelationshipBranch(startId) {{
    const seenNodes = new Set([startId]);
    const seenEdges = new Set();
    const queue = [startId];
    while (queue.length) {{
      const cur = queue.shift();
      // 'You' is shown as a linked endpoint, but is not used to fan out into
      // every other direct contact unless the user actually selected 'You'.
      if (cur === hubName && startId !== hubName) continue;
      const neighbours = graphAdjacency.get(cur) || [];
      for (const item of neighbours) {{
        seenEdges.add(item.edge);
        if (!seenNodes.has(item.node)) {{
          seenNodes.add(item.node);
          queue.push(item.node);
        }}
      }}
    }}
    // Only keep edges whose endpoints are both in the traced branch. This
    // removes unrelated spokes from a terminal hub such as 'You'.
    const cleanEdges = new Set();
    originalEdges.forEach(e => {{
      if (seenNodes.has(e.from) && seenNodes.has(e.to)) {{
        if (!(startId !== hubName && (e.from === hubName || e.to === hubName) && e.from !== startId && e.to !== startId)) {{
          cleanEdges.add(e.id);
        }}
      }}
    }});
    return {{nodes:seenNodes, edges:cleanEdges}};
  }}

  function settleCollisions(iterations=100) {{
    collisionSettlePending = true;
    network.setOptions({{physics:{{
      enabled:true,
      solver:'barnesHut',
      barnesHut:{{gravitationalConstant:-5200,centralGravity:0.025,springLength:235,springConstant:0.018,damping:0.72,avoidOverlap:1}},
      stabilization:{{enabled:true,iterations:iterations,fit:false}}
    }}}});
    network.stabilize(iterations);
  }}

  function transparentFont(n) {{
    const f = Object.assign({{}}, n.font || {{}});
    f.color = 'rgba(248,250,252,0)';
    f.strokeColor = 'rgba(11,16,32,0)';
    return f;
  }}

  function visibleFont(n, size=null) {{
    const f = Object.assign({{}}, n.font || {{}});
    f.color = '#F8FAFC';
    f.strokeColor = '#0B1020';
    f.strokeWidth = 4;
    if (size !== null) f.size = size;
    return f;
  }}

  function boxesOverlap(a,b,pad=5) {{
    return !(a.x2 + pad < b.x1 || b.x2 + pad < a.x1 || a.y2 + pad < b.y1 || b.y2 + pad < a.y1);
  }}

  function refreshLabels() {{
    if (!network || !nodes.length) return;
    const scale = network.getScale();
    const all = originalNodes.slice();
    const inTrace = (id) => !activeTraceNodes || activeTraceNodes.has(id);

    if (labelMode === 'all') {{
      nodes.update(all.map(n => ({{id:n.id, font:inTrace(n.id) ? visibleFont(n, n.nodeType && n.nodeType.toLowerCase()==='organisation' ? 12 : 13) : transparentFont(n)}})));
      return;
    }}
    if (labelMode === 'none') {{
      nodes.update(all.map(n => ({{id:n.id, font:(inTrace(n.id) && (n.id===rootName || n.id===selectedNodeId || n.id===hoveredNodeId)) ? visibleFont(n,14) : transparentFont(n)}})));
      return;
    }}

    const updates = [];
    const occupied = [];
    const people = [];

    for (const n of all) {{
      if (!inTrace(n.id)) {{
        updates.push({{id:n.id, font:transparentFont(n)}});
        continue;
      }}
      const type = String(n.nodeType || '').toLowerCase();
      if (type === 'organisation') {{
        updates.push({{id:n.id, font:visibleFont(n, scale < 0.7 ? 10 : 11)}});
      }} else {{
        const degree = network.getConnectedEdges(n.id).length;
        const relevance = Number(n.relevance || (n.details && n.details.Relevance) || 0);
        const forced = n.id===rootName || n.id===selectedNodeId || n.id===hoveredNodeId;
        people.push({{n, degree, relevance, forced}});
      }}
    }}

    let relevanceFloor = 0;
    if (scale < 0.48) relevanceFloor = 8;
    else if (scale < 0.62) relevanceFloor = 6;
    else if (scale < 0.78) relevanceFloor = 4;
    else if (scale < 0.95) relevanceFloor = 2;

    people.sort((a,b) =>
      Number(b.forced)-Number(a.forced) ||
      b.relevance-a.relevance ||
      b.degree-a.degree ||
      String(a.n.label).localeCompare(String(b.n.label))
    );

    for (const item of people) {{
      const n = item.n;
      const forced = item.forced;
      const show = forced || item.relevance >= relevanceFloor;
      if (!show) {{
        updates.push({{id:n.id, font:transparentFont(n)}});
        continue;
      }}

      const pos = network.getPositions([n.id])[n.id];
      if (!pos) {{
        updates.push({{id:n.id, font:transparentFont(n)}});
        continue;
      }}
      const p = network.canvasToDOM(pos);
      const nodeRadius = Math.max(12, Number(n.size || 20) * Math.max(0.55, Math.min(scale,1.1)));
      const fontSize = forced ? 14 : (scale < 0.65 ? 10 : scale < 0.95 ? 11 : 12);
      const width = Math.min(180, Math.max(42, String(n.label || '').length * fontSize * 0.55));
      const box = {{x1:p.x-width/2, x2:p.x+width/2, y1:p.y+nodeRadius+2, y2:p.y+nodeRadius+fontSize+7}};
      const collision = !forced && occupied.some(o => boxesOverlap(box,o,4));
      if (collision) {{
        updates.push({{id:n.id, font:transparentFont(n)}});
      }} else {{
        updates.push({{id:n.id, font:visibleFont(n,fontSize)}});
        occupied.push(box);
      }}
    }}
    nodes.update(updates);
  }}

  function scheduleLabelRefresh(delay=35) {{
    if (labelRefreshTimer) clearTimeout(labelRefreshTimer);
    labelRefreshTimer = setTimeout(refreshLabels, delay);
  }}

  function esc(v) {{
    return String(v ?? '').replace(/[&<>'"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}}[c]));
  }}
  function scoreDots(value) {{
    let v = Number(value || 0); let count = v >= 7 ? 3 : v >= 4 ? 2 : v > 0 ? 1 : 0;
    return `<div class="score">${{[0,1,2].map(i=>`<span class="dot ${{i<count?'on':''}}"></span>`).join('')}}</div>`;
  }}
  function relationRows(nodeId) {{
    const ids = network.getConnectedEdges(nodeId);
    const rows = ids.map(id => {{
      const e = edges.get(id);
      const other = e.from === nodeId ? e.to : e.from;
      return {{name:other, relationship:e.relationship || 'Relationship', status:e.status || ''}};
    }}).sort((a,b)=>a.relationship.localeCompare(b.relationship)||a.name.localeCompare(b.name));
    if(!rows.length) return '<div class="relMeta">No visible relationships under the current filters.</div>';
    return rows.map(r=>`<div class="relation"><div class="relName">${{esc(r.name)}}</div><div class="relMeta">${{esc(r.relationship)}}${{r.status ? ' · '+esc(r.status):''}}</div></div>`).join('');
  }}
  function showNodePanel(id) {{
    const n = nodes.get(id); const d = n.details || {{}};
    document.getElementById('panelTitle').textContent = n.label;
    document.getElementById('panelSubtitle').textContent = [d.Role,d.Institution].filter(Boolean).join(' · ') || [d.Type,d.Country].filter(Boolean).join(' · ');
    const badges = [d.Category,d.Country,d.Type].filter(Boolean).map(x=>`<span class="badge">${{esc(x)}}</span>`).join('');
    const desc = d.Description ? `<div class="sectionTitle">Profile</div><div class="v">${{esc(d.Description)}}</div>` : '';
    const notes = d.Notes ? `<div class="sectionTitle">Notes</div><div class="v">${{esc(d.Notes)}}</div>` : '';
    document.getElementById('panelBody').innerHTML = `
      <div>${{badges}}</div>
      <div class="sectionTitle">Stakeholder profile</div>
      <div class="detailGrid">
        <div class="tile"><div class="k">Relevance</div><div class="v">${{esc(d.Relevance || '—')}}</div>${{scoreDots(d.Relevance)}}</div>
        <div class="tile"><div class="k">Reachability</div><div class="v">${{esc(d.Reachability || '—')}}</div>${{scoreDots(d.Reachability)}}</div>
        <div class="tile"><div class="k">Alignment</div><div class="v">${{esc(d.Alignment || '—')}}</div>${{scoreDots(d.Alignment)}}</div>
        <div class="tile"><div class="k">Status</div><div class="v">${{esc(d['Relationship status'] || '—')}}</div></div>
        <div class="tile"><div class="k">Network connections</div><div class="v">${{esc(d['Network connections'] ?? 0)}}</div><div class="relMeta">Drives node size</div></div>
      </div>
      <div class="sectionTitle">How to read these fields</div>
      <div class="definitionBox">
        <div class="definitionRow"><b>Relevance</b> — how important this stakeholder is to the objective or topic you are mapping.</div>
        <div class="definitionRow"><b>Reachability</b> — how easily your team can access the stakeholder directly or through existing relationships.</div>
        <div class="definitionRow"><b>Alignment</b> — how supportive, compatible or positively disposed the stakeholder is toward the objective.</div>
        <div class="definitionRow"><b>Status</b> — the current relationship stage recorded in the workbook, such as current, historical or target.</div>
        <div class="definitionRow"><b>Node size</b> — calculated automatically from the number of visible connected entities; it is not based on Relevance.</div>
      </div>
      ${{desc}}${{notes}}
      <div class="sectionTitle">Visible relationships (${{network.getConnectedEdges(id).length}})</div>
      ${{relationRows(id)}}`;
    document.getElementById('panel').style.display='block';
  }}
  function showEdgePanel(edgeId) {{
    const e = edges.get(edgeId); const d=e.details||{{}};
    document.getElementById('panelTitle').textContent = `${{e.from}} → ${{e.to}}`;
    document.getElementById('panelSubtitle').textContent = e.relationship || 'Relationship';
    document.getElementById('panelBody').innerHTML = Object.entries(d).filter(([k,v])=>v!==null&&v!==undefined&&String(v).trim()!=='').map(([k,v])=>`<div class="tile" style="margin-bottom:8px"><div class="k">${{esc(k)}}</div><div class="v">${{esc(v)}}</div></div>`).join('');
    document.getElementById('panel').style.display='block';
  }}
  function resetStyles() {{
    activeTraceNodes = null;
    activeTraceEdges = null;
    nodes.update(originalNodes.map(n=>({{id:n.id, hidden:false, opacity:1, font:n.font, color:n.color, borderWidth:n.borderWidth}})));
    edges.update(originalEdges.map(e=>({{id:e.id, hidden:false, color:e.color, width:e.width}})));
  }}
  function highlightNode(id) {{
    selectedNodeId = id;
    const trace = traceRelationshipBranch(id);
    activeTraceNodes = trace.nodes;
    activeTraceEdges = trace.edges;

    nodes.update(originalNodes.map(n=>({{
      id:n.id,
      hidden:!trace.nodes.has(n.id),
      opacity:1,
      color:n.color,
      borderWidth:n.id===id ? Math.max(4,n.borderWidth||1.5) : n.borderWidth
    }})));
    edges.update(originalEdges.map(e=>({{
      id:e.id,
      hidden:!trace.edges.has(e.id),
      color:trace.edges.has(e.id) ? {{color:'#CBD5E1',highlight:'#FFFFFF',hover:'#F8FAFC',inherit:false}} : e.color,
      width:trace.edges.has(e.id) ? Math.max(2.0,e.width||1) : e.width
    }})));
    network.selectNodes([id]);
    showNodePanel(id);
    scheduleLabelRefresh(0);

    const visibleIds = Array.from(trace.nodes);
    if (visibleIds.length && visibleIds.length < originalNodes.length * 0.82) {{
      setTimeout(()=>network.fit({{nodes:visibleIds, animation:{{duration:420,easingFunction:'easeInOutQuad'}}}}), 20);
    }}
  }}
  function clearSelection() {{
    selectedNodeId = null;
    resetStyles(); network.unselectAll(); document.getElementById('panel').style.display='none';
    scheduleLabelRefresh();
  }}

  network.on('zoom', () => scheduleLabelRefresh(20));
  network.on('dragEnd', params => {{
    scheduleLabelRefresh(20);
    if (params.nodes && params.nodes.length) settleCollisions(70);
  }});
  network.on('stabilized', () => {{
    if (collisionSettlePending) {{
      collisionSettlePending = false;
      network.setOptions({{physics:{{enabled:false}}}});
      scheduleLabelRefresh(30);
    }}
  }});
  network.on('hoverNode', params => {{
    hoveredNodeId = params.node;
    scheduleLabelRefresh(0);
  }});
  network.on('blurNode', params => {{
    if (hoveredNodeId === params.node) hoveredNodeId = null;
    scheduleLabelRefresh(20);
  }});

  network.on('click', params => {{
    if(params.nodes.length) highlightNode(params.nodes[0]);
    else if(params.edges.length) {{ network.selectEdges([params.edges[0]]); showEdgePanel(params.edges[0]); }}
    else clearSelection();
  }});
  network.on('doubleClick', params => {{ if(!params.nodes.length && !params.edges.length) {{ clearSelection(); network.fit({{animation:{{duration:400}}}}); }} }});
  network.on('hoverEdge', params => {{ const e=edges.get(params.edge); edges.update({{id:e.id, width:Math.max(2.5,e.width||1)}}); }});
  network.on('blurEdge', params => {{ const orig=originalEdges.find(e=>e.id===params.edge); if(orig) edges.update({{id:orig.id,width:orig.width}}); }});
  document.getElementById('fitBtn').onclick=()=>network.fit({{animation:{{duration:350}}}});
  document.getElementById('resetBtn').onclick=()=>{{clearSelection();network.fit({{animation:{{duration:350}}}});}};
  document.getElementById('labelsBtn').onclick=()=>{{
    labelMode = labelMode === 'auto' ? 'all' : (labelMode === 'all' ? 'none' : 'auto');
    document.getElementById('labelsBtn').textContent = 'Labels: ' + labelMode.charAt(0).toUpperCase() + labelMode.slice(1);
    refreshLabels();
  }};

  let initialViewApplied = false;
  function applyInitialView() {{
    if (initialViewApplied) return;
    initialViewApplied = true;
    network.setOptions({{physics:{{enabled:false}}}});
    network.fit({{animation:false}});
    setTimeout(()=>refreshLabels(), 80);
    const target = jumpTo && nodes.get(jumpTo) ? jumpTo : rootName;
    if(target && nodes.get(target)) {{
      setTimeout(()=>{{
        network.focus(target, {{scale: jumpTo ? 1.15 : 0.75, animation:{{duration:450,easingFunction:'easeInOutQuad'}}}});
        if(jumpTo) highlightNode(target);
      }}, 120);
    }}
  }}
  network.once('stabilized', applyInitialView);
  setTimeout(applyInitialView, 1800);
</script>
</body>
</html>
"""


st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 1.5rem; max-width: 100%;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(148,163,184,.18); min-width:275px;}
[data-testid="stMetric"] {background:rgba(15,23,42,.18); padding:6px 10px; border-radius:10px;}
h1 {margin-bottom:.1rem; font-size:2rem !important;}
.small-muted {color:#718096;font-size:.82rem;margin-bottom:.8rem;}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {font-size:11px;}
</style>
""", unsafe_allow_html=True)

st.title("Stakeholder Rolodex")
st.markdown('<div class="small-muted">Relationship intelligence prototype · V0.4</div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Data")
    upload = st.file_uploader("Upload Kumu-format Excel", type=["xlsx"], label_visibility="collapsed")
    local_default = find_default_workbook()
    source = upload if upload is not None else local_default
    if upload is None and local_default is not None:
        st.caption(f"Using: {local_default.name}")

if source is None:
    st.info("Upload a Kumu-format .xlsx workbook in the sidebar, or place one .xlsx file beside app.py.")
    st.stop()

try:
    elements, connections = load_workbook(source)
except Exception as exc:
    st.error(f"Could not read the workbook: {exc}")
    st.stop()

if upload is not None:
    upload.seek(0)

elements, connections = normalize_data(elements, connections)

all_countries = sorted(elements["country"].fillna("Unknown").astype(str).unique())
all_categories = sorted(elements["category"].fillna("Unknown").astype(str).unique())
all_types = sorted(elements["Type"].fillna("Unknown").astype(str).unique())
all_edge_types = sorted(connections["edge_type"].fillna("Unknown").astype(str).unique())
all_statuses = sorted(connections["status"].fillna("Unknown").astype(str).unique())
all_names = sorted(elements["Label"].dropna().astype(str).unique())

with st.sidebar:
    st.subheader("Filters")
    with st.expander("Entity filters", expanded=True):
        countries = st.multiselect("Country", all_countries, default=all_countries)
        categories = st.multiselect("Category", all_categories, default=all_categories)
        entity_types = st.multiselect("Entity type", all_types, default=all_types)
    with st.expander("Relationship filters", expanded=False):
        edge_types = st.multiselect("Relationship", all_edge_types, default=all_edge_types)
        statuses = st.multiselect("Status", all_statuses, default=all_statuses)
    st.caption("Tip: keep filters broad, then use search + scope to investigate a stakeholder.")

# Main interaction controls: search is intentionally prominent rather than buried in sidebar.
c_search, c_scope = st.columns([3, 1.15])
default_root = "You" if "You" in all_names else (all_names[0] if all_names else "")
with c_search:
    jump_to = st.selectbox(
        "Search / jump to stakeholder",
        [""] + all_names,
        index=0,
        placeholder="Type a person or organisation...",
    )
with c_scope:
    scope_label = st.selectbox("View scope", ["Entire map", "1 degree", "2 degrees", "3 degrees"], index=0)

root_name = jump_to or default_root
scope_hops = None if scope_label == "Entire map" else int(scope_label.split()[0])

nodes, edges, root_name = prepare_graph(
    elements, connections, countries, categories, entity_types,
    edge_types, statuses, root_name, scope_hops
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Entities", len(nodes))
m2.metric("Relationships", len(edges))
m3.metric("People", sum(1 for n in nodes if n["nodeType"].lower() == "person"))
m4.metric("Organisations", sum(1 for n in nodes if n["nodeType"].lower() == "organisation"))

if not nodes:
    st.warning("No entities match the current filters.")
else:
    components.html(graph_html(nodes, edges, root_name, jump_to), height=870, scrolling=False)

with st.expander("Legend"):
    active_categories = sorted({n["category"] for n in nodes}) if nodes else []
    cols = st.columns(min(4, max(1, len(active_categories))))
    for i, category in enumerate(active_categories):
        color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["Unknown"])
        cols[i % len(cols)].markdown(
            f'<span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:50%;margin-right:6px"></span>{category}',
            unsafe_allow_html=True,
        )
    st.caption("Node size is calculated from the number of visible connected entities using compact square-root scaling. Collision physics separates nodes before the map is frozen. Selecting a stakeholder traces its full connected branch; 'You' is treated as a terminal hub unless 'You' itself is selected. Relevance, reachability and alignment remain stakeholder attributes shown in the detail card.")
