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


def graph_html(nodes, edges, root_name, jump_to, theme_mode="dark", height=840):
    theme_mode = "light" if str(theme_mode).lower() == "light" else "dark"
    themes = {
        "dark": {
            "page_bg": "#090E19",
            "map_bg_1": "#111C31",
            "map_bg_2": "#090E19",
            "border": "#1E293B",
            "surface": "rgba(10,16,29,.97)",
            "surface_sticky": "rgba(10,16,29,.98)",
            "surface_2": "#101827",
            "surface_3": "#0D1524",
            "control": "rgba(15,23,42,.94)",
            "control_hover": "#1E293B",
            "text": "#F8FAFC",
            "text_2": "#E5E7EB",
            "muted": "#94A3B8",
            "muted_2": "#7F8EA3",
            "node_stroke": "#0B1020",
            "node_border": "#F8FAFC",
            "node_highlight": "#FFFFFF",
            "edge": "rgba(100,116,139,.32)",
            "edge_contact": "rgba(245,158,11,.38)",
            "edge_highlight": "#F8FAFC",
            "edge_hover": "#CBD5E1",
            "trace_edge": "#38BDF8",
            "trace_edge_contact": "#FBBF24",
            "trace_edge_shadow": "rgba(56,189,248,.42)",
            "shadow": "rgba(0,0,0,.45)",
            "hint_bg": "rgba(9,14,25,.78)",
        },
        "light": {
            "page_bg": "#F8FAFC",
            "map_bg_1": "#FFFFFF",
            "map_bg_2": "#F1F5F9",
            "border": "#CBD5E1",
            "surface": "rgba(255,255,255,.97)",
            "surface_sticky": "rgba(255,255,255,.99)",
            "surface_2": "#F8FAFC",
            "surface_3": "#F1F5F9",
            "control": "rgba(255,255,255,.96)",
            "control_hover": "#F1F5F9",
            "text": "#0F172A",
            "text_2": "#1E293B",
            "muted": "#64748B",
            "muted_2": "#64748B",
            "node_stroke": "#FFFFFF",
            "node_border": "#475569",
            "node_highlight": "#0F172A",
            "edge": "rgba(71,85,105,.30)",
            "edge_contact": "rgba(217,119,6,.42)",
            "edge_highlight": "#0F172A",
            "edge_hover": "#475569",
            "trace_edge": "#0369A1",
            "trace_edge_contact": "#B45309",
            "trace_edge_shadow": "rgba(3,105,161,.22)",
            "shadow": "rgba(15,23,42,.16)",
            "hint_bg": "rgba(255,255,255,.88)",
        },
    }
    theme = themes[theme_mode]

    themed_nodes = []
    for node in nodes:
        n = dict(node)
        font = dict(n.get("font", {}))
        font.update({"color": theme["text"], "strokeColor": theme["node_stroke"], "strokeWidth": 4})
        n["font"] = font
        color = dict(n.get("color", {}))
        color["border"] = "#FBBF24" if n.get("id") == "You" else theme["node_border"]
        highlight = dict(color.get("highlight", {}))
        highlight["border"] = theme["node_highlight"]
        color["highlight"] = highlight
        n["color"] = color
        themed_nodes.append(n)

    themed_edges = []
    for edge in edges:
        e = dict(edge)
        color = dict(e.get("color", {}))
        is_contact = e.get("relationship") == "My Contact"
        color.update({
            "color": theme["edge_contact"] if is_contact else theme["edge"],
            "highlight": theme["edge_highlight"],
            "hover": theme["edge_hover"],
            "inherit": False,
        })
        e["color"] = color
        themed_edges.append(e)

    nodes_json = json.dumps(themed_nodes, ensure_ascii=False)
    edges_json = json.dumps(themed_edges, ensure_ascii=False)
    root_json = json.dumps(root_name, ensure_ascii=False)
    jump_json = json.dumps(jump_to or "", ensure_ascii=False)
    theme_json = json.dumps(theme, ensure_ascii=False)
    themes_json = json.dumps(themes, ensure_ascii=False)
    theme_mode_json = json.dumps(theme_mode)
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ box-sizing:border-box; }}
  :root {{
    --page-bg:{theme["page_bg"]}; --map-bg-1:{theme["map_bg_1"]}; --map-bg-2:{theme["map_bg_2"]}; --border:{theme["border"]};
    --surface:{theme["surface"]}; --surface-sticky:{theme["surface_sticky"]}; --surface-2:{theme["surface_2"]}; --surface-3:{theme["surface_3"]};
    --control:{theme["control"]}; --control-hover:{theme["control_hover"]}; --text:{theme["text"]}; --text-2:{theme["text_2"]};
    --muted:{theme["muted"]}; --muted-2:{theme["muted_2"]}; --shadow:{theme["shadow"]}; --hint-bg:{theme["hint_bg"]};
  }}
  html, body {{ margin:0; padding:0; background:transparent !important; color:var(--text); font-family:Inter,Segoe UI,Arial,sans-serif; overflow:hidden; }}
  #wrap {{ position:relative; width:100%; height:{height}px; background:transparent !important; border:none; border-radius:14px; overflow:hidden; }}
  #network {{ width:100%; height:100%; }}
  #toolbar {{ position:absolute; z-index:12; top:14px; left:14px; display:flex; gap:7px; }}
  .btn {{ background:var(--control); color:var(--text-2); border:1px solid var(--border); border-radius:9px; padding:8px 11px; cursor:pointer; font-size:12px; box-shadow:0 5px 18px var(--shadow); }}
  .btn:hover {{ background:var(--control-hover); }}
  #panel {{ position:absolute; z-index:11; top:14px; right:14px; width:min(460px,calc(100% - 28px)); max-height:calc(100% - 28px); overflow:auto; background:var(--surface); color:var(--text-2); border:1px solid var(--border); border-radius:15px; padding:0; box-shadow:0 16px 40px var(--shadow); display:none; }}
  #panelHead {{ padding:22px 22px 17px; border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--surface-sticky); z-index:2; }}
  #panelTitle {{ margin:0; font-size:25px; font-weight:750; color:var(--text); line-height:1.2; }}
  #panelSubtitle {{ color:var(--muted); margin-top:8px; font-size:14px; line-height:1.45; }}
  #panelBody {{ padding:18px 22px 24px; }}
  .sectionTitle {{ margin:22px 0 11px; color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.12em; }}
  .detailGrid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
  .tile {{ border:1px solid var(--border); background:var(--surface-2); border-radius:10px; padding:12px 13px; min-height:68px; }}
  .k {{ color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.06em; }}
  .v {{ color:var(--text); font-size:15px; margin-top:5px; line-height:1.45; }}
  .relation {{ padding:12px 0; border-bottom:1px solid var(--border); }}
  .relation:last-child {{ border-bottom:none; }}
  .relName {{ color:var(--text); font-size:14px; font-weight:650; line-height:1.35; }}
  .relMeta {{ color:var(--muted); font-size:12px; margin-top:4px; line-height:1.4; }}
  .badge {{ display:inline-block; border:1px solid var(--border); background:var(--surface-3); color:var(--text-2); border-radius:999px; padding:5px 9px; font-size:11px; margin:5px 5px 0 0; }}
  .score {{ display:flex; gap:4px; margin-top:8px; }}
  .dot {{ width:9px; height:9px; border-radius:50%; background:var(--border); }}
  .dot.on {{ background:#F59E0B; }}
  #hint {{ position:absolute; left:16px; bottom:14px; color:var(--muted-2); font-size:10px; background:var(--hint-bg); padding:7px 10px; border-radius:8px; border:1px solid var(--border); }}
  #rootPill {{ position:absolute; z-index:10; top:16px; left:50%; transform:translateX(-50%); background:var(--control); border:1px solid var(--border); color:var(--text-2); border-radius:999px; padding:7px 11px; font-size:11px; pointer-events:none; }}
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
  <div id="hint">Scroll to zoom · drag to pan · static layout · drag a node to reposition it · click a stakeholder to trace its full relationship branch</div>
</div>
<script>
  const originalNodes = {nodes_json};
  const originalEdges = {edges_json};
  const rootName = {root_json};
  const jumpTo = {jump_json};
  const THEMES = {themes_json};
  const initialThemeMode = {theme_mode_json};
  let themePreference = 'auto';
  let activeThemeName = initialThemeMode;
  let theme = THEMES[activeThemeName];
  document.getElementById('rootLabel').textContent = rootName;

  const nodes = new vis.DataSet(originalNodes);
  const edges = new vis.DataSet(originalEdges);
  const container = document.getElementById('network');
  const options = {{
    autoResize:true,
    layout:{{improvedLayout:false}},
    interaction:{{hover:true, multiselect:false, navigationButtons:false, keyboard:true, tooltipDelay:220}},
    physics:{{enabled:false}},
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
  const hubName = originalNodes.some(n => n.id === 'You') ? 'You' : rootName;

  function cssRgbToMode(value) {{
    const m = String(value || '').match(/rgba?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)/i);
    if (!m) return null;
    const r = Number(m[1]), g = Number(m[2]), b = Number(m[3]);
    const lum = (0.2126*r + 0.7152*g + 0.0722*b) / 255;
    return lum >= 0.56 ? 'light' : 'dark';
  }}

  function detectParentTheme() {{
    try {{
      const doc = window.parent.document;
      const candidates = [
        doc.querySelector('[data-testid="stAppViewContainer"]'),
        doc.querySelector('.stApp'),
        doc.body,
        doc.documentElement
      ].filter(Boolean);
      for (const el of candidates) {{
        const cs = window.parent.getComputedStyle(el);
        const mode = cssRgbToMode(cs.backgroundColor);
        if (mode) return mode;
      }}
    }} catch (err) {{}}
    try {{
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }} catch (err) {{
      return initialThemeMode;
    }}
  }}

  function setCssTheme(t) {{
    const root = document.documentElement.style;
    const vars = {{
      '--page-bg':t.page_bg, '--map-bg-1':t.map_bg_1, '--map-bg-2':t.map_bg_2, '--border':t.border,
      '--surface':t.surface, '--surface-sticky':t.surface_sticky, '--surface-2':t.surface_2, '--surface-3':t.surface_3,
      '--control':t.control, '--control-hover':t.control_hover, '--text':t.text, '--text-2':t.text_2,
      '--muted':t.muted, '--muted-2':t.muted_2, '--shadow':t.shadow, '--hint-bg':t.hint_bg
    }};
    Object.entries(vars).forEach(([k,v]) => root.setProperty(k,v));
  }}

  function applyTheme(mode, force=false) {{
    const next = mode === 'light' ? 'light' : 'dark';
    if (!force && next === activeThemeName) return;
    activeThemeName = next;
    theme = THEMES[next];
    setCssTheme(theme);

    nodes.update(originalNodes.map(n => {{
      const current = nodes.get(n.id) || {{}};
      const baseColor = Object.assign({{}}, n.color || {{}});
      const highlight = Object.assign({{}}, baseColor.highlight || {{}}, {{border:theme.node_highlight}});
      baseColor.border = n.id === 'You' ? '#FBBF24' : theme.node_border;
      baseColor.highlight = highlight;
      const baseFont = Object.assign({{}}, current.font || n.font || {{}}, {{
        color: theme.text,
        strokeColor: theme.node_stroke,
        strokeWidth: 4
      }});
      if (current.font && String(current.font.color || '').includes('rgba(0,0,0,0)')) {{
        baseFont.color = 'rgba(0,0,0,0)';
        baseFont.strokeColor = 'rgba(0,0,0,0)';
      }}
      return {{id:n.id, color:baseColor, font:baseFont}};
    }}));

    edges.update(originalEdges.map(e => {{
      const current = edges.get(e.id) || {{}};
      const traced = activeTraceEdges && activeTraceEdges.has(e.id);
      const isContact = e.relationship === 'My Contact';
      return {{
        id:e.id,
        hidden:current.hidden === true,
        color:{{
          color: traced ? (isContact ? theme.trace_edge_contact : theme.trace_edge) : (isContact ? theme.edge_contact : theme.edge),
          highlight: traced ? (isContact ? theme.trace_edge_contact : theme.trace_edge) : theme.edge_highlight,
          hover: traced ? (isContact ? theme.trace_edge_contact : theme.trace_edge) : theme.edge_hover,
          inherit:false
        }},
        width: traced ? Math.max(3.2, e.width || 1) : (current.width || e.width),
        shadow: traced ? {{enabled:true,color:theme.trace_edge_shadow,size:5,x:0,y:0}} : {{enabled:false}}
      }};
    }}));
    scheduleLabelRefresh(0);
  }}

  function syncAutoTheme(force=false) {{
    if (themePreference !== 'auto') return;
    applyTheme(detectParentTheme(), force);
  }}

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



  function transparentFont(n) {{
    const f = Object.assign({{}}, n.font || {{}});
    f.color = 'rgba(0,0,0,0)';
    f.strokeColor = 'rgba(0,0,0,0)';
    return f;
  }}

  function visibleFont(n, size=null) {{
    const f = Object.assign({{}}, n.font || {{}});
    f.color = theme.text;
    f.strokeColor = theme.node_stroke;
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
      color:trace.edges.has(e.id) ? {{
        color:e.relationship === 'My Contact' ? theme.trace_edge_contact : theme.trace_edge,
        highlight:e.relationship === 'My Contact' ? theme.trace_edge_contact : theme.trace_edge,
        hover:e.relationship === 'My Contact' ? theme.trace_edge_contact : theme.trace_edge,
        inherit:false
      }} : e.color,
      width:trace.edges.has(e.id) ? Math.max(3.2,e.width||1) : e.width,
      shadow:trace.edges.has(e.id) ? {{enabled:true,color:theme.trace_edge_shadow,size:5,x:0,y:0}} : {{enabled:false}}
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
    // Physics is deliberately disabled. Dragging moves only the node(s) the
    // user touched and the rest of the map remains completely stable.
    scheduleLabelRefresh(20);
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


  // Theme contrast is supplied by Streamlit; the canvas itself stays transparent.


  let initialViewApplied = false;
  function applyInitialView() {{
    if (initialViewApplied) return;
    initialViewApplied = true;
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
  // There is no stabilization lifecycle because physics is disabled.
  // The Python-generated x/y coordinates are the authoritative layout.
  applyTheme(initialThemeMode, true);
  setTimeout(applyInitialView, 60);
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
st.markdown('<div class="small-muted">Relationship intelligence prototype · V0.10</div>', unsafe_allow_html=True)

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
c_search, c_scope, c_theme = st.columns([3, 1.15, 1.05])
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
with c_theme:
    map_appearance = st.selectbox("Map appearance", ["Auto", "Light", "Dark"], index=0, help="Auto uses Streamlit's current theme for text/edge contrast. The map canvas itself is transparent and follows the page background.")

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
    if map_appearance == "Light":
        theme_mode = "light"
    elif map_appearance == "Dark":
        theme_mode = "dark"
    else:
        try:
            theme_mode = str(st.context.theme.type).lower()
        except Exception:
            configured_base = str(st.get_option("theme.base") or "").lower()
            theme_mode = "dark" if configured_base == "dark" else "light"
        if theme_mode not in {"light", "dark"}:
            theme_mode = "light"
    components.html(graph_html(nodes, edges, root_name, jump_to, theme_mode=theme_mode), height=870, scrolling=False)

with st.expander("Legend"):
    active_categories = sorted({n["category"] for n in nodes}) if nodes else []
    cols = st.columns(min(4, max(1, len(active_categories))))
    for i, category in enumerate(active_categories):
        color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["Unknown"])
        cols[i % len(cols)].markdown(
            f'<span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:50%;margin-right:6px"></span>{category}',
            unsafe_allow_html=True,
        )
    st.markdown("**How to read the stakeholder fields**")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Relevance** — how important the stakeholder is to the objective or topic being mapped.  \
**Reachability** — how easily the team can access the stakeholder directly or through existing relationships.  \
**Alignment** — how supportive, compatible, or positively disposed the stakeholder is toward the objective.")
    with d2:
        st.markdown("**Status** — the recorded relationship stage, such as current, historical, or target.  \
**Node size** — calculated automatically from the number of visible connected entities; it is not based on Relevance.  \
**Relationship trace** — selecting a stakeholder shows their full connected branch; `You` acts as a terminal hub unless `You` itself is selected.")
    st.caption("The map uses deterministic server-side positioning with no continuous physics.")
