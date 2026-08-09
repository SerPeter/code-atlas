// The map view — a vanilla-JS port of the v1.1 design's AtlasCanvas component and
// the map slice of its page logic, driven by /map/api instead of mock data.
//
// The canvas is DOM + SVG exactly as the design draws it: edges are SVG lines (or
// taper polygons), nodes are positioned spans shaped per kind, arrowheads are
// clip-path divs, labels are chips placed by the design's outward seat search.
// Every colour resolves through CSS custom properties, which is what lets the
// oklch() community ramp and color-mix() dimming work with no JS colour parser.
//
// Positions arrive with the payload (the layout runs server-side, same algorithm
// and constants as the design's, so the same graph looks the same on every load).

(function () {
  "use strict";

  var S = 1000;
  var EW = { structural: 1, resolved: 0.72, guessed: 0.45, unknown: 0.22 };

  // Twelve node kinds, distinguished by silhouette; colour carries community at
  // every level, so shape answers "what is it" and colour answers "where does it
  // belong" — externals get their own community, which is the boundary itself.
  var KIND_SHAPE = {
    module: "ring", package: "ring", class: "square",
    function: "circle", method: "circle",
    constant: "diamond", env_var: "diamond",
    doc_file: "tab", doc_section: "tab", knowledge_note: "tab",
    external_package: "outside", external_symbol: "outside", external: "outside",
    test: "dashed", noncode: "solid",
  };

  function SHAPES(n, c, r) {
    var shape = KIND_SHAPE[n.kind] || "circle";
    var base = { w: (r * 2).toFixed(1), h: (r * 2).toFixed(1), radius: "50%", rot: "0deg" };
    switch (shape) {
      case "ring":
        return Object.assign(base, { fill: "var(--color-bg)", border: Math.max(2.5, r * 0.42).toFixed(1) + "px solid " + c });
      case "square":
        return Object.assign(base, { radius: "2px", fill: c, border: "1px solid var(--color-bg)" });
      case "diamond":
        return Object.assign(base, { w: (r * 1.7).toFixed(1), h: (r * 1.7).toFixed(1), radius: "1px", rot: "45deg", fill: c, border: "1px solid var(--color-bg)" });
      case "tab":
        return Object.assign(base, { w: (r * 2.4).toFixed(1), h: (r * 1.5).toFixed(1), radius: "2px 2px 2px 0", fill: c, border: "1px solid var(--color-bg)" });
      case "outside":
        return Object.assign(base, { fill: "transparent", border: "1.5px dashed " + c });
      case "dashed":
        return Object.assign(base, { fill: "var(--color-bg)", border: "1.5px dashed " + c });
      case "solid":
        return Object.assign(base, { fill: "var(--color-neutral-300)", border: "1.5px solid " + c });
      default:
        return Object.assign(base, { fill: c, border: "1px solid var(--color-bg)" });
    }
  }

  // The rail's swatch shapes for the kind list — mirrors the canvas's shape map.
  var SHAPE_SWATCH = {
    circle: { w: 11, h: 11, radius: "50%", rot: "0deg", fill: "var(--color-text)", border: "0" },
    square: { w: 11, h: 11, radius: "2px", rot: "0deg", fill: "var(--color-text)", border: "0" },
    diamond: { w: 9, h: 9, radius: "1px", rot: "45deg", fill: "var(--color-text)", border: "0" },
    rect: { w: 13, h: 9, radius: "2px 2px 2px 0", rot: "0deg", fill: "var(--color-text)", border: "0" },
    ring: { w: 12, h: 12, radius: "50%", rot: "0deg", fill: "transparent", border: "3px solid var(--color-text)" },
    hollow: { w: 11, h: 11, radius: "50%", rot: "0deg", fill: "transparent", border: "1.5px dashed var(--color-text)" },
  };

  var CHIP = {
    structural: { bg: "var(--color-neutral-300)", fg: "var(--color-neutral-900)", bd: "1px solid transparent", t: "STRUCTURAL" },
    resolved: { bg: "var(--color-neutral-200)", fg: "var(--color-neutral-800)", bd: "1px solid transparent", t: "RESOLVED" },
    guessed: { bg: "var(--color-accent-100)", fg: "var(--color-accent-800)", bd: "1px solid var(--color-accent)", t: "GUESSED" },
    unknown: { bg: "transparent", fg: "color-mix(in srgb, var(--color-text) 50%, transparent)", bd: "1px dashed var(--color-divider)", t: "UNKNOWN" },
  };

  var MUTED10 = "color-mix(in srgb, var(--color-text) 50%, transparent)";
  var LABEL10 = 'font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:' + MUTED10;

  function pick(fromUrl, stored, allowed, fallback) {
    if (allowed.indexOf(fromUrl) >= 0) return fromUrl;
    if (allowed.indexOf(stored) >= 0) return stored;
    return fallback;
  }

  // ── State ────────────────────────────────────────────────────────────────
  var qs = new URLSearchParams(window.location.search);
  var state = {
    level: qs.get("level") === "entity" ? "entity" : "module",
    scope: qs.get("module") || "",
    // null until the payload names the applied set — the defaults live server-side,
    // and duplicating them here would let the two drift.
    hidden: qs.has("hide") ? new Set(qs.get("hide").split(",").filter(Boolean)) : null,
    expandMethods: qs.get("expand") === "1",
    showTests: qs.has("show_tests") ? qs.get("show_tests") === "1" : localStorage.getItem("atlas.showTests") === "1",
    showNoncode: qs.has("show_noncode") ? qs.get("show_noncode") === "1" : localStorage.getItem("atlas.showNoncode") === "1",
    showExternal: qs.has("show_external") ? qs.get("show_external") === "1" : localStorage.getItem("atlas.showExternal") === "1",
    focus: qs.has("focus") ? parseInt(qs.get("focus"), 10) : -1,
    selected: qs.get("selected") || null,
    // URL wins (a shared link shows what its sender saw); the stored default fills
    // in otherwise. The rail and the Settings page both edit the same stored value.
    labels: pick(qs.get("labels"), localStorage.getItem("atlas.labels"), ["few", "some", "all"], "some"),
    hops: parseInt(pick(qs.get("hops"), localStorage.getItem("atlas.hops"), ["1", "2", "3"], "1"), 10),
    direction: pick(qs.get("direction"), localStorage.getItem("atlas.direction"), ["arrows", "taper", "fade", "flow"], "arrows"),
    hover: null,
    // Transient rail-hover highlights: a kind id, or a community id.
    kindHover: null,
    commHover: -1,
    // Directories the scope tree shows open. Null until first render, which opens
    // the path to the current scope.
    scopeOpen: null,
    legend: false,
    // Display settings start collapsed: what is on the map matters more than how it is drawn.
    open: [],
    arming: false,
    pathFrom: null,
    pathTo: null,
    k: 1, tx: 0, ty: 0,
  };

  var D = null; // the current payload
  var cache = {};
  var els = {
    aside: document.getElementById("map-aside"),
    main: document.getElementById("map-main"),
    ctx: document.getElementById("map-context"),
  };
  if (!els.aside || !els.main || !els.ctx) return;

  function esc(s) {
    return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function fmt(n) {
    return (n || 0).toLocaleString("en-US");
  }

  function payloadUrl() {
    var p = new URLSearchParams();
    if (state.level === "entity") {
      p.set("level", "entity");
      if (state.scope) p.set("module", state.scope);
      if (state.expandMethods) p.set("expand", "1");
      if (state.hidden !== null) p.set("hide", Array.from(state.hidden).sort().join(","));
    }
    if (state.showTests) p.set("show_tests", "1");
    if (state.showNoncode) p.set("show_noncode", "1");
    if (state.showExternal && state.level !== "entity") p.set("show_external", "1");
    return "/map/api?" + p.toString();
  }

  function syncUrl() {
    var p = new URLSearchParams();
    // Theme overrides ride along — they are view state, not map state, but dropping
    // them here would flip a shared dark link back to light on the first click.
    var current = new URLSearchParams(window.location.search);
    if (current.get("theme")) p.set("theme", current.get("theme"));
    if (current.get("palette")) p.set("palette", current.get("palette"));
    if (state.level === "entity") {
      p.set("level", "entity");
      if (state.scope) p.set("module", state.scope);
      if (state.expandMethods) p.set("expand", "1");
      if (state.hidden !== null) p.set("hide", Array.from(state.hidden).sort().join(","));
    }
    if (state.showTests) p.set("show_tests", "1");
    if (state.showNoncode) p.set("show_noncode", "1");
    if (state.showExternal && state.level !== "entity") p.set("show_external", "1");
    if (state.direction !== "arrows") p.set("direction", state.direction);
    if (state.hops !== 1) p.set("hops", String(state.hops));
    if (state.labels !== "some") p.set("labels", state.labels);
    if (state.focus >= 0) p.set("focus", String(state.focus));
    if (state.selected) p.set("selected", state.selected);
    var q = p.toString();
    window.history.replaceState(null, "", q ? "?" + q : window.location.pathname);
  }

  function load() {
    // A static export embeds the module-level payload it was written with; anything
    // beyond it (filters, the entity level) needs the live server and says so.
    if (window.ATLAS_EMBED && state.level === "module" && !state.showTests && !state.showNoncode) {
      D = window.ATLAS_EMBED;
      render();
      return;
    }
    var url = payloadUrl();
    if (cache[url]) {
      D = cache[url];
      render();
      return;
    }
    // The full-project layout takes tens of seconds the first time (cached after);
    // an empty canvas for that long reads as broken, so say what is happening.
    if (state.level === "entity" && !state.scope) {
      canvasReady = false;
      els.main.innerHTML =
        '<div style="position:absolute;left:50%;top:44%;transform:translate(-50%,-50%);width:min(420px,80%);background:var(--color-bg);border:1px solid var(--color-divider);border-radius:12px;padding:14px 16px;box-shadow:var(--shadow-md)">' +
        '<div style="' + LABEL10 + '">Computing the layout</div>' +
        '<div style="margin-top:6px;font-size:13px;line-height:1.55">The whole project at once — position is computed from every edge, which takes tens of seconds on the first load and is cached afterwards.</div>' +
        "</div>";
    }
    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        cache[url] = data;
        D = data;
        // The server names the hidden set it actually applied (its defaults, when
        // the request stayed silent); the rail's toggles render from that truth.
        if (data.level === "entity" && state.hidden === null) {
          state.hidden = new Set(data.hidden_kinds || []);
        }
        render();
      })
      .catch(function () {
        D = {
          unavailable: window.ATLAS_EMBED
            ? "This export carries the module map it was written with. Filters and the entity level need the live server."
            : "The map data could not be fetched — is the server still running?",
          nodes: [], edges: [], communities: [], kinds: [],
        };
        render();
      });
  }

  function set(patch) {
    Object.assign(state, patch);
    syncUrl();
  }

  // A state change that alters *what is drawn* refetches; one that alters *how it
  // is drawn* only re-renders. The design's own sidebar makes the same split.
  function setAndLoad(patch) {
    set(patch);
    load();
  }
  function setAndRender(patch) {
    set(patch);
    render();
  }

  function byId() {
    var out = {};
    (D.nodes || []).forEach(function (n) { out[n.id] = n; });
    return out;
  }

  function communityById() {
    var out = {};
    (D.communities || []).forEach(function (c) { out[c.id] = c; });
    return out;
  }

  function nodeColor(n, comms) {
    var c = comms[n.community];
    return c ? c.color : "var(--atlas-c8)";
  }

  // ── Path search (port of the design's shortestPath) ──────────────────────
  function shortestPath(a, b) {
    if (!a || !b || a === b) return null;
    var adj = new Map();
    (D.edges || []).forEach(function (e) {
      // Containment edges would make every pair two hops apart via the module —
      // a "path" that answers nothing. Only dependency edges are walkable.
      if (e.rel === "defines") return;
      if (!adj.has(e.s)) adj.set(e.s, []);
      if (!adj.has(e.t)) adj.set(e.t, []);
      adj.get(e.s).push({ to: e.t, ev: e.ev, dir: "out" });
      adj.get(e.t).push({ to: e.s, ev: e.ev, dir: "in" });
    });
    var prev = new Map([[a, null]]);
    var q = [a];
    while (q.length) {
      var cur = q.shift();
      if (cur === b) break;
      (adj.get(cur) || []).forEach(function (nx) {
        if (prev.has(nx.to)) return;
        prev.set(nx.to, { from: cur, ev: nx.ev, dir: nx.dir });
        q.push(nx.to);
      });
    }
    if (!prev.has(b)) return null;
    var hops = [];
    var at = b;
    while (prev.get(at)) {
      var p = prev.get(at);
      hops.unshift({ from: p.from, to: at, ev: p.ev, dir: p.dir });
      at = p.from;
    }
    return hops;
  }

  function pathSets() {
    if (!(state.pathFrom && state.pathTo)) return { nodes: new Set(), edges: new Set(), hops: null };
    var hops = shortestPath(state.pathFrom, state.pathTo);
    if (!hops) return { nodes: new Set(), edges: new Set(), hops: null };
    var nodes = new Set([state.pathFrom]);
    var edges = new Set();
    hops.forEach(function (h) {
      nodes.add(h.to);
      edges.add(h.from + "|" + h.to);
    });
    return { nodes: nodes, edges: edges, hops: hops };
  }

  // ── Render ───────────────────────────────────────────────────────────────
  function render() {
    renderSidebar();
    renderMain();
    renderContext();
  }

  // The scope tree: every indexed module arranged by directory, any level
  // selectable — a directory scopes to everything beneath it, since the server's
  // module summary is prefix-based.
  function buildScopeTree() {
    var root = { dirs: {}, files: [], count: 0 };
    (D.scope_options || []).forEach(function (o) {
      var parts = String(o.id).replace(/\\/g, "/").split("/");
      var node = root;
      root.count += o.entities || 0;
      for (var i = 0; i < parts.length - 1; i++) {
        var dir = parts[i];
        if (!node.dirs[dir]) node.dirs[dir] = { dirs: {}, files: [], count: 0 };
        node = node.dirs[dir];
        node.count += o.entities || 0;
      }
      node.files.push({ id: o.id, name: parts[parts.length - 1], count: o.entities || 0 });
    });
    return root;
  }

  function scopeRow(id, name, count, depth, expandable, open) {
    var on = state.scope === id;
    return (
      '<div style="display:flex;align-items:center">' +
      (expandable
        ? '<button data-scope-exp="' + esc(id) + '" style="flex:none;width:18px;background:transparent;border:0;cursor:pointer;color:color-mix(in srgb, var(--color-text) 45%, transparent);font:inherit;font-size:10px;padding:0 0 0 ' + (4 + depth * 12) + 'px">' + (open ? "▾" : "▸") + "</button>"
        : '<span style="flex:none;width:18px;padding-left:' + (4 + depth * 12) + 'px"></span>') +
      '<button data-scope="' + esc(id) + '" class="hv6" style="display:flex;flex:1;min-width:0;align-items:center;gap:7px;background:' +
      (on ? "color-mix(in srgb, var(--color-accent) 10%, transparent)" : "transparent") +
      ';border:0;border-radius:6px;padding:3px 7px;cursor:pointer;color:' + (on ? "var(--color-accent-700)" : "inherit") +
      ';font:inherit;font-size:12px;text-align:left">' +
      '<span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(name) + "</span>" +
      '<span style="flex:none;font-size:10.5px;font-variant-numeric:tabular-nums;color:color-mix(in srgb, var(--color-text) 45%, transparent)">' + fmt(count) + "</span>" +
      "</button></div>"
    );
  }

  function scopeTreeHtml() {
    if (state.scopeOpen === null) {
      // First render: open the ancestors of the current scope so it is visible.
      state.scopeOpen = new Set();
      var parts = state.scope.replace(/\\/g, "/").split("/");
      var at = "";
      for (var i = 0; i < parts.length - 1; i++) {
        at = at ? at + "/" + parts[i] : parts[i];
        state.scopeOpen.add(at);
      }
    }
    var tree = buildScopeTree();
    var out = [scopeRow("", "Whole project", tree.count, 0, false, false)];
    var walk = function (node, base, depth) {
      Object.keys(node.dirs).sort().forEach(function (dir) {
        var path = base ? base + "/" + dir : dir;
        var open = state.scopeOpen.has(path);
        out.push(scopeRow(path, dir + "/", node.dirs[dir].count, depth, true, open));
        if (open) walk(node.dirs[dir], path, depth + 1);
      });
      node.files.sort(function (a, b) { return a.name < b.name ? -1 : 1; }).forEach(function (f) {
        out.push(scopeRow(f.id, f.name, f.count, depth, false, false));
      });
    };
    walk(tree, "", 0);
    return out.join("");
  }

  function toggleHtml(on, extra) {
    return (
      '<span style="width:32px;height:18px;flex:none;border-radius:999px;background:' +
      (on ? "var(--color-accent)" : "transparent") + ";border:1px solid " +
      (on ? "var(--color-accent)" : "var(--color-divider)") +
      ';position:relative;display:block;' + (extra || "transition:background 120ms") + '">' +
      '<span style="position:absolute;top:2px;left:' + (on ? "16px" : "2px") +
      ';width:12px;height:12px;border-radius:999px;background:' +
      (on ? "var(--color-bg)" : "color-mix(in srgb, var(--color-text) 45%, transparent)") +
      ';display:block;' + (extra ? "" : "transition:left 120ms") + '"></span></span>'
    );
  }

  function sectionHead(id, label, value) {
    var on = state.open.indexOf(id) >= 0;
    return (
      '<section style="background:var(--color-surface)">' +
      '<button data-section="' + id + '" class="hv5" style="display:flex;align-items:center;gap:8px;width:100%;background:transparent;border:0;border-top:1px solid var(--color-divider);padding:9px 14px;cursor:pointer;color:inherit;font:inherit;text-align:left">' +
      '<span style="font-size:12.5px;font-weight:600">' + label + "</span>" +
      '<span style="margin-left:auto;font-size:11px;color:color-mix(in srgb, var(--color-text) 50%, transparent)">' + value + "</span>" +
      '<span style="font-size:10px;color:color-mix(in srgb, var(--color-text) 45%, transparent)">' + (on ? "▾" : "▸") + "</span>" +
      "</button></section>"
    );
  }

  function renderSidebar() {
    if (!D) return;
    var entity = state.level === "entity";
    // Every total below sums over the SAME community table the panel lists —
    // including the files communities — so the arithmetic on screen always closes:
    // the sum equals the indexed module count by construction.
    var comms = D.communities || [];
    var moduleTotal = comms.reduce(function (a, c) { return a + c.count; }, 0);
    var drawnCount = {};
    (D.nodes || []).forEach(function (n) { drawnCount[n.community] = (drawnCount[n.community] || 0) + 1; });

    var html = "<div>";

    // ── Search, level, scope ──
    html +=
      '<div style="padding:13px 14px 12px;border-bottom:1px solid var(--color-divider)">' +
      '<input class="input" id="rail-search" placeholder="Search entities, symbols, docs…" style="height:34px;min-height:34px;font-size:13px">' +
      '<div style="margin-top:12px;' + LABEL10 + ';margin-bottom:7px">Level</div>' +
      '<div class="seg" style="width:100%;overflow:hidden">' +
      '<label class="seg-opt" style="flex:1;justify-content:center"><input type="radio" name="level" data-level="module"' + (entity ? "" : " checked") + "><span>Modules</span></label>" +
      '<label class="seg-opt" style="flex:1;justify-content:center"><input type="radio" name="level" data-level="entity"' + (entity ? " checked" : "") + "><span>Entities</span></label>" +
      "</div>";

    var mapSummary, levelNote;
    if (entity) {
      var drawnN = (D.nodes || []).length;
      var fullScope = !state.scope;
      var holder = fullScope ? "the project's" : "this module's";
      mapSummary = drawnN === D.in_module
        ? D.in_module + " entities · " + (D.edges || []).length + " edges"
        : "Drawing " + drawnN + " of " + holder + " " + fmt(D.in_module) + " entities · " + fmt((D.edges || []).length) + " edges";
      var methodTally = (D.tally || []).filter(function (t) { return t.id === "method"; })[0];
      var methods = methodTally ? methodTally.in_module : 0;
      var hiddenTotal = (D.tally || []).reduce(function (a, t) {
        return a + ((state.hidden && state.hidden.has(t.id)) ? t.in_module : 0);
      }, 0);
      levelNote =
        "Each node is an entity as the graph stores it — silhouette and colour both carry kind. " +
        (fullScope
          ? "The whole index at once: " + fmt(D.in_module) + " entities across " + D.module_total + " modules."
          : "This module holds " + D.in_module + " of the " + fmt(D.entity_total) + " entities indexed across " + D.module_total + " modules.") +
        (D.collapsed && methods
          ? " " + fmt(methods) + " methods are folded into the classes that hold them — a class's size is how many it holds, and calls into a method are drawn to its class."
          : "") +
        (hiddenTotal ? " " + fmt(hiddenTotal) + " entities are hidden by the kind filters below, counted on their rows." : "") +
        (D.truncated ? " Above the " + fmt(4000) + "-node cap the map truncates; it is drawing the " + fmt(drawnN) + " most connected." : "");
    } else {
      mapSummary = moduleTotal + " modules · " + D.edge_total + " edges · " + comms.length + " communities";
      levelNote =
        "Each node is a module — an aggregation. " + fmt(D.entity_total) + " indexed entities roll up into " +
        moduleTotal + " module nodes, and each edge stands for every entity-level dependency between two modules.";
    }
    html +=
      '<div style="margin-top:7px;font-size:11.5px;line-height:1.45;color:color-mix(in srgb, var(--color-text) 62%, transparent)">' + esc(mapSummary) + "</div>" +
      '<div style="margin-top:6px;font-size:11px;line-height:1.5;color:color-mix(in srgb, var(--color-text) 55%, transparent)">' + esc(levelNote) + "</div>";

    if (entity) {
      html +=
        '<div style="margin-top:10px">' +
        '<div style="' + LABEL10 + ';margin-bottom:5px">Scope</div>' +
        '<div style="max-height:230px;overflow-y:auto;border:1px solid var(--color-divider);border-radius:8px;background:var(--color-surface);padding:3px">' +
        scopeTreeHtml() +
        "</div></div>";
    }
    html += "</div>";

    // ── Filters — every view: the same toggles, every hidden thing counted ──
    {
      var testCount = D.test_count || 0;
      var ncCount = D.noncode_count || 0;
      var drawnNodes = entity ? D.in_module : (D.nodes || []).length;
      // "the most connected" only when something was actually cut — with nothing
      // truncated, what is drawn is everything the filters admit.
      var connectedNote = D.truncated ? " — the most connected." : ".";
      var extCount = entity
        ? (D.tally || []).reduce(function (a, t) {
            return a + (t.id === "external_package" || t.id === "external_symbol" ? t.in_module : 0);
          }, 0)
        : D.external_count || 0;
      var extShown = entity
        ? !(state.hidden && (state.hidden.has("external_package") || state.hidden.has("external_symbol")))
        : state.showExternal;
      var unitWord = entity ? "entities" : "modules";
      var unitTotal = entity ? D.in_module + testCount + ncCount : moduleTotal;
      var hiddenParts = [];
      if (!state.showTests && testCount) hiddenParts.push(fmt(testCount) + " test " + unitWord);
      if (!state.showNoncode && ncCount) hiddenParts.push(fmt(ncCount) + " non-code " + unitWord);
      if (!extShown && extCount) hiddenParts.push(fmt(extCount) + " external packages");
      var hiddenNote = "Drawing " + fmt(drawnNodes) + " of " + fmt(unitTotal) + " " + unitWord + connectedNote +
        (hiddenParts.length ? " " + hiddenParts.join(" and ") + " are hidden by the filters above." : "") +
        (entity && D.collapsed ? " Folded methods and filtered kinds are counted below." : "");
      html +=
        '<section style="padding:13px 14px 14px;border-bottom:1px solid var(--color-divider)">' +
        '<div style="' + LABEL10 + ';margin-bottom:8px">Filters</div>' +
        '<button data-toggle="tests" class="hv6" style="display:flex;align-items:center;gap:10px;width:100%;background:transparent;border:0;border-radius:8px;padding:6px 4px;cursor:pointer;color:inherit;font:inherit;font-size:13px;text-align:left">' +
        toggleHtml(state.showTests) +
        '<span style="flex:1">Test modules</span>' +
        '<span style="font-size:11px;font-variant-numeric:tabular-nums;color:color-mix(in srgb, var(--color-text) 55%, transparent)">' + testCount + (state.showTests ? " shown" : " hidden") + "</span>" +
        "</button>" +
        '<button data-toggle="noncode" class="hv6" style="display:flex;align-items:center;gap:10px;width:100%;background:transparent;border:0;border-radius:8px;padding:6px 4px;cursor:pointer;color:inherit;font:inherit;font-size:13px;text-align:left">' +
        toggleHtml(state.showNoncode) +
        '<span style="flex:1">Non-code files</span>' +
        '<span style="font-size:11px;font-variant-numeric:tabular-nums;color:color-mix(in srgb, var(--color-text) 55%, transparent)">' + ncCount + (state.showNoncode ? " shown" : " hidden") + "</span>" +
        "</button>" +
        '<button data-toggle="external" class="hv6" style="display:flex;align-items:center;gap:10px;width:100%;background:transparent;border:0;border-radius:8px;padding:6px 4px;cursor:pointer;color:inherit;font:inherit;font-size:13px;text-align:left">' +
        toggleHtml(extShown) +
        '<span style="flex:1">External packages</span>' +
        '<span style="font-size:11px;font-variant-numeric:tabular-nums;color:color-mix(in srgb, var(--color-text) 55%, transparent)">' + fmt(extCount) + (extShown ? " shown" : " hidden") + "</span>" +
        "</button>" +
        '<div style="margin-top:8px;padding:8px 10px;border-radius:8px;background:var(--color-surface);font-size:11px;line-height:1.45;color:color-mix(in srgb, var(--color-text) 65%, transparent)">' + esc(hiddenNote) + "</div>" +
        "</section>";
    }

    // ── Node kinds (entity level only) ──
    if (entity) {
      var tally = D.tally || [];
      var methodRow = tally.filter(function (t) { return t.id === "method"; })[0];
      var methodCount = methodRow ? methodRow.in_module : 0;
      html +=
        '<section style="padding:13px 14px 14px;border-bottom:1px solid var(--color-divider)">' +
        '<div style="display:flex;align-items:baseline;margin-bottom:8px">' +
        '<div style="' + LABEL10 + '">Node kinds</div>' +
        '<div style="margin-left:auto;padding-left:10px;white-space:nowrap;flex:none;font-size:11px;color:color-mix(in srgb, var(--color-text) 45%, transparent)">' + tally.length + " of 12 kinds present</div>" +
        "</div>" +
        tally.map(function (t) {
          var kind = (D.kinds || []).filter(function (k) { return k.id === t.id; })[0] || { label: t.id, shape: "circle" };
          var sw = SHAPE_SWATCH[kind.shape] || SHAPE_SWATCH.circle;
          var fill = sw.fill;
          var border = sw.border;
          // The skeleton kinds are not toggleable: containers anchor everything
          // and classes are the fold target. The Method row toggles the fold
          // itself — one control, not a switch plus a row describing it.
          var isMethod = t.id === "method";
          var toggleable = isMethod || ["module", "package", "class"].indexOf(t.id) < 0;
          var isHidden = !isMethod && state.hidden && state.hidden.has(t.id);
          var drawnNote = isMethod ? (D.collapsed ? "folded" : "")
            : isHidden ? "hidden"
            : t.drawn === t.in_module ? ""
            : t.drawn ? t.drawn + " drawn" : "not drawn";
          var swatch =
            '<span style="width:14px;display:flex;justify-content:center;flex:none">' +
            '<span style="width:' + sw.w + "px;height:" + sw.h + "px;border-radius:" + sw.radius + ";rotate:" + sw.rot + ";background:" + fill + ";border:" + border + ';display:block"></span>' +
            "</span>" +
            '<span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(kind.label) + "</span>" +
            '<span style="font-size:10.5px;color:color-mix(in srgb, var(--color-text) 45%, transparent)">' + drawnNote + "</span>" +
            '<span style="font-variant-numeric:tabular-nums;color:color-mix(in srgb, var(--color-text) 55%, transparent)">' + fmt(t.in_module) + "</span>";
          if (!toggleable) {
            return '<div data-kind-row="' + t.id + '" style="display:flex;align-items:center;gap:10px;padding:4px 0;font-size:12px;opacity:' + (t.drawn ? 1 : 0.5) + '">' + swatch + "</div>";
          }
          var title = isMethod
            ? (D.collapsed ? "Expand methods out of their classes" : "Fold methods into their classes")
            : (isHidden ? "Show " : "Hide ") + esc(kind.label.toLowerCase()) + "s";
          var faded = isHidden || (isMethod && D.collapsed);
          return (
            '<button data-kind-toggle="' + t.id + '" data-kind-row="' + t.id + '" class="hv6" title="' + title + '" style="display:flex;align-items:center;gap:10px;width:100%;background:transparent;border:0;border-radius:8px;padding:4px 0;cursor:pointer;color:inherit;font:inherit;font-size:12px;text-align:left;opacity:' + (faded ? 0.5 : 1) + '">' +
            swatch + "</button>"
          );
        }).join("") +
        '<div style="margin-top:8px;font-size:11px;line-height:1.5;color:color-mix(in srgb, var(--color-text) 62%, transparent)">Silhouette carries kind; colour carries the community of the module that holds the entity. Dashed means it sits outside the index.</div>' +
        "</section>";
    }

    // ── Communities — wherever the payload names them ──
    if ((D.communities || []).length) {
      var CAP = 10, rowH = 28.6;
      var commRows = (D.communities || []).map(function (c) {
        var drawn = drawnCount[c.id] || 0;
        return (
          '<button data-comm="' + c.id + '" class="hv6" style="display:flex;align-items:center;gap:9px;width:100%;background:' +
          (state.focus === c.id ? "color-mix(in srgb, var(--color-accent) 10%, transparent)" : "transparent") +
          ";opacity:" + (drawn ? 1 : 0.45) +
          ';border:0;border-radius:8px;padding:5px 8px;cursor:pointer;color:inherit;font:inherit;font-size:12px;text-align:left">' +
          '<span style="width:9px;height:9px;flex:none;border-radius:999px;background:' + c.color + ';display:block"></span>' +
          '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;direction:rtl;text-align:left">' + esc(c.name) + "</span>" +
          '<span style="font-size:9.5px;letter-spacing:0.04em;white-space:nowrap;color:color-mix(in srgb, var(--color-text) 42%, transparent)">' +
          (c.files ? "files · " : "") + (drawn ? drawn + " drawn" : "not drawn") + "</span>" +
          '<span style="width:22px;text-align:right;font-variant-numeric:tabular-nums;color:color-mix(in srgb, var(--color-text) 55%, transparent)">' + c.count + "</span>" +
          "</button>"
        );
      });
      html +=
        '<section style="padding:13px 10px 14px;border-bottom:1px solid var(--color-divider)">' +
        '<div style="display:flex;align-items:baseline;padding:0 4px;margin-bottom:7px">' +
        '<div style="' + LABEL10 + '">Communities</div>' +
        '<div style="margin-left:auto;padding-left:10px;white-space:nowrap;flex:none;font-size:11px;color:color-mix(in srgb, var(--color-text) 45%, transparent)">' + comms.length + " · " + fmt(moduleTotal) + (entity ? " entities" : " modules") + "</div>" +
        "</div>" +
        '<div style="max-height:' + ((D.communities || []).length > CAP ? CAP * rowH + "px" : "none") + ';overflow-y:auto">' +
        commRows.join("") +
        "</div>" +
        ((D.communities || []).length > CAP
          ? '<div style="padding:6px 8px 0;font-size:11px;color:color-mix(in srgb, var(--color-text) 50%, transparent)">' + (D.communities || []).length + " listed, sized as in the index — scroll for the rest</div>"
          : "") +
        "</section>";
    }

    // ── Display settings ──
    var dirValue = { arrows: "Arrows", taper: "Taper", fade: "Fade", flow: "Flow" }[state.direction];
    var dirNote = {
      arrows: "A head just short of the target. Most literal, busiest at " + (entity ? (D.edges || []).length : D.edge_total) + " edges.",
      taper: "The line thins toward what it depends on. No extra marks; evidence reads from thickness alone.",
      fade: "The line brightens toward what it depends on. Quietest of the four.",
      flow: "Dashes travel toward the dependency; faster where the edge is better evidenced.",
    }[state.direction];
    var hopNote = state.hops === 1
      ? "Hovering a node isolates it, what it depends on, and what depends on it."
      : "What the node depends on travels " + state.hops + " hops downstream, each ring a step fainter. What depends on the node is always shown one hop only.";

    html +=
      '<div style="margin-top:2px;padding:11px 14px 6px;background:var(--color-surface);border-top:1px solid var(--color-divider)">' +
      '<div style="font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:color-mix(in srgb, var(--color-text) 45%, transparent)">Display settings</div>' +
      '<div style="margin-top:3px;font-size:11px;line-height:1.45;color:color-mix(in srgb, var(--color-text) 52%, transparent)">How the map is drawn. These change nothing about what is in it.</div>' +
      "</div>";

    html += sectionHead("dir", "Direction", dirValue);
    if (state.open.indexOf("dir") >= 0) {
      var dirOptions = [
        ["arrows", "Arrows", "polygon(0 42%, 62% 42%, 62% 20%, 100% 50%, 62% 80%, 62% 58%, 0 58%)"],
        ["taper", "Taper", "polygon(0 12%, 100% 44%, 100% 56%, 0 88%)"],
        ["fade", "Fade", "polygon(0 40%, 100% 40%, 100% 60%, 0 60%)"],
        ["flow", "Flow", "polygon(0 40%, 22% 40%, 22% 60%, 0 60%, 34% 40%, 56% 40%, 56% 60%, 34% 60%, 68% 40%, 90% 40%, 90% 60%, 68% 60%)"],
      ];
      html +=
        '<section style="padding:2px 14px 14px;background:var(--color-surface)">' +
        '<div style="display:flex;flex-wrap:wrap;gap:6px">' +
        dirOptions.map(function (d) {
          var on = state.direction === d[0];
          return (
            '<button data-dir="' + d[0] + '" class="hv7" style="display:flex;align-items:center;gap:7px;background:' +
            (on ? "color-mix(in srgb, var(--color-accent) 12%, transparent)" : "transparent") +
            ";color:" + (on ? "var(--color-accent-700)" : "var(--color-text)") +
            ";border:1px solid " + (on ? "var(--color-accent)" : "var(--color-divider)") +
            ';border-radius:999px;padding:4px 11px 4px 8px;cursor:pointer;font:inherit;font-size:11.5px">' +
            '<span style="width:26px;height:10px;display:block;background:' +
            (on ? "var(--color-accent)" : "color-mix(in srgb, var(--color-text) 60%, transparent)") +
            ";clip-path:" + d[2] + ";opacity:" + (on ? 1 : 0.7) + '"></span>' +
            "<span>" + d[1] + "</span></button>"
          );
        }).join("") +
        "</div>" +
        '<div style="margin-top:9px;font-size:11px;line-height:1.5;color:color-mix(in srgb, var(--color-text) 60%, transparent)">' + esc(dirNote) + "</div>" +
        "</section>";
    }

    html += sectionHead("hop", "Neighbourhood", state.hops === 1 ? "1 hop" : state.hops + " hops");
    if (state.open.indexOf("hop") >= 0) {
      html +=
        '<section style="padding:2px 14px 14px;background:var(--color-surface)">' +
        '<div class="seg" style="width:100%;overflow:hidden">' +
        [1, 2, 3].map(function (h) {
          return (
            '<label class="seg-opt" style="flex:1;justify-content:center"><input type="radio" name="hops" data-hops="' + h + '"' +
            (state.hops === h ? " checked" : "") + "><span>" + h + " hop" + (h > 1 ? "s" : "") + "</span></label>"
          );
        }).join("") +
        "</div>" +
        '<div style="margin-top:9px;font-size:11px;line-height:1.5;color:color-mix(in srgb, var(--color-text) 60%, transparent)">' + esc(hopNote) + "</div>" +
        "</section>";
    }

    html += sectionHead("lbl", "Labels", { few: "Hubs", some: "Most", all: "All" }[state.labels]);
    if (state.open.indexOf("lbl") >= 0) {
      html +=
        '<section style="padding:2px 14px 18px;background:var(--color-surface)">' +
        '<div class="seg" style="width:100%;overflow:hidden">' +
        [["few", "Hubs"], ["some", "Most"], ["all", "All"]].map(function (l) {
          return (
            '<label class="seg-opt" style="flex:1;justify-content:center"><input type="radio" name="labels" data-labels="' + l[0] + '"' +
            (state.labels === l[0] ? " checked" : "") + "><span>" + l[1] + "</span></label>"
          );
        }).join("") +
        "</div>" +
        '<div style="margin-top:12px;font-size:11px;line-height:1.5;color:color-mix(in srgb, var(--color-text) 60%, transparent)">' +
        "Size = dependency degree. Colour = community. Position is computed from the edges — coupled modules sit near each other. Scroll to zoom, drag to pan." +
        "</div></section>";
    }

    html += "</div>";
    els.aside.innerHTML = html;
  }

  // ── Canvas ───────────────────────────────────────────────────────────────
  var canvasReady = false;

  function ensureCanvas() {
    if (canvasReady) return;
    els.main.innerHTML =
      '<div style="position:absolute;inset:0">' +
      '<div data-atlas-canvas style="position:absolute;inset:0;background:var(--color-bg);overflow:hidden">' +
      '<div data-atlas-inner style="position:absolute;inset:38px 34px">' +
      '<div data-atlas-layer style="position:absolute;inset:0;transform:none;transform-origin:0 0"></div>' +
      "</div>" +
      '<div style="position:absolute;left:12px;top:12px;z-index:25;display:flex;align-items:center;gap:2px;background:var(--color-bg);border:1px solid var(--color-divider);border-radius:10px;padding:3px">' +
      '<button data-zoom="out" class="hv8" style="width:26px;height:24px;background:transparent;border:0;border-radius:7px;cursor:pointer;color:inherit;font:inherit;font-size:14px;line-height:1">−</button>' +
      '<button data-zoom="reset" class="hv8" style="min-width:46px;height:24px;background:transparent;border:0;border-radius:7px;cursor:pointer;color:inherit;font:inherit;font-size:11px;font-variant-numeric:tabular-nums">100%</button>' +
      '<button data-zoom="in" class="hv8" style="width:26px;height:24px;background:transparent;border:0;border-radius:7px;cursor:pointer;color:inherit;font:inherit;font-size:14px;line-height:1">+</button>' +
      '<span style="width:1px;height:16px;background:var(--color-divider);display:block;margin:0 2px"></span>' +
      '<button data-zoom="reset2" title="Reset zoom and position" class="hv8" style="display:flex;align-items:center;justify-content:center;width:26px;height:24px;background:transparent;border:0;border-radius:7px;cursor:default;opacity:0.35;color:inherit;font:inherit;font-size:13px;line-height:1">⤾</button>' +
      "</div>" +
      "</div>" +
      '<div data-overlays style="position:absolute;inset:0;pointer-events:none"></div>' +
      "</div>";
    canvasReady = true;
    wireCanvas();
  }

  function canvasEl() { return els.main.querySelector("[data-atlas-canvas]"); }
  function innerEl() { return els.main.querySelector("[data-atlas-inner]"); }
  function layerEl() { return els.main.querySelector("[data-atlas-layer]"); }

  function applyTransform() {
    var layer = layerEl();
    if (!layer) return;
    layer.style.transform = (state.k === 1 && state.tx === 0 && state.ty === 0)
      ? "none"
      : "translate(" + state.tx.toFixed(1) + "px," + state.ty.toFixed(1) + "px) scale(" + state.k.toFixed(3) + ")";
    var label = els.main.querySelector('[data-zoom="reset"]');
    if (label) label.textContent = Math.round(state.k * 100) + "%";
    var reset = els.main.querySelector('[data-zoom="reset2"]');
    if (reset) {
      var atDefault = state.k === 1 && state.tx === 0 && state.ty === 0;
      reset.style.cursor = atDefault ? "default" : "pointer";
      reset.style.opacity = atDefault ? "0.35" : "1";
      reset.disabled = atDefault;
    }
  }

  function zoomAt(px, py, f) {
    var k2 = Math.max(0.4, Math.min(8, state.k * f));
    if (k2 === state.k) return;
    var sc = k2 / state.k;
    state.k = k2;
    state.tx = px - (px - state.tx) * sc;
    state.ty = py - (py - state.ty) * sc;
    applyTransform();
  }

  function half() {
    var el = canvasEl();
    var r = el ? el.getBoundingClientRect() : { width: 600, height: 400 };
    return { x: r.width / 2, y: r.height / 2 };
  }

  var mainWired = false;

  function wireCanvas() {
    var el = canvasEl();
    if (!el) return;
    el.addEventListener("wheel", function (e) {
      e.preventDefault();
      var r = el.getBoundingClientRect();
      zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.15 : 1 / 1.15);
    }, { passive: false });
    var drag = null;
    el.addEventListener("pointerdown", function (e) {
      // No pointer capture here: capturing on pointerdown retargets the
      // compatibility click to this element and would stop nodes being selected.
      drag = { x: e.clientX, y: e.clientY, moved: 0, captured: false };
    });
    el.addEventListener("pointermove", function (e) {
      if (!drag) return;
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.moved += Math.abs(dx) + Math.abs(dy);
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.moved > 4 && !drag.captured) {
        drag.captured = true;
        el.style.cursor = "grabbing";
        try { el.setPointerCapture(e.pointerId); } catch (err) { /* no-op */ }
      }
      if (!drag.captured) return;
      state.tx += dx;
      state.ty += dy;
      applyTransform();
    });
    var end = function (e) {
      if (drag && drag.captured) suppressClick = true;
      drag = null;
      el.style.cursor = "grab";
      try { el.releasePointerCapture(e.pointerId); } catch (err) { /* no-op */ }
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    el.style.cursor = "grab";

    // The remaining listeners live on the container, which is never rebuilt —
    // registering them per canvas stacked duplicates that cancelled each other.
    if (mainWired) return;
    mainWired = true;

    els.main.addEventListener("click", function (e) {
      hideMenu();
      if (suppressClick) {
        // The click that ends a pan is not a selection gesture.
        suppressClick = false;
        return;
      }
      var z = e.target.closest("[data-zoom]");
      if (z) {
        var mode = z.getAttribute("data-zoom");
        if (mode === "in") zoomAt(half().x, half().y, 1.25);
        else if (mode === "out") zoomAt(half().x, half().y, 1 / 1.25);
        else { state.k = 1; state.tx = 0; state.ty = 0; applyTransform(); }
        return;
      }
      var node = e.target.closest("[data-node]");
      if (node) {
        onSelect(node.getAttribute("data-node"));
        return;
      }
      // Clicking empty canvas clears: first an armed path pick, then the selection.
      if (e.target.closest("[data-atlas-canvas]")) {
        if (state.arming) {
          set({ arming: false, pathFrom: null, pathTo: null });
        } else if (state.selected || state.pathTo) {
          set({ selected: null, pathFrom: null, pathTo: null });
        } else {
          return;
        }
        renderCanvasContents();
        renderOverlays();
        renderContext();
      }
    });

    // Right-click on a node opens its actions where the cursor is.
    els.main.addEventListener("contextmenu", function (e) {
      var node = e.target.closest("[data-node]");
      if (!node) {
        hideMenu();
        return;
      }
      e.preventDefault();
      showMenu(node.getAttribute("data-node"), e);
    });
    // Hover is whatever the pointer is over RIGHT NOW: entering a node sets it,
    // moving over empty canvas clears it. Tracking enter/leave pairs instead
    // loses the leave event whenever the hover re-render replaces the element
    // under the cursor, and the isolation sticks until the next click.
    els.main.addEventListener("mouseover", function (e) {
      if (e.target.closest("[data-ctx-menu]")) return;
      var node = e.target.closest("[data-node]");
      var id = node ? node.getAttribute("data-node") : null;
      if (state.hover !== id) {
        state.hover = id;
        restyleHover();
      }
    });
    // Leaving the frame entirely — including out of the window — always clears.
    els.main.addEventListener("mouseleave", function () {
      if (state.hover !== null) {
        state.hover = null;
        restyleHover();
      }
    });
  }

  var suppressClick = false;

  // ── Node context menu ────────────────────────────────────────────────────
  function hideMenu() {
    var menu = els.main.querySelector("[data-ctx-menu]");
    if (menu) menu.remove();
  }

  function showMenu(id, event) {
    hideMenu();
    var n = byId()[id];
    if (!n) return;
    var rect = els.main.getBoundingClientRect();
    var row = function (action, label) {
      return (
        '<button data-menu-action="' + action + '" class="hv6" style="display:block;width:100%;background:transparent;border:0;border-radius:7px;padding:6px 11px;cursor:pointer;color:inherit;font:inherit;font-size:12.5px;text-align:left;white-space:nowrap">' +
        label + "</button>"
      );
    };
    var items = "";
    if (n.uid) items += row("detail", "Open detail");
    items += row("path", "Path to…");
    if (state.level !== "entity" && n.path) items += row("entities", "Entities");
    if (state.level !== "entity" && n.community >= 0) items += row("focus", "Focus community");
    if (state.selected === id) items += row("deselect", "Deselect");
    var menu = document.createElement("div");
    menu.setAttribute("data-ctx-menu", id);
    menu.style.cssText =
      "position:absolute;z-index:30;min-width:150px;background:var(--color-bg);border:1px solid var(--color-divider);border-radius:10px;box-shadow:var(--shadow-md);padding:4px;left:" +
      Math.min(event.clientX - rect.left, rect.width - 170) + "px;top:" +
      Math.min(event.clientY - rect.top, rect.height - 160) + "px";
    menu.innerHTML =
      '<div style="padding:5px 11px 3px;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:' + MUTED10 +
      ';overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:220px;direction:rtl;text-align:left">' + esc(n.label) + "</div>" + items;
    menu.addEventListener("click", function (e) {
      var action = e.target.closest("[data-menu-action]");
      if (!action) return;
      e.stopPropagation();
      hideMenu();
      var kind = action.getAttribute("data-menu-action");
      if (kind === "detail" && n.uid) {
        window.location.href = "/entity/" + encodeURIComponent(n.uid);
      } else if (kind === "path") {
        set({ selected: id, arming: true, pathFrom: id, pathTo: null });
        render();
      } else if (kind === "entities" && n.path) {
        setAndLoad({ level: "entity", scope: n.path, selected: null });
      } else if (kind === "focus") {
        setAndRender({ focus: state.focus === n.community ? -1 : n.community });
      } else if (kind === "deselect") {
        set({ selected: null, pathFrom: null, pathTo: null, arming: false });
        render();
      }
    });
    els.main.appendChild(menu);
  }

  function onSelect(id) {
    if (state.arming && state.pathFrom && state.pathFrom !== id) {
      set({ pathTo: id, arming: false, selected: id });
    } else {
      set({ selected: state.selected === id ? null : id, pathFrom: null, pathTo: null, arming: false });
    }
    renderCanvasContents();
    renderOverlays();
    renderContext();
  }

  function renderMain() {
    if (!D) return;
    if (D.unavailable) {
      canvasReady = false;
      els.main.innerHTML =
        '<div style="position:absolute;left:50%;top:44%;transform:translate(-50%,-50%);width:min(420px,80%);background:var(--color-bg);border:1px solid var(--color-divider);border-radius:12px;padding:14px 16px;box-shadow:var(--shadow-md)">' +
        '<div style="' + LABEL10 + '">Map unavailable</div>' +
        '<div style="margin-top:6px;font-size:13px;line-height:1.55">' + esc(D.unavailable) + "</div>" +
        "</div>";
      return;
    }
    ensureCanvas();
    renderCanvasContents();
    renderOverlays();
    applyTransform();
  }

  // The scene: everything a hover cannot change, plus references to the drawn
  // elements. Hover restyles the scene in place — at full scope a DOM rebuild per
  // pointer movement would cost hundreds of milliseconds; a style pass costs a few.
  var scene = null;
  var CHIP_BUDGET = 140;

  function buildStatics() {
    var layer = layerEl();
    var inner = innerEl();
    if (!layer || !inner) return null;

    var ns = D.nodes || [];
    var es = D.edges || [];
    var ids = byId();
    var comms = communityById();
    var pos = {};
    ns.forEach(function (n) { pos[n.id] = { x: n.x, y: n.y }; });

    var box = inner.getBoundingClientRect();
    var CW = box.width, CH = box.height;
    // Node size is relative to the space each node actually gets, so a dense
    // graph draws smaller marks instead of overlapping ones.
    var pitch = (CW > 0 && ns.length) ? Math.sqrt((CW * CH) / ns.length) : 40;
    var maxDeg = ns.reduce(function (m, n) { return Math.max(m, n.deg); }, 1);
    var rCap = Math.max(5, Math.min(14, pitch * 0.28));
    var radius = function (n) { return 2.6 + (rCap - 2.6) * Math.sqrt(n.deg / maxDeg); };
    var edgeColor = function (id) {
      return nodeColor(ids[id], comms);
    };
    var SX = CW > 0 ? CW / S : 1, SY = CH > 0 ? CH / S : 1;

    var E = es.map(function (e, i) {
      var a = pos[e.s], b = pos[e.t];
      if (!a || !b) return null;
      var ev = e.ev || "unknown";
      var cross = ids[e.s] && ids[e.t] && ids[e.s].community !== ids[e.t].community;
      return {
        i: i, s: e.s, t: e.t, ev: ev, defines: e.rel === "defines",
        weight: EW[ev] * (0.7 + e.w * 0.1),
        cross: cross,
        baseColor: cross ? "var(--color-text)" : edgeColor(e.s),
        baseDash: ev === "guessed" ? "5 4" : ev === "unknown" ? "1.5 5" : "none",
        x1: a.x * SX, y1: a.y * SY, x2: b.x * SX, y2: b.y * SY,
        rt: ids[e.t] ? radius(ids[e.t]) : 4, rs: ids[e.s] ? radius(ids[e.s]) : 4,
      };
    }).filter(Boolean);

    // Ranked, not thresholded: degrees are quantized, so a degree cut-off can
    // admit everything at one level and nothing at another.
    var ranked = ns.slice().sort(function (a, b) { return b.deg - a.deg || a.id.localeCompare(b.id); });
    var share = state.labels === "all" ? 1 : state.labels === "some" ? 0.6 : 0.25;
    var eligible = new Set(ranked.slice(0, Math.max(3, Math.round(ranked.length * share))).map(function (n) { return n.id; }));

    // Node discs in a coarse grid, so a label seat checks a handful of nearby
    // discs instead of every node on the map.
    var discs = CW > 0 ? ns.map(function (n) {
      var r = radius(n);
      var cx = pos[n.id].x / S * CW, cy = pos[n.id].y / S * CH;
      return { id: n.id, l: cx - r, t: cy - r, r: cx + r, b: cy + r };
    }) : [];
    var CELL = 64;
    var discGrid = new Map();
    discs.forEach(function (d, idx) {
      for (var gx = Math.floor(d.l / CELL); gx <= Math.floor(d.r / CELL); gx++) {
        for (var gy = Math.floor(d.t / CELL); gy <= Math.floor(d.b / CELL); gy++) {
          var key = gx + "," + gy;
          var bucket = discGrid.get(key);
          if (!bucket) discGrid.set(key, (bucket = []));
          bucket.push(idx);
        }
      }
    });

    return {
      layer: layer, ns: ns, es: es, ids: ids, comms: comms, pos: pos,
      CW: CW, CH: CH, radius: radius, eligible: eligible,
      discs: discs, discGrid: discGrid, CELL: CELL,
      // Past ~1,200 edges the resting field attenuates so structure reads as
      // texture; anything highlighted (hover, selection, path) stays at full
      // strength — the isolation is the answer, the field is the context.
      density: Math.min(1, Math.sqrt(1200 / Math.max(1, E.length))),
      tight: CW < 420, mode: state.direction, E: E,
      sel: state.selected, focus: state.focus,
      edgeEls: null, gradEls: null, arrowEls: null, nodeEls: null, ringEls: null,
      labelsHost: null, leadersHost: null,
    };
  }

  // Everything the hover CAN change: the isolation sets and the (possibly
  // previewed) path. Recomputed per pointer move, applied by restyleHover.
  function hoverCtx(st) {
    var sel = st.sel;
    var active = state.hover || sel;

    // Downstream travels: what this node depends on, and what those depend on in
    // turn, expands to the full hop setting. What depends on the node is shown one
    // hop only — following it further answers a different question. Containment
    // edges are excluded: the module defines everything, so walking them would
    // light the whole scope and the isolation would show nothing.
    var nbr = new Set();
    var dist = new Map();
    var depth = Math.max(1, Math.min(4, state.hops));
    if (active) {
      nbr.add(active); dist.set(active, 0);
      var deps = st.es.filter(function (e) { return e.rel !== "defines"; });
      var frontier = [active];
      for (var d = 0; d < depth; d++) {
        var next = [];
        deps.forEach(function (e) {
          if (frontier.indexOf(e.s) >= 0 && !nbr.has(e.t)) { nbr.add(e.t); dist.set(e.t, d + 1); next.push(e.t); }
        });
        frontier = next;
        if (!frontier.length) break;
      }
      deps.forEach(function (e) {
        if (e.t === active && !nbr.has(e.s)) { nbr.add(e.s); dist.set(e.s, 1); }
      });
    }
    // Each hop out from the hovered node reads a step fainter, so the ring a
    // module sits in is legible without losing it entirely.
    var HOP_FADE = [1, 1, 0.55, 0.32, 0.2];
    var hopOpacity = function (id) {
      var dd = dist.has(id) ? dist.get(id) : 99;
      return HOP_FADE[Math.min(4, dd)] != null && dd <= 4 ? HOP_FADE[Math.min(4, dd)] : 0.13;
    };

    var path = pathSets();
    // With a selection (or an armed "Path to…"), hovering another node previews
    // the route between them before any click commits it.
    var previewFrom = state.pathFrom || sel;
    if (!path.hops && previewFrom && state.hover && state.hover !== previewFrom) {
      var previewHops = shortestPath(previewFrom, state.hover);
      if (previewHops) {
        var pn = new Set([previewFrom]);
        var pe = new Set();
        previewHops.forEach(function (h) { pn.add(h.to); pe.add(h.from + "|" + h.to); });
        path = { nodes: pn, edges: pe, hops: previewHops };
      }
    }

    return {
      sel: sel, active: active, nbr: nbr, hopOpacity: hopOpacity,
      // A selection keeps its neighbourhood isolated — the highlight should not
      // vanish the moment the pointer moves on. A hover retargets the isolation.
      dimming: !!active,
      path: path, pathing: path.nodes.size > 0,
    };
  }

  // Containment is scaffolding, not signal: a fixed hairline at low opacity,
  // whatever the direction mode, so a dense scope's call graph stays readable.
  function edgeDyn(st, ctx, e) {
    var onPath = ctx.path.edges.has(e.s + "|" + e.t) || ctx.path.edges.has(e.t + "|" + e.s);
    var rel = ctx.dimming
      ? (ctx.nbr.has(e.s) && ctx.nbr.has(e.t) ? Math.min(ctx.hopOpacity(e.s), ctx.hopOpacity(e.t)) : 0.05)
      : 1;
    var fo = st.focus >= 0
      ? ((st.ids[e.s].community === st.focus && st.ids[e.t].community === st.focus) ? 1
        : (st.ids[e.s].community === st.focus || st.ids[e.t].community === st.focus) ? 0.45 : 0.05)
      : 1;
    // A rail hover (kind or community row) lights matching nodes; edges keep
    // only the links inside the lit set so the membership reads as one shape.
    var rail = 1;
    if (state.kindHover) {
      rail = (st.ids[e.s].kind === state.kindHover && st.ids[e.t].kind === state.kindHover) ? 1 : 0.25;
    } else if (state.commHover >= 0) {
      rail = (st.ids[e.s].community === state.commHover && st.ids[e.t].community === state.commHover) ? 1 : 0.25;
    }
    return {
      onPath: onPath,
      w: e.defines ? 0.5 : ctx.pathing && onPath ? 2.6 : 0.5 + e.weight * 2.4,
      color: ctx.pathing && onPath ? "var(--color-accent)" : e.baseColor,
      dash: e.defines ? "none" : st.mode === "flow" ? "7 5" : e.baseDash,
      o: e.defines
        ? (ctx.pathing ? 0.04 : 0.1 * rel * fo * st.density * rail)
        : ctx.pathing
          ? (onPath ? 1 : 0.07)
          : ctx.dimming && ctx.nbr.has(e.s) && ctx.nbr.has(e.t)
            ? ((e.cross ? 0.24 : 0.14) + e.weight * 0.5) * rel * fo * rail
            : ((e.cross ? 0.24 : 0.14) + e.weight * 0.5) * rel * fo * st.density * rail,
    };
  }

  function nodeDyn(st, ctx, n) {
    var isSel = n.id === ctx.sel;
    var isHot = isSel || n.id === state.hover;
    var o = 1;
    if (ctx.pathing) o = ctx.path.nodes.has(n.id) ? 1 : 0.1;
    else if (state.kindHover) o = n.kind === state.kindHover ? 1 : 0.1;
    else if (state.commHover >= 0) o = n.community === state.commHover ? 1 : 0.1;
    else if (st.focus >= 0 && n.community !== st.focus) o = 0.12;
    else if (ctx.dimming) o = ctx.nbr.has(n.id) ? ctx.hopOpacity(n.id) : 0.13;
    return {
      o: o, z: isHot ? 3 : 1, hot: isHot,
      ring: (isSel || (ctx.pathing && ctx.path.nodes.has(n.id))) ? "0 0 0 3px var(--color-accent)" : "none",
    };
  }

  // ── Labels: rank order, four adjacent seats, then the outward search — with
  // a hard chip budget so a thousand-node scope pays for the labels it can
  // actually fit, not for a seat search over every candidate.
  function labelChips(st, ctx) {
    var chips = [];
    if (st.CW < 200 || st.CH < 160) return chips;
    var hit = function (a, b) { return a.l < b.r && a.r > b.l && a.t < b.b && a.b > b.t; };
    var area = function (b) { return Math.max(1, (b.r - b.l) * (b.b - b.t)); };
    var overlapArea = function (a, b) {
      return Math.max(0, Math.min(a.r, b.r) - Math.max(a.l, b.l)) * Math.max(0, Math.min(a.b, b.b) - Math.max(a.t, b.t));
    };
    var discHit = function (rect, selfId) {
      for (var gx = Math.floor(rect.l / st.CELL); gx <= Math.floor(rect.r / st.CELL); gx++) {
        for (var gy = Math.floor(rect.t / st.CELL); gy <= Math.floor(rect.b / st.CELL); gy++) {
          var bucket = st.discGrid.get(gx + "," + gy);
          if (!bucket) continue;
          for (var bi = 0; bi < bucket.length; bi++) {
            var d2 = st.discs[bucket[bi]];
            if (d2.id !== selfId && overlapArea(rect, d2) > 0.22 * area(d2)) return true;
          }
        }
      }
      return false;
    };

    var order = st.ns.slice().sort(function (a, b) {
      var av = (a.id === ctx.active ? 2 : ctx.nbr.has(a.id) ? 1 : 0);
      var bv = (b.id === ctx.active ? 2 : ctx.nbr.has(b.id) ? 1 : 0);
      return bv - av || b.deg - a.deg;
    });

    for (var oi = 0; oi < order.length; oi++) {
      if (chips.length >= CHIP_BUDGET) break;
      var n = order[oi];
      var r = st.radius(n);
      // Below ~420px of canvas the graph is too dense for breadcrumbs; only the
      // node under the cursor (or selected) is labelled there.
      var interesting = ctx.pathing ? ctx.path.nodes.has(n.id)
        : st.tight ? n.id === ctx.active
        : ((ctx.active && ctx.nbr.has(n.id)) || n.id === ctx.sel || st.eligible.has(n.id));
      if (!interesting) continue;
      var w = Math.min(160, Math.max(96, n.label.length * 6.2 + 12));
      if (w > st.CW - 8) continue;
      var cx = st.pos[n.id].x / S * st.CW, cy = st.pos[n.id].y / S * st.CH;
      var H = 17;
      // Seats are tried adjacent first, then pushed outward along the vector
      // away from the graph's centre. Without the outward ring the densest
      // nodes — which are exactly the hubs — find every adjacent seat taken
      // and get dropped, handing the label quota to the periphery.
      var ax = cx - st.CW / 2, ay = cy - st.CH / 2;
      var alen = Math.hypot(ax, ay) || 1;
      ax /= alen; ay /= alen;
      var seats = [
        { l: cx - w / 2, t: cy + r + 5 },
        { l: cx - w / 2, t: cy - r - 5 - H },
        { l: cx + r + 6, t: cy - H / 2 },
        { l: cx - r - 6 - w, t: cy - H / 2 },
      ];
      for (var step = 1; step <= 7; step++) {
        var dd = r + 10 + step * 16;
        [0, 0.5, -0.5, 1, -1, 1.6, -1.6].forEach(function (a) {
          var ux = ax * Math.cos(a) - ay * Math.sin(a);
          var uy = ax * Math.sin(a) + ay * Math.cos(a);
          seats.push({ l: cx + ux * dd - w / 2, t: cy + uy * dd - H / 2, far: true });
        });
      }
      for (var si = 0; si < seats.length; si++) {
        var seat = seats[si];
        var l = Math.max(2, Math.min(st.CW - w - 2, seat.l));
        var t = Math.max(2, Math.min(st.CH - H, seat.t));
        var rect = { l: l, t: t, r: l + w, b: t + H };
        var pad = { l: l - 3, t: t - 3, r: l + w + 3, b: t + H + 3 };
        if (chips.some(function (c) { return hit(pad, c.rect); })) continue;
        // The chip is opaque and painted above the discs, so a graze is
        // legible; only a real collision with a node's core disqualifies it.
        if (!st.tight && discHit(rect, n.id)) continue;
        // A chip that had to move away from its node gets a leader line, so
        // the association stays unambiguous.
        var mx = l + w / 2, my = t + H / 2;
        var far = seat.far && Math.hypot(mx - cx, my - cy) > r + 14;
        chips.push({ n: n, rect: rect, w: w, leader: far ? { cx: cx, cy: cy, mx: mx, my: my, r: r } : null });
        break;
      }
    }
    return chips;
  }

  function leadersHtml(st, ctx, chips) {
    var parts = ['<svg style="position:absolute;inset:0;width:100%;height:100%;display:block;overflow:visible;pointer-events:none;z-index:2">'];
    chips.forEach(function (c) {
      if (!c.leader) return;
      var dx = c.leader.mx - c.leader.cx, dy = c.leader.my - c.leader.cy;
      var len = Math.hypot(dx, dy) || 1;
      parts.push(
        '<line x1="' + (c.leader.cx + (dx / len) * c.leader.r).toFixed(1) + '" y1="' + (c.leader.cy + (dy / len) * c.leader.r).toFixed(1) +
        '" x2="' + (c.leader.mx - (dx / len) * 3).toFixed(1) + '" y2="' + (c.leader.my - (dy / len) * 3).toFixed(1) +
        '" vector-effect="non-scaling-stroke" style="stroke:var(--color-text);stroke-width:1px;opacity:' +
        ((c.n.id === ctx.sel || c.n.id === state.hover) ? 0.75 : 0.32) + '"></line>'
      );
    });
    parts.push("</svg>");
    return parts.join("");
  }

  function labelsHtml(st, ctx, chips) {
    return chips.map(function (c) {
      var isHot = c.n.id === ctx.sel || c.n.id === state.hover || (ctx.pathing && ctx.path.nodes.has(c.n.id));
      var o = 1;
      if (ctx.pathing) o = ctx.path.nodes.has(c.n.id) ? 1 : 0.1;
      else if (st.focus >= 0 && c.n.community !== st.focus) o = 0.12;
      else if (ctx.dimming) o = ctx.nbr.has(c.n.id) ? Math.max(0.45, ctx.hopOpacity(c.n.id)) : 0.13;
      return (
        '<div data-atlas-label="' + (isHot ? "active" : "idle") + '" style="position:absolute;left:' + c.rect.l.toFixed(1) + "px;top:" + c.rect.t.toFixed(1) +
        "px;width:" + c.w.toFixed(0) + "px;display:flex;justify-content:center;opacity:" + o.toFixed(2) + ";z-index:" + (isHot ? 4 : 2) + ';pointer-events:none">' +
        '<span style="max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;direction:rtl;text-align:left;padding:0 3px;border-radius:4px;background:color-mix(in srgb, var(--color-bg) 86%, transparent);font-size:11.5px;line-height:1.35;font-weight:' +
        (isHot ? 700 : 400) + ';color:var(--color-text);letter-spacing:-0.01em">' + esc(c.n.label) + "</span></div>"
      );
    }).join("");
  }

  function renderCanvasContents() {
    if (!D || D.unavailable) return;
    var st = buildStatics();
    if (!st) return;
    var ctx = hoverCtx(st);
    var mode = st.mode;
    var CW = st.CW, CH = st.CH;
    var E = st.E;

    var edgeViewBox = CW > 0 ? "0 0 " + CW.toFixed(0) + " " + CH.toFixed(0) : "0 0 " + S + " " + S;
    var lineMode = mode !== "taper";
    var dyn = E.map(function (e) { return edgeDyn(st, ctx, e); });

    var svgParts = ['<svg viewBox="' + edgeViewBox + '" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%;display:block;overflow:visible">'];
    if (mode === "fade" && CW > 0) {
      svgParts.push("<defs>");
      E.forEach(function (e, i) {
        if (e.defines) return;
        svgParts.push(
          '<linearGradient data-gi="' + i + '" id="ge' + e.i + '" gradientUnits="userSpaceOnUse" x1="' + e.x1.toFixed(1) + '" y1="' + e.y1.toFixed(1) + '" x2="' + e.x2.toFixed(1) + '" y2="' + e.y2.toFixed(1) + '">' +
          '<stop offset="0%" stop-color="' + e.baseColor + '" stop-opacity="' + (dyn[i].o * 0.15).toFixed(3) + '"></stop>' +
          '<stop offset="100%" stop-color="' + e.baseColor + '" stop-opacity="' + Math.min(1, dyn[i].o * 1.5).toFixed(3) + '"></stop>' +
          "</linearGradient>"
        );
      });
      svgParts.push("</defs>");
    }
    if (lineMode) {
      E.forEach(function (e, i) {
        var d = dyn[i];
        var fade = mode === "fade" && !d.onPath && !e.defines;
        var stroke = fade ? "url(#ge" + e.i + ")" : d.color;
        var o = fade ? 1 : d.o.toFixed(3);
        var anim = mode === "flow" && !e.defines && d.o > 0.12 ? "atlas-flow " + (1.6 + (1 - e.weight) * 1.6).toFixed(2) + "s linear infinite" : "none";
        svgParts.push(
          '<line data-ei="' + i + '" x1="' + e.x1.toFixed(1) + '" y1="' + e.y1.toFixed(1) + '" x2="' + e.x2.toFixed(1) + '" y2="' + e.y2.toFixed(1) +
          '" vector-effect="non-scaling-stroke" style="stroke:' + stroke + ";stroke-width:" + d.w + "px;stroke-dasharray:" + d.dash + ";opacity:" + o + ";stroke-linecap:round;animation:" + anim + '"></line>'
        );
      });
    } else if (CW > 0) {
      E.forEach(function (e, i) {
        var d = dyn[i];
        // Taper mode draws wedges for dependencies; scaffolding stays a hairline.
        if (e.defines) {
          svgParts.push(
            '<line data-ei="' + i + '" x1="' + e.x1.toFixed(1) + '" y1="' + e.y1.toFixed(1) + '" x2="' + e.x2.toFixed(1) + '" y2="' + e.y2.toFixed(1) +
            '" vector-effect="non-scaling-stroke" style="stroke:' + d.color + ";stroke-width:0.5px;opacity:" + d.o.toFixed(3) + '"></line>'
          );
          return;
        }
        var dx = e.x2 - e.x1, dy = e.y2 - e.y1;
        var len = Math.hypot(dx, dy) || 1;
        var nx = -dy / len, ny = dx / len;
        var hs = (d.w * 1.9) / 2, ht = 0.35;
        var tx2 = e.x2 - (dx / len) * (e.rt + 1), ty2 = e.y2 - (dy / len) * (e.rt + 1);
        var pt = function (x, y) { return x.toFixed(1) + "," + y.toFixed(1); };
        svgParts.push(
          '<polygon data-ei="' + i + '" points="' +
          [pt(e.x1 + nx * hs, e.y1 + ny * hs), pt(tx2 + nx * ht, ty2 + ny * ht), pt(tx2 - nx * ht, ty2 - ny * ht), pt(e.x1 - nx * hs, e.y1 - ny * hs)].join(" ") +
          '" style="fill:' + d.color + ";opacity:" + Math.min(1, d.o * 1.25).toFixed(3) + '"></polygon>'
        );
      });
    }
    svgParts.push("</svg>");

    // Direction: a small arrowhead just short of the target node. Drawn in the
    // measured pixel space (not the stretched SVG) so it never skews. Past two
    // thousand edges only the well-evidenced get heads — at that density a head
    // per edge is texture, not direction.
    var arrowParts = [];
    var arrowMin = E.length > 2000 ? 0.25 : 0.09;
    if ((mode === "arrows" || (ctx.pathing && mode !== "taper")) && CW > 0) {
      E.forEach(function (e, i) {
        var d = dyn[i];
        if (e.defines || d.o < arrowMin) return;
        if (mode !== "arrows" && !d.onPath) return;
        var dx = e.x2 - e.x1, dy = e.y2 - e.y1;
        var len = Math.hypot(dx, dy) || 1;
        arrowParts.push(
          '<div data-ai="' + i + '" style="position:absolute;left:' + (e.x2 - (dx / len) * (e.rt + 5)).toFixed(1) + "px;top:" + (e.y2 - (dy / len) * (e.rt + 5)).toFixed(1) +
          "px;width:9px;height:7px;background:" + d.color + ";opacity:" + Math.min(1, d.o + 0.15).toFixed(2) +
          ";transform:translate(-50%,-50%) rotate(" + (Math.atan2(dy, dx) * 180 / Math.PI).toFixed(1) + "deg);clip-path:polygon(100% 50%, 0 0, 0 100%);pointer-events:none\"></div>"
        );
      });
    }

    var chips = labelChips(st, ctx);
    var nodeParts = st.ns.map(function (n) {
      var r = st.radius(n);
      var c = nodeColor(n, st.comms);
      var d = nodeDyn(st, ctx, n);
      var sh = SHAPES(n, c, r);
      return (
        '<div data-node="' + esc(n.id) + '" style="position:absolute;left:' + (st.pos[n.id].x / S * 100).toFixed(2) + "%;top:" + (st.pos[n.id].y / S * 100).toFixed(2) +
        "%;display:flex;flex-direction:column;align-items:center;gap:3px;cursor:pointer;opacity:" + d.o.toFixed(2) + ";z-index:" + d.z +
        ';transform:translate(-50%,-50%)">' +
        '<span style="width:' + sh.w + "px;height:" + sh.h + "px;flex:none;border-radius:" + sh.radius + ";rotate:" + sh.rot +
        ";display:block;background:" + sh.fill + ";border:" + sh.border + ";box-shadow:" + d.ring + '"></span>' +
        "</div>"
      );
    });

    st.layer.innerHTML =
      svgParts.join("") + arrowParts.join("") + nodeParts.join("") +
      '<div data-leaders-host style="position:absolute;inset:0;pointer-events:none">' + leadersHtml(st, ctx, chips) + "</div>" +
      '<div data-labels-host style="position:absolute;inset:0;pointer-events:none">' + labelsHtml(st, ctx, chips) + "</div>";

    // Collect references so a hover can restyle without rebuilding.
    st.edgeEls = {};
    st.layer.querySelectorAll("[data-ei]").forEach(function (el) { st.edgeEls[el.getAttribute("data-ei")] = el; });
    st.gradEls = {};
    st.layer.querySelectorAll("[data-gi]").forEach(function (el) { st.gradEls[el.getAttribute("data-gi")] = el.querySelectorAll("stop"); });
    st.arrowEls = {};
    st.layer.querySelectorAll("[data-ai]").forEach(function (el) { st.arrowEls[el.getAttribute("data-ai")] = el; });
    st.nodeEls = {};
    st.ringEls = {};
    st.layer.querySelectorAll("[data-node]").forEach(function (el) {
      var id = el.getAttribute("data-node");
      st.nodeEls[id] = el;
      st.ringEls[id] = el.firstElementChild;
    });
    st.leadersHost = st.layer.querySelector("[data-leaders-host]");
    st.labelsHost = st.layer.querySelector("[data-labels-host]");
    scene = st;
  }

  // The hover path: restyle the scene in place. Structure never changes here —
  // only opacity, stroke, width and the label layers, which are budget-bounded.
  function restyleHover() {
    if (!scene) {
      renderCanvasContents();
      return;
    }
    var st = scene;
    var ctx = hoverCtx(st);
    var mode = st.mode;

    st.E.forEach(function (e, i) {
      var d = edgeDyn(st, ctx, e);
      var el = st.edgeEls[i];
      if (el) {
        if (el.tagName === "polygon") {
          el.style.fill = d.color;
          el.style.opacity = Math.min(1, d.o * 1.25).toFixed(3);
        } else {
          var fade = mode === "fade" && !d.onPath && !e.defines;
          el.style.stroke = fade ? "url(#ge" + e.i + ")" : d.color;
          el.style.opacity = fade ? 1 : d.o.toFixed(3);
          el.style.strokeWidth = d.w + "px";
          if (!e.defines) el.style.strokeDasharray = d.dash;
          if (mode === "flow" && !e.defines) {
            el.style.animation = d.o > 0.12 ? "atlas-flow " + (1.6 + (1 - e.weight) * 1.6).toFixed(2) + "s linear infinite" : "none";
          }
        }
      }
      var stops = st.gradEls[i];
      if (stops && stops.length === 2) {
        stops[0].setAttribute("stop-opacity", (d.o * 0.15).toFixed(3));
        stops[1].setAttribute("stop-opacity", Math.min(1, d.o * 1.5).toFixed(3));
      }
      var arrow = st.arrowEls[i];
      if (arrow) {
        arrow.style.background = d.color;
        arrow.style.opacity = d.o < 0.09 ? 0 : Math.min(1, d.o + 0.15).toFixed(2);
      }
    });

    st.ns.forEach(function (n) {
      var d = nodeDyn(st, ctx, n);
      var el = st.nodeEls[n.id];
      if (el) {
        el.style.opacity = d.o.toFixed(2);
        el.style.zIndex = d.z;
      }
      var ring = st.ringEls[n.id];
      if (ring) ring.style.boxShadow = d.ring;
    });

    var chips = labelChips(st, ctx);
    st.leadersHost.innerHTML = leadersHtml(st, ctx, chips);
    st.labelsHost.innerHTML = labelsHtml(st, ctx, chips);
  }

  // ── Overlays: arming banner, focus chip, focus-empty card, legend ────────
  function renderOverlays() {
    if (!D || D.unavailable) return;
    var overlay = els.main.querySelector("[data-overlays]");
    if (!overlay) return;
    var comms = communityById();
    var html = "";

    if (state.arming && state.pathFrom) {
      var from = byId()[state.pathFrom];
      html +=
        '<div style="position:absolute;left:50%;top:14px;transform:translateX(-50%);z-index:22;display:flex;align-items:center;gap:10px;background:var(--color-accent);color:var(--color-bg);border-radius:999px;padding:6px 8px 6px 14px;box-shadow:var(--shadow-md);font-size:12px;pointer-events:auto">' +
        "<span>Pick a second node — path from " + esc(from ? from.label : "") + "</span>" +
        '<button data-clear-path style="background:color-mix(in srgb, #000 18%, transparent);border:0;border-radius:999px;padding:2px 9px;cursor:pointer;color:inherit;font:inherit;font-size:11px">Cancel</button>' +
        "</div>";
    }

    if (state.focus >= 0) {
      var c = comms[state.focus];
      var drawn = (D.nodes || []).some(function (n) { return n.community === state.focus; });
      var focusName = c ? c.name.split(".").slice(-1)[0] : String(state.focus);
      if (!drawn && c) {
        // Focusing a community whose modules all sit outside the drawn subset
        // must read as "not drawn here", never as "empty".
        html +=
          '<div style="position:absolute;left:50%;top:44%;transform:translate(-50%,-50%);z-index:21;width:min(420px,80%);background:var(--color-bg);border:1px solid var(--color-divider);border-radius:12px;padding:14px 16px;box-shadow:var(--shadow-md);pointer-events:auto">' +
          '<div style="' + LABEL10 + '">Nothing to show for this community</div>' +
          '<div style="margin-top:6px;font-size:13px;line-height:1.55">None of ' + esc(c.name) + "'s " + c.count +
          " members are among the " + (D.nodes || []).length +
          " drawn here — hidden by the filters or the fold, not missing from the index.</div>" +
          '<button data-clear-focus class="btn btn-secondary" style="margin-top:11px;height:28px;font-size:12px">Clear focus</button>' +
          "</div>";
      }
      html +=
        '<div style="position:absolute;right:14px;top:14px;z-index:20;pointer-events:auto">' +
        '<button data-clear-focus class="tag tag-outline" style="font-size:11px;padding:4px 11px;background:var(--color-bg);cursor:pointer;font-family:inherit">Focused: ' + esc(focusName) + "  ✕</button>" +
        "</div>";
    }

    // Edge evidence legend — collapsible, bottom right.
    var evidenceRows = [
      ["Structural", 2.9, "none", 0.95, "weight 1.0"],
      ["Resolved", 2.2, "none", 0.75, "weight 0.7"],
      ["Guessed", 1.5, "5 4", 0.6, "weight 0.45"],
      ["Unknown", 1.0, "1.5 5", 0.45, "weight 0.2"],
    ];
    var taper = state.direction === "taper";
    var legendNote = {
      arrows: "The arrow points from the dependent to what it depends on.",
      taper: "Each line thins toward what it depends on; in this mode evidence reads from weight alone.",
      fade: "Each line brightens toward what it depends on.",
      flow: "Dashes travel toward what the module depends on.",
    }[state.direction] + " Thickness and opacity carry the edge's weight — how well evidenced it is, not how often the call runs. Unknown means evidence was never looked up, not that none exists.";
    html +=
      '<div style="position:absolute;right:14px;bottom:14px;z-index:20;width:230px;background:var(--color-bg);border:1px solid var(--color-divider);border-radius:12px;box-shadow:var(--shadow-md);overflow:hidden;pointer-events:auto">' +
      '<button data-legend class="hv6" style="display:flex;align-items:center;gap:8px;width:100%;background:transparent;border:0;padding:9px 12px;cursor:pointer;color:inherit;font:inherit;text-align:left">' +
      '<span style="font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:color-mix(in srgb, var(--color-text) 55%, transparent)">Edge evidence</span>' +
      '<span style="margin-left:auto;font-size:11px;color:color-mix(in srgb, var(--color-text) 50%, transparent)">' + (state.legend ? "▾" : "▸") + "</span>" +
      "</button>";
    if (state.legend) {
      html += '<div style="padding:2px 12px 11px">';
      evidenceRows.forEach(function (row) {
        var h = row[1] / 2;
        // taper mode draws no strokes, so the key shows wedges and drops the
        // dash column rather than illustrating an encoding that isn't on screen
        var taperPoints = [[0, 5 - h], [30, 4.7], [30, 5.3], [0, 5 + h]].map(function (pt) { return pt[0].toFixed(2) + "," + pt[1].toFixed(2); }).join(" ");
        html +=
          '<div style="display:flex;align-items:center;gap:9px;padding:3px 0;font-size:11.5px">' +
          '<svg width="30" height="10" viewBox="0 0 30 10" style="flex:none;display:block">' +
          '<line x1="0" y1="5" x2="30" y2="5" style="display:' + (taper ? "none" : "block") + ";stroke:var(--color-text);stroke-width:" + row[1] + "px;stroke-dasharray:" + (taper ? "none" : row[2]) + ";opacity:" + row[3] + ';stroke-linecap:round"></line>' +
          '<polygon points="' + taperPoints + '" style="display:' + (taper ? "block" : "none") + ";fill:var(--color-text);opacity:" + row[3] + '"></polygon>' +
          "</svg>" +
          '<span style="flex:1">' + row[0] + "</span>" +
          '<span style="color:color-mix(in srgb, var(--color-text) 50%, transparent)">' + row[4] + "</span>" +
          "</div>";
      });
      html +=
        '<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--color-divider);font-size:10.5px;line-height:1.45;color:color-mix(in srgb, var(--color-text) 60%, transparent)">' + legendNote + "</div></div>";
    }
    html += "</div>";
    overlay.innerHTML = html;
  }

  // ── Context panel ────────────────────────────────────────────────────────
  function chipHtml(ev) {
    var c = CHIP[ev] || CHIP.unknown;
    return (
      '<span style="flex:none;font-size:9.5px;letter-spacing:0.06em;border-radius:999px;padding:2px 7px;background:' +
      c.bg + ";color:" + c.fg + ";border:" + c.bd + '">' + c.t + "</span>"
    );
  }

  function renderContext() {
    if (!D) return;
    var entity = state.level === "entity";
    var ids = byId();
    var path = pathSets();
    var html = "";

    if (path.hops) {
      var a = ids[state.pathFrom], b = ids[state.pathTo];
      var guessed = path.hops.filter(function (h) { return h.ev === "guessed"; }).length;
      var unknown = path.hops.filter(function (h) { return h.ev === "unknown"; }).length;
      var note = (guessed || unknown)
        ? "This path is not fully resolved: " +
          [guessed ? guessed + " guessed" : null, unknown ? unknown + " unlooked-up" : null].filter(Boolean).join(" and ") +
          " of " + path.hops.length + " hops."
        : "Every hop on this path is structural or resolved.";
      html +=
        '<div style="padding:14px">' +
        '<div style="background:var(--color-surface);border-radius:12px;padding:12px 13px">' +
        '<div style="display:flex;align-items:baseline;margin-bottom:6px">' +
        '<div style="' + LABEL10 + '">Path</div>' +
        '<button data-clear-path class="hv8" style="margin-left:auto;background:transparent;border:0;border-radius:6px;padding:0 5px;cursor:pointer;color:inherit;font:inherit;font-size:12px;opacity:0.6">✕</button>' +
        "</div>" +
        '<div style="font-family:var(--font-heading);font-weight:800;font-size:15px;line-height:1.25;word-break:break-word">' + esc(a ? a.label : "") + "</div>" +
        '<div style="font-size:12px;color:var(--color-accent-700);margin:3px 0">↓ ' + (path.hops.length === 1 ? "1 hop" : path.hops.length + " hops") + "</div>" +
        '<div style="font-family:var(--font-heading);font-weight:800;font-size:15px;line-height:1.25;word-break:break-word">' + esc(b ? b.label : "") + "</div>" +
        '<div style="margin-top:9px;font-size:11.5px;line-height:1.5;color:color-mix(in srgb, var(--color-text) 62%, transparent)">' + note + "</div>" +
        "</div>" +
        '<div style="margin-top:14px">' +
        '<div style="' + LABEL10 + ';margin-bottom:6px;padding:0 2px">Hop by hop</div>' +
        path.hops.map(function (h, i) {
          var to = ids[h.to];
          return (
            '<div style="display:flex;align-items:center;gap:8px;padding:6px 8px;border-bottom:1px solid var(--color-divider);font-size:12px">' +
            '<span style="flex:none;width:16px;font-variant-numeric:tabular-nums;color:color-mix(in srgb, var(--color-text) 45%, transparent)">' + (i + 1) + "</span>" +
            '<span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;direction:rtl;text-align:left">' + esc(to ? to.label : h.to) + "</span>" +
            chipHtml(h.ev) +
            "</div>"
          );
        }).join("") +
        "</div>" +
        '<div style="margin-top:12px;font-size:11px;line-height:1.5;color:color-mix(in srgb, var(--color-text) 60%, transparent)">' +
        "Shortest path over indexed edges, direction ignored. Edges the resolver never looked up cannot appear here, so a shorter route may exist." +
        "</div></div>";
      els.ctx.innerHTML = html;
      return;
    }

    var n = state.selected ? ids[state.selected] : null;
    if (n) {
      var inbound = [], outbound = [];
      (D.edges || []).forEach(function (e) {
        // Containment scaffolding stays on the canvas; listing a module's every
        // member as "depends on this" would say nothing about dependencies.
        if (e.rel === "defines") return;
        if (e.t === n.id && ids[e.s]) inbound.push({ node: ids[e.s], ev: e.ev });
        if (e.s === n.id && ids[e.t]) outbound.push({ node: ids[e.t], ev: e.ev });
      });
      var kindDef = (D.kinds || []).filter(function (k) { return k.id === n.kind; })[0];
      var selKind = entity ? ((kindDef && kindDef.label) || n.kind) : "module";
      var outside = n.kind === "external_package" || n.kind === "external_symbol";
      var selFile;
      if (entity) {
        selFile = outside
          ? "outside the index — no file recorded"
          : "in " + (state.scope ? (D.scope_name || state.scope) : (n.path || "the index")) + (n.lines ? " · lines " + n.lines : "");
      } else {
        selFile = (n.path || "") + " · degree " + n.deg + " in full index";
      }
      var relRow = function (r) {
        return (
          '<button data-goto="' + esc(r.node.id) + '" class="hv6" style="display:flex;align-items:center;gap:8px;width:100%;background:transparent;border:0;border-radius:8px;padding:5px 8px;cursor:pointer;color:inherit;font:inherit;font-size:12px;text-align:left">' +
          '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;direction:rtl;text-align:left">' + esc(r.node.label) + "</span>" +
          chipHtml(r.ev) +
          "</button>"
        );
      };
      html +=
        '<div style="padding:14px">' +
        '<div style="background:var(--color-surface);border-radius:12px;padding:12px 13px">' +
        '<div style="display:flex;align-items:baseline;margin-bottom:6px">' +
        '<div style="' + LABEL10 + '">' + (entity ? "Selected entity" : "Selected module") + "</div>" +
        '<button data-clear-sel class="hv8" style="margin-left:auto;background:transparent;border:0;border-radius:6px;padding:0 5px;cursor:pointer;color:inherit;font:inherit;font-size:12px;opacity:0.6">✕</button>' +
        "</div>" +
        '<div style="font-family:var(--font-heading);font-weight:800;font-size:17px;line-height:1.2;word-break:break-word">' + esc(n.label) + "</div>" +
        '<div style="margin-top:6px;font-size:11.5px;color:color-mix(in srgb, var(--color-text) 60%, transparent)">' + esc(selFile) + "</div>" +
        '<div style="display:flex;gap:6px;margin-top:9px">' +
        '<span class="tag tag-neutral" style="font-size:10.5px">' + esc(selKind) + "</span>" +
        '<span class="tag tag-neutral" style="font-size:10.5px">' +
        (entity ? inbound.length + " in · " + outbound.length + " out, within this scope" : "degree " + n.deg + " in full index") +
        "</span>" +
        "</div>" +
        '<div style="display:flex;gap:6px;margin-top:11px">' +
        (n.uid ? '<button data-open-detail="' + esc(n.uid) + '" class="btn btn-primary" style="height:30px;font-size:12px">Open detail</button>' : "") +
        '<button data-arm-path class="btn btn-secondary" style="height:30px;font-size:12px">Path to…</button>' +
        (!entity && n.path ? '<button data-open-entities="' + esc(n.path) + '" class="btn btn-secondary" style="height:30px;font-size:12px">Entities</button>' : "") +
        "</div>" +
        "</div>" +
        '<div style="margin-top:14px">' +
        '<div style="display:flex;align-items:baseline;margin-bottom:6px;padding:0 2px">' +
        '<div style="' + LABEL10 + '">Depends on this</div>' +
        '<div style="margin-left:auto;font-size:11px;color:color-mix(in srgb, var(--color-text) 50%, transparent)">' + inbound.length + "</div>" +
        "</div>" +
        inbound.map(relRow).join("") +
        "</div>" +
        '<div style="margin-top:14px">' +
        '<div style="display:flex;align-items:baseline;margin-bottom:6px;padding:0 2px">' +
        '<div style="' + LABEL10 + '">This depends on</div>' +
        '<div style="margin-left:auto;font-size:11px;color:color-mix(in srgb, var(--color-text) 50%, transparent)">' + outbound.length + "</div>" +
        "</div>" +
        outbound.map(relRow).join("") +
        "</div>" +
        '<div style="margin-top:14px;font-size:11px;line-height:1.5;color:color-mix(in srgb, var(--color-text) 60%, transparent)">' +
        (entity
          ? "Counts cover this module's scope only. Dependencies crossing into other modules are on the module level."
          : "Neighbours listed are those the index holds. Callers reached only through unresolved dynamic dispatch are not counted here.") +
        "</div></div>";
      els.ctx.innerHTML = html;
      return;
    }

    els.ctx.innerHTML =
      '<div style="padding:14px;font-size:12.5px;line-height:1.55;color:color-mix(in srgb, var(--color-text) 60%, transparent)">' +
      '<div style="font-size:10px;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px">No selection</div>' +
      "Click a node to see its dependencies, evidence and file location. Hover to isolate its immediate neighbourhood." +
      "</div>";
  }

  // ── Wiring: sidebar + context delegation ─────────────────────────────────
  els.aside.addEventListener("click", function (e) {
    var section = e.target.closest("[data-section]");
    if (section) {
      var id = section.getAttribute("data-section");
      var at = state.open.indexOf(id);
      if (at >= 0) state.open.splice(at, 1);
      else state.open.push(id);
      renderSidebar();
      return;
    }
    var toggle = e.target.closest("[data-toggle]");
    if (toggle) {
      var which = toggle.getAttribute("data-toggle");
      if (which === "tests") {
        localStorage.setItem("atlas.showTests", state.showTests ? "0" : "1");
        setAndLoad({ showTests: !state.showTests, selected: null });
      } else if (which === "noncode") {
        localStorage.setItem("atlas.showNoncode", state.showNoncode ? "0" : "1");
        setAndLoad({ showNoncode: !state.showNoncode, selected: null });
      } else if (which === "external") {
        if (state.level === "entity") {
          var hidden2 = new Set(state.hidden || []);
          if (hidden2.has("external_package") || hidden2.has("external_symbol")) {
            hidden2.delete("external_package");
            hidden2.delete("external_symbol");
          } else {
            hidden2.add("external_package");
            hidden2.add("external_symbol");
          }
          setAndLoad({ hidden: hidden2, selected: null });
        } else {
          localStorage.setItem("atlas.showExternal", state.showExternal ? "0" : "1");
          setAndLoad({ showExternal: !state.showExternal, selected: null });
        }
      } else if (which === "expand") {
        setAndLoad({ expandMethods: !state.expandMethods, selected: null });
      }
      return;
    }
    var scopeExp = e.target.closest("[data-scope-exp]");
    if (scopeExp) {
      var dirPath = scopeExp.getAttribute("data-scope-exp");
      if (state.scopeOpen.has(dirPath)) state.scopeOpen.delete(dirPath);
      else state.scopeOpen.add(dirPath);
      renderSidebar();
      return;
    }
    var scopePick = e.target.closest("[data-scope]");
    if (scopePick) {
      setAndLoad({ scope: scopePick.getAttribute("data-scope"), selected: null, focus: -1 });
      return;
    }
    var kindToggle = e.target.closest("[data-kind-toggle]");
    if (kindToggle) {
      var kindId = kindToggle.getAttribute("data-kind-toggle");
      if (kindId === "method") {
        setAndLoad({ expandMethods: !state.expandMethods, selected: null });
        return;
      }
      var hidden = new Set(state.hidden || []);
      if (hidden.has(kindId)) hidden.delete(kindId);
      else hidden.add(kindId);
      setAndLoad({ hidden: hidden, selected: null });
      return;
    }
    var comm = e.target.closest("[data-comm]");
    if (comm) {
      var cid = parseInt(comm.getAttribute("data-comm"), 10);
      setAndRender({ focus: state.focus === cid ? -1 : cid });
      return;
    }
    var dir = e.target.closest("[data-dir]");
    if (dir) {
      localStorage.setItem("atlas.direction", dir.getAttribute("data-dir"));
      setAndRender({ direction: dir.getAttribute("data-dir") });
      return;
    }
  });
  els.aside.addEventListener("change", function (e) {
    var t = e.target;
    if (t.dataset.level) {
      var level = t.dataset.level;
      if (level === "entity") setAndLoad({ level: "entity", scope: "", selected: null, focus: -1 });
      else setAndLoad({ level: "module", selected: null, focus: -1 });
    } else if (t.dataset.hops) {
      localStorage.setItem("atlas.hops", t.dataset.hops);
      setAndRender({ hops: parseInt(t.dataset.hops, 10) });
    } else if (t.dataset.labels) {
      localStorage.setItem("atlas.labels", t.dataset.labels);
      setAndRender({ labels: t.dataset.labels });
    }
  });
  els.aside.addEventListener("focusin", function (e) {
    if (e.target.id === "rail-search") window.location.href = "/search";
  });
  // Hovering a rail row lights its members on the canvas — a preview, not a state.
  els.aside.addEventListener("mouseover", function (e) {
    var kindRow = e.target.closest("[data-kind-row]");
    var commRow = e.target.closest("[data-comm]");
    var kind = kindRow ? kindRow.getAttribute("data-kind-row") : null;
    var comm = commRow ? parseInt(commRow.getAttribute("data-comm"), 10) : -1;
    if (state.kindHover !== kind || state.commHover !== comm) {
      state.kindHover = kind;
      state.commHover = comm;
      restyleHover();
    }
  });
  els.aside.addEventListener("mouseleave", function () {
    if (state.kindHover !== null || state.commHover >= 0) {
      state.kindHover = null;
      state.commHover = -1;
      restyleHover();
    }
  });
  els.main.addEventListener("click", function (e) {
    if (e.target.closest("[data-clear-path]")) {
      set({ arming: false, pathFrom: null, pathTo: null });
      render();
    } else if (e.target.closest("[data-clear-focus]")) {
      setAndRender({ focus: -1 });
    } else if (e.target.closest("[data-legend]")) {
      state.legend = !state.legend;
      renderOverlays();
    }
  });
  els.ctx.addEventListener("click", function (e) {
    var open = e.target.closest("[data-open-detail]");
    if (open) {
      window.location.href = "/entity/" + encodeURIComponent(open.getAttribute("data-open-detail"));
      return;
    }
    if (e.target.closest("[data-arm-path]")) {
      set({ arming: true, pathFrom: state.selected, pathTo: null });
      renderOverlays();
      return;
    }
    var entities = e.target.closest("[data-open-entities]");
    if (entities) {
      setAndLoad({ level: "entity", scope: entities.getAttribute("data-open-entities"), selected: null });
      return;
    }
    var go = e.target.closest("[data-goto]");
    if (go) {
      onSelect(go.getAttribute("data-goto"));
      return;
    }
    if (e.target.closest("[data-clear-sel]")) {
      set({ selected: null });
      renderCanvasContents();
      renderContext();
      return;
    }
    if (e.target.closest("[data-clear-path]")) {
      set({ arming: false, pathFrom: null, pathTo: null });
      render();
    }
  });
  window.addEventListener("resize", function () {
    renderCanvasContents();
  });
  window.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    hideMenu();
    if (state.arming) {
      set({ arming: false, pathFrom: null, pathTo: null });
      render();
    }
  });

  load();
})();
