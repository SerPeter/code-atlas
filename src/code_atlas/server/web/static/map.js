/**
 * Map renderer.
 *
 * The one client-side island: everything else is server-rendered, but a pannable,
 * zoomable graph is what HTMX cannot express.
 *
 * Node positions arrive precomputed and seeded from the server. No force simulation runs
 * here — a layout that settles in the browser draws the same graph differently on every
 * reload, which destroys the map's only real job: letting someone recognise their
 * codebase and notice when something moved.
 *
 * Display settings arrive as data attributes on the canvas, set from the query string,
 * so a view is reproducible by pasting its address.
 */
(function () {
  "use strict";

  var container = document.getElementById("map-canvas");
  var status = document.getElementById("map-status");
  if (!container || !status) return;
  if (typeof graphology === "undefined" || typeof Sigma === "undefined") {
    status.textContent = "Renderer failed to load — /static/vendor assets are missing.";
    return;
  }

  var opts = {
    direction: container.dataset.direction || "arrows",
    hops: parseInt(container.dataset.hops || "1", 10),
    labels: container.dataset.labels || "some",
    level: container.dataset.level || "module",
    focus: parseInt(container.dataset.focus || "-1", 10),
  };

  var NEUTRAL = "var(--atlas-c8)";
  function communityColor(i) {
    return "var(--atlas-c" + (i >= 0 && i < 8 ? i : 8) + ")";
  }

  /* Sigma paints through its own colour parser, which understands hex and rgb()/rgba()
     and nothing else. The palette is authored in oklch() and the dimmed variants in
     color-mix() — neither parses, so every node and edge came out pure black.

     The browser does the conversion: assign the value to a detached element and read it
     back, which yields rgb()/rgba() whatever notation went in. One palette definition
     still serves both the rail's swatches and the canvas. */
  var probe = document.createElement("span");
  probe.style.display = "none";
  document.body.appendChild(probe);
  var colorCache = {};

  function resolve(token) {
    if (token in colorCache) return colorCache[token];
    probe.style.color = "";
    probe.style.color = token;
    var out = getComputedStyle(probe).color;
    // An unparseable value leaves the property unset; fall back rather than paint black.
    if (!out || out === "rgba(0, 0, 0, 0)") out = "#888888";
    colorCache[token] = out;
    return out;
  }

  function alpha(token, a) {
    // Resolve first, then apply opacity numerically. Handing color-mix() to sigma would
    // only fail again; rgb() is trivial to reopen.
    var m = /rgba?\(([^)]+)\)/.exec(resolve(token));
    if (!m) return resolve(token);
    var p = m[1].split(",");
    return "rgba(" + parseFloat(p[0]) + "," + parseFloat(p[1]) + "," + parseFloat(p[2]) + "," + a + ")";
  }

  function loadMap() {
    var embedded = document.getElementById("map-data");
    if (embedded) return Promise.resolve(JSON.parse(embedded.textContent));
    return fetch(container.dataset.api || "/map/api").then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  loadMap()
    .then(function (data) {
      var graph = new graphology.Graph({ type: "directed", multi: false });

      /* Node size is the degree the server computed, already log-compressed: linear
         sizing lets one 59-degree hub render every other module as a dot. */
      data.nodes.forEach(function (n) {
        var base = n.is_external ? NEUTRAL : communityColor(n.community);
        graph.addNode(n.id, {
          label: n.label,
          x: n.x,
          y: n.y,
          size: n.size,
          color: resolve(base),
          baseColor: resolve(base),
          community: n.community,
          kind: n.kind || "",
          external: !!n.is_external,
          project: n.project,
        });
      });

      /* Edge thickness tracks the stored ADR-0017 weight. Raw weights span 0.0027 to
         126.79 on the real graph — a 47,000x range — so it is log-compressed or one
         edge would be wider than the viewport.

         An edge inside a community takes that community's colour; one that crosses
         stays neutral. Doing it the other way round would accent 81% of the edges,
         which accents nothing. */
      var weights = data.edges.map(function (e) {
        return e.weight;
      });
      var maxW = Math.max.apply(null, weights.concat([1]));
      data.edges.forEach(function (e) {
        if (!graph.hasNode(e.source) || !graph.hasNode(e.target)) return;
        if (graph.hasEdge(e.source, e.target)) return;
        var within = !e.crosses_community;
        var tint = within ? communityColor(graph.getNodeAttribute(e.source, "community")) : NEUTRAL;
        graph.addEdge(e.source, e.target, {
          size: 0.4 + (Math.log1p(e.weight) / Math.log1p(maxW)) * 5.6,
          color: alpha(tint, within ? 0.55 : 0.28),
          baseSize: 0.4 + (Math.log1p(e.weight) / Math.log1p(maxW)) * 5.6,
          type: opts.direction === "arrows" ? "arrow" : opts.direction === "curved" ? "curve" : "line",
          weight: e.weight,
        });
      });

      /* Labels are ranked, not thresholded: "Hubs" must label the actual hubs rather
         than whatever happened to have empty space beside it. */
      var ranked = graph
        .nodes()
        .slice()
        .sort(function (a, b) {
          return graph.degree(b) - graph.degree(a);
        });
      var labelCount =
        opts.labels === "all" ? ranked.length : opts.labels === "few" ? Math.ceil(ranked.length * 0.12) : Math.ceil(ranked.length * 0.45);
      var labelled = new Set(ranked.slice(0, labelCount));

      var hovered = null;
      var neighbourhood = null;

      function withinHops(root, hops) {
        var seen = new Set([root]);
        var frontier = [root];
        for (var i = 0; i < hops; i++) {
          var next = [];
          frontier.forEach(function (id) {
            graph.neighbors(id).forEach(function (n) {
              if (!seen.has(n)) {
                seen.add(n);
                next.push(n);
              }
            });
          });
          frontier = next;
        }
        return seen;
      }

      var DIM = alpha("var(--color-text)", 0.12);

      var renderer = new Sigma(graph, container, {
        renderEdgeLabels: false,
        defaultEdgeType: opts.direction === "arrows" ? "arrow" : "line",
        labelDensity: 1,
        labelGridCellSize: 0,
        labelRenderedSizeThreshold: 0,
        nodeReducer: function (key, attrs) {
          var out = attrs;
          if (!labelled.has(key)) out = Object.assign({}, out, { label: "" });
          if (opts.focus >= 0 && attrs.community !== opts.focus) {
            out = Object.assign({}, out, { color: DIM, label: "" });
          }
          if (neighbourhood && !neighbourhood.has(key)) {
            out = Object.assign({}, out, { color: DIM, label: "" });
          }
          if (hovered === key) out = Object.assign({}, out, { label: attrs.label, highlighted: true });
          return out;
        },
        edgeReducer: function (key, attrs) {
          if (neighbourhood) {
            var ends = graph.extremities(key);
            if (!neighbourhood.has(ends[0]) || !neighbourhood.has(ends[1])) {
              return Object.assign({}, attrs, { hidden: true });
            }
          }
          if (opts.focus >= 0) {
            var e = graph.extremities(key);
            if (
              graph.getNodeAttribute(e[0], "community") !== opts.focus &&
              graph.getNodeAttribute(e[1], "community") !== opts.focus
            ) {
              return Object.assign({}, attrs, { hidden: true });
            }
          }
          return attrs;
        },
      });

      function describe(n) {
        var a = graph.getNodeAttributes(n);
        return a.external
          ? n + " — external, owned by '" + a.project + "'"
          : n + " — " + graph.degree(n) + " dependencies" + (a.kind ? " · " + a.kind : "");
      }

      renderer.on("enterNode", function (e) {
        hovered = e.node;
        neighbourhood = withinHops(e.node, opts.hops);
        renderer.refresh();
        status.textContent = describe(e.node);
      });
      renderer.on("leaveNode", function () {
        hovered = null;
        neighbourhood = null;
        renderer.refresh();
        status.textContent = summary;
      });
      renderer.on("clickNode", function (e) {
        selectNode(e.node);
      });

      var summary =
        graph.order + (opts.level === "entity" ? " entities, " : " modules, ") + graph.size + " dependencies.";
      status.textContent = summary;

      /* Selection panel in the rail. */
      var panel = document.getElementById("rail-selection");
      function selectNode(id) {
        if (!panel) return;
        var a = graph.getNodeAttributes(id);
        panel.hidden = false;
        document.getElementById("sel-name").textContent = a.label || id;
        document.getElementById("sel-meta").textContent =
          graph.degree(id) + " dependencies" + (a.kind ? " · " + a.kind : "") + (a.external ? " · external" : "");
        /* Both need a server. In a static export there is none, so the links are left
           inert rather than pointing at an address that cannot answer. */
        var live = !document.getElementById("map-data");
        var detail = document.getElementById("sel-detail");
        var impact = document.getElementById("sel-impact");
        if (detail) detail.href = live ? "/entity/" + encodeURIComponent(id) : "#";
        if (impact) impact.href = live ? "/impact?uid=" + encodeURIComponent(id) : "#";
      }

    })
    .catch(function (error) {
      status.textContent = "Could not load the map: " + error.message;
    });
})();
