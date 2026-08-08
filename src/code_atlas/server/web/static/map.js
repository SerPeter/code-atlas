/**
 * Community map renderer.
 *
 * The one client-side island in this UI. Everything else is server-rendered HTML, but a
 * pannable, zoomable graph is the part HTMX cannot express.
 *
 * Node positions arrive precomputed from the server. No force simulation runs here, and
 * that is deliberate: a layout that settles in the browser draws the same graph
 * differently on every reload, which destroys the map's only real job — letting someone
 * recognise their codebase and notice when a module has moved between subsystems.
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

  /* Sequential, not random: a community keeps its colour across reloads. A map that
     recolours itself cannot be compared with the one you looked at last week. */
  var PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#b07aa1",
    "#76b7b2",
    "#edc948",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
  ];
  var EXTERNAL_COLOR = "#9aa0a6";
  var DIMMED = "#d8d8d8";

  function colorFor(node) {
    if (node.is_external || node.community < 0) return EXTERNAL_COLOR;
    return PALETTE[node.community % PALETTE.length];
  }

  fetch("/map/api")
    .then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    })
    .then(function (data) {
      var graph = new graphology.Graph({ type: "directed", multi: false });

      data.nodes.forEach(function (node) {
        graph.addNode(node.id, {
          /* External modules are prefixed rather than given their own shape: sigma's
             default build ships one node program, and adding another to say "outside"
             is more machinery than the distinction needs. */
          label: node.is_external ? "↗ " + node.label : node.label,
          x: node.x,
          y: node.y,
          size: node.size,
          color: colorFor(node),
          community: node.community,
          external: node.is_external,
          project: node.project,
        });
      });

      data.edges.forEach(function (edge) {
        if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) return;
        if (graph.hasEdge(edge.source, edge.target)) return;
        graph.addEdge(edge.source, edge.target, {
          /* Thickness tracks the stored weight (ADR-0017), log-compressed so one heavy
             edge does not render every other dependency as a hairline. */
          size: Math.min(6, 0.4 + Math.log1p(edge.weight) * 1.4),
          color: edge.crosses_community ? "rgba(192,87,74,0.35)" : "rgba(127,127,127,0.25)",
        });
      });

      var hovered = null;

      var renderer = new Sigma(graph, container, {
        renderEdgeLabels: false,
        labelDensity: 0.6,
        labelGridCellSize: 70,
        labelRenderedSizeThreshold: 5,
        nodeReducer: function (key, attrs) {
          if (!hovered || key === hovered || graph.areNeighbors(key, hovered)) return attrs;
          return Object.assign({}, attrs, { color: DIMMED, label: "" });
        },
        edgeReducer: function (key, attrs) {
          if (!hovered || graph.hasExtremity(key, hovered)) return attrs;
          return Object.assign({}, attrs, { hidden: true });
        },
      });

      status.textContent = graph.order + " modules, " + graph.size + " dependencies. Scroll to zoom, drag to pan.";

      /* Hovering dims everything the node does not touch — the only way to read one
         module's reach out of a picture this dense. */
      renderer.on("enterNode", function (event) {
        hovered = event.node;
        renderer.refresh();
      });
      renderer.on("leaveNode", function () {
        hovered = null;
        renderer.refresh();
      });
      renderer.on("clickNode", function (event) {
        var attrs = graph.getNodeAttributes(event.node);
        status.textContent = attrs.external
          ? event.node + " — external, owned by project '" + attrs.project + "'"
          : event.node + " — " + graph.degree(event.node) + " dependencies";
      });

      /* Focus a subsystem without reloading the page. The camera works in framed-graph
         coordinates, not the graph's own, so the centroid has to be read back from
         sigma's display data rather than computed from the x/y we supplied. */
      document.querySelectorAll(".community-focus").forEach(function (button) {
        button.addEventListener("click", function () {
          var id = parseInt(button.dataset.community, 10);
          var members = graph.filterNodes(function (_key, attrs) {
            return attrs.community === id;
          });
          if (!members.length) return;

          var sum = members.reduce(
            function (acc, key) {
              var display = renderer.getNodeDisplayData(key);
              return display ? { x: acc.x + display.x, y: acc.y + display.y, n: acc.n + 1 } : acc;
            },
            { x: 0, y: 0, n: 0 },
          );
          if (!sum.n) return;

          renderer.getCamera().animate({ x: sum.x / sum.n, y: sum.y / sum.n, ratio: 0.35 }, { duration: 400 });
          status.textContent = members.length + " modules in this subsystem.";
        });
      });
    })
    .catch(function (error) {
      status.textContent = "Could not load the map: " + error.message;
    });
})();
