// Shell behaviour shared by every page: theme axes and the projects dialog.
// The dialog markup is the v1.1 design's, rendered client-side so the row
// template exists in exactly one place and works from any page.

(function () {
  "use strict";

  // ── Theme ────────────────────────────────────────────────────────────────
  var media = window.matchMedia("(prefers-color-scheme: dark)");

  function applyTheme() {
    // A ?theme=/?palette= override wins for this view only — the boot script in
    // base.html applied it before first paint, and re-applying from storage here
    // would silently undo it.
    var q = new URLSearchParams(window.location.search);
    var stored = q.get("theme") || localStorage.getItem("atlas.theme") || "light";
    var ground = stored === "auto" ? (media.matches ? "dark" : "light") : stored;
    document.documentElement.dataset.theme = ground;
    document.documentElement.dataset.palette = q.get("palette") || localStorage.getItem("atlas.palette") || "modernist";
  }
  media.addEventListener("change", applyTheme);
  applyTheme();

  window.AtlasTheme = {
    set: function (key, value) {
      localStorage.setItem("atlas." + key, value);
      applyTheme();
    },
    get: function (key, fallback) {
      return localStorage.getItem("atlas." + key) || fallback;
    },
  };

  // ── Projects dialog ──────────────────────────────────────────────────────
  var STATE_CHIP = {
    fresh: { sbg: "transparent", sfg: "color-mix(in srgb, var(--color-text) 55%, transparent)", sbd: "1px solid transparent", label: "" },
    stale: { sbg: "var(--color-neutral-200)", sfg: "var(--color-neutral-800)", sbd: "1px solid transparent", label: "STALE" },
    unindexed: { sbg: "transparent", sfg: "var(--color-accent-700)", sbd: "1px dashed var(--color-accent)", label: "NOT INDEXED" },
  };

  var mount = document.getElementById("projects-modal");
  var chip = document.getElementById("project-chip");
  if (!mount || !chip) return;

  var picker = null;
  var picked = [];

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function flat(projects) {
    var out = [];
    projects.forEach(function (p) {
      out.push(p);
      (p.children || []).forEach(function (c) { out.push(c); });
    });
    return out;
  }

  function row(p, child) {
    var on = picked.indexOf(p.name) >= 0;
    var sc = STATE_CHIP[p.state] || STATE_CHIP.fresh;
    var pad = child ? "7px 12px 7px 36px" : "9px 12px";
    var nameStyle = child
      ? 'style="font-size:13.5px"'
      : 'style="font-family:var(--font-heading);font-weight:800;font-size:15px"';
    return (
      '<button data-pick="' + esc(p.name) + '" class="hv6" style="display:flex;align-items:center;gap:10px;width:100%;background:' +
      (on ? "color-mix(in srgb, var(--color-accent) 8%, transparent)" : "transparent") +
      ';border:0;border-radius:10px;padding:' + pad + ';cursor:' + (p.state === "unindexed" ? "default" : "pointer") + ';color:inherit;font:inherit;text-align:left">' +
      '<span style="width:14px;height:14px;flex:none;border-radius:5px;border:1px solid var(--color-divider);background:' +
      (on ? "var(--color-accent)" : "transparent") + ';display:block"></span>' +
      "<span " + nameStyle + ">" + esc(p.label || p.name) + "</span>" +
      '<span style="font-size:10px;letter-spacing:0.06em;border-radius:999px;padding:2px 8px;background:' + sc.sbg + ";color:" + sc.sfg + ";border:" + sc.sbd + '">' + sc.label + "</span>" +
      '<span style="margin-left:auto;font-size:12px;font-variant-numeric:tabular-nums;color:color-mix(in srgb, var(--color-text) 62%, transparent)">' +
      (p.state === "unindexed" ? "—" : p.entities.toLocaleString("en-US")) + "</span>" +
      '<span style="width:104px;text-align:right;font-size:11px;color:color-mix(in srgb, var(--color-text) 50%, transparent)">' +
      (p.state === "unindexed" ? "never indexed" : "indexed " + esc(p.indexed_ago || "")) + "</span>" +
      "</button>"
    );
  }

  function costNote(chosen, entities) {
    if (chosen.length <= 1) return "One project — " + entities.toLocaleString("en-US") + " entities on the map.";
    return "Combined map across " + chosen.length + " projects. Above 1,500 nodes the map truncates.";
  }

  function render() {
    var chosen = flat(picker.projects).filter(function (p) { return picked.indexOf(p.name) >= 0; });
    var entities = chosen.reduce(function (a, p) { return a + p.entities; }, 0);
    mount.innerHTML =
      '<div class="dialog-backdrop" style="align-items:start;padding-top:64px;z-index:40">' +
      '<div class="dialog" style="width:min(620px,100%);max-height:78vh;gap:0;padding:0;background:var(--color-bg);border:1px solid var(--color-divider);border-radius:14px;overflow:hidden">' +
      '<div style="display:flex;align-items:baseline;padding:16px 18px 14px;border-bottom:1px solid var(--color-divider)">' +
      "<div>" +
      '<div class="dialog-title" style="font-size:20px">Projects</div>' +
      '<div style="font-size:11.5px;color:color-mix(in srgb, var(--color-text) 60%, transparent);margin-top:2px">Select one or several. Cross-project dependencies appear when more than one is loaded.</div>' +
      "</div>" +
      '<button data-close class="btn btn-secondary" style="margin-left:auto;height:28px;font-size:12px">Close</button>' +
      "</div>" +
      '<div style="overflow-y:auto;flex:1;padding:8px">' +
      picker.projects
        .map(function (p) {
          return "<div>" + row(p, false) + (p.children || []).map(function (c) { return row(c, true); }).join("") + "</div>";
        })
        .join("") +
      "</div>" +
      '<div style="display:flex;align-items:center;gap:14px;padding:13px 18px;border-top:1px solid var(--color-divider)">' +
      '<div style="font-size:12px">' +
      "<div><strong>" + chosen.length + "</strong> selected · " + entities.toLocaleString("en-US") + " entities combined</div>" +
      '<div style="font-size:11px;color:color-mix(in srgb, var(--color-text) 60%, transparent);margin-top:2px">' + costNote(chosen, entities) + "</div>" +
      "</div>" +
      '<button data-load class="btn btn-primary" style="margin-left:auto"' + (chosen.length ? "" : " disabled") + ">Load map</button>" +
      "</div></div></div>";
    mount.hidden = false;
  }

  function open() {
    if (picker) {
      render();
      return;
    }
    fetch("/api/projects")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        picker = data;
        picked = (data.selected || []).slice();
        render();
      });
  }

  chip.addEventListener("click", open);
  mount.addEventListener("click", function (e) {
    var pick = e.target.closest("[data-pick]");
    if (pick) {
      var name = pick.getAttribute("data-pick");
      var all = flat(picker.projects);
      var found = all.filter(function (p) { return p.name === name; })[0];
      if (found && found.state !== "unindexed") {
        var at = picked.indexOf(name);
        if (at >= 0) picked.splice(at, 1);
        else picked.push(name);
        render();
      }
      return;
    }
    if (e.target.closest("[data-load]")) {
      document.cookie = "atlas_projects=" + encodeURIComponent(picked.join(",")) + ";path=/;max-age=31536000";
      window.location.href = "/";
      return;
    }
    if (e.target.closest("[data-close]") || e.target.classList.contains("dialog-backdrop")) {
      mount.hidden = true;
      mount.innerHTML = "";
    }
  });
})();
