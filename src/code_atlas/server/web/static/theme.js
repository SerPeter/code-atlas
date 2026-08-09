/**
 * Appearance persistence.
 *
 * The server keeps no preferences — there is no account to keep them on — so the choice
 * lives in this browser and is reapplied before paint on every page.
 */
(function () {
  "use strict";
  var root = document.documentElement;

  // A link on the settings page carries the choice; anywhere else, restore it.
  var url = new URL(location.href);
  var palette = url.searchParams.get("palette");
  var theme = url.searchParams.get("theme");
  if (palette) localStorage.setItem("atlas.palette", palette);
  if (theme) localStorage.setItem("atlas.theme", theme);

  apply();
  function apply() {
    root.dataset.palette = localStorage.getItem("atlas.palette") || "modernist";
    var t = localStorage.getItem("atlas.theme") || "light";
    // "auto" is not a ground; resolve it so the CSS only ever sees light or dark.
    root.dataset.theme =
      t === "auto" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : t;
  }

  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    if ((localStorage.getItem("atlas.theme") || "light") === "auto") apply();
  });
})();
