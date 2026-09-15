(function () {
  var prefix = "/interpretable-ml-trading";
  var path = window.location.pathname;
  var match = path.match(/^\/interpretable-ml-trading\/en(\/.*)?$/);
  if (!match) {
    return;
  }
  var rest = match[1] || "/";
  window.location.replace(prefix + rest + window.location.search + window.location.hash);
})();