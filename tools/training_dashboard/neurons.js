"use strict";

(function () {
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const valueUnits = semantics => {
    if (semantics?.value_code === "expected_remaining_placements_times_0.001") return {
      multiplier: 1000, label: "placements", description: "Estimated placements (raw critic output × 1,000)."
    };
    if (semantics?.value_code === "discounted_hyperlayers") return {
      multiplier: 1, label: "discounted hyperlayers", description: "Estimated discounted future hyperlayer clears."
    };
    return {multiplier: 1, label: "critic units", description: "Raw critic output; value units are unspecified."};
  };
  const criticChange = (unit, semantics) => {
    const effect = unit.intervention || {};
    const units = valueUnits(semantics);
    if (units.multiplier === 1000 && finite(effect.absolute_value_change_in_placements)) return effect.absolute_value_change_in_placements;
    return finite(effect.absolute_value_change) ? effect.absolute_value_change * units.multiplier : null;
  };
  const metrics = {
    mean: {label: "Mean activation · bounded units only", maximum: 1, percent: false,
      value: (unit, group) => group && group.sigmoid === true ? unit.activation?.mean : null},
    saturation: {label: "Fraction near sigmoid limits (0 or 1)", maximum: 1, percent: true,
      value: (unit, group) => group && group.sigmoid === true ? unit.activation?.saturated_fraction : null},
    choice: {label: "Fraction of sampled states with a different greedy choice", maximum: 1, percent: true,
      value: unit => unit.intervention?.top1_changed_fraction},
    tv: {label: "Move-choice probability total variation · 0–1", maximum: 1, percent: false,
      value: unit => unit.intervention?.probability_total_variation},
    critic: {label: "Mean absolute critic change", maximum: null, percent: false,
      value: (unit, group, semantics) => criticChange(unit, semantics)}
  };
  function metricValue(unit, group, metric, semantics) {
    const value = (metrics[metric] || metrics.mean).value(unit, group, semantics);
    return finite(value) ? value : null;
  }
  function metricScale(units, groups, metric, semantics) {
    const specification = metrics[metric] || metrics.mean;
    const measured = units.map(unit => metricValue(unit, groups.get(unit.group), metric, semantics)).filter(finite);
    return {minimum: 0, maximum: specification.maximum || Math.max(0, ...measured) || 1};
  }
  function colorFor(value, maximum) {
    if (!finite(value)) return null;
    const ratio = Math.max(0, Math.min(1, value / (maximum || 1)));
    const start = [36, 50, 58], end = [128, 201, 188];
    return `rgb(${start.map((channel, index) => Math.round(channel + (end[index] - channel) * ratio)).join(",")})`;
  }
  function activationRange(activation, bounded, multiplier = 1) {
    if (!activation || ![activation.minimum, activation.maximum, activation.mean, activation.std].every(finite)) return null;
    const minimum = activation.minimum * multiplier, maximum = activation.maximum * multiplier;
    const mean = activation.mean * multiplier, std = activation.std * multiplier;
    let lower = bounded ? 0 : Math.min(0, minimum), upper = bounded ? 1 : Math.max(0, maximum);
    if (upper <= lower) { lower -= 1; upper += 1; }
    return {minimum, maximum, mean, std, lower, upper,
      bandMinimum: Math.max(minimum, mean - std), bandMaximum: Math.min(maximum, mean + std)};
  }
  function matchingUnits(units, group, query) {
    const needle = String(query || "").trim().toLowerCase();
    return units.filter(unit => (!group || unit.group === group) &&
      (!needle || [unit.id, unit.group, unit.role].some(value => String(value || "").toLowerCase().includes(needle))));
  }
  // Pure helpers also run under Node for small data-contract checks.
  if (typeof module !== "undefined" && module.exports) module.exports = {metricValue, metricScale, criticChange, valueUnits, colorFor, activationRange, matchingUnits};
  if (typeof document === "undefined") return;

  const $ = id => document.getElementById(id);
  const make = (tag, className, content) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content !== undefined) element.textContent = String(content);
    return element;
  };
  const number = (value, digits = 3) => {
    if (!finite(value)) return "—";
    if (value !== 0 && (Math.abs(value) < 0.0001 || Math.abs(value) >= 1000000)) return value.toExponential(2);
    return value.toLocaleString(undefined, {maximumFractionDigits: digits});
  };
  const percent = value => finite(value) ? `${number(value * 100, 2)}%` : "—";
  const metricText = (value, metric) => metrics[metric].percent ? percent(value) : number(value, 4);
  let report = null, groupMap = new Map(), unitMap = new Map(), selectedId = null, visible = [], loading = false;

  function layerButton(group) {
    const button = make("button", "layer-button");
    button.type = "button";
    button.append(make("strong", "", group.id), make("small", "", `${number(group.units, 0)} units${group.sigmoid ? " · sigmoid" : " · linear"}`));
    button.setAttribute("aria-label", `Show ${group.units} units in ${group.id}`);
    button.addEventListener("click", () => {
      $("group-select").value = group.id;
      $("unit-search").value = "";
      renderInventory();
      $("inventory").scrollIntoView({behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"});
      $("group-select").focus({preventScroll: true});
    });
    return button;
  }
  function renderFlow(id, groups) {
    const container = $(id);
    container.replaceChildren();
    groups.forEach((group, index) => {
      if (index) { const arrow = make("span", "layer-arrow", "→"); arrow.setAttribute("aria-hidden", "true"); container.append(arrow); }
      container.append(layerButton(group));
    });
  }
  function renderArchitecture() {
    const input = report.input_schema || {};
    $("architecture-input").textContent = `${input.board_channels?.length ?? "—"} board channels · ${input.context?.length ?? "—"} context features · ${input.candidate_features?.length ?? "—"} candidate features`;
    renderFlow("encoder-flow", report.groups.filter(group => group.spatial));
    renderFlow("actor-flow", report.groups.filter(group => ["rank1", "rank2", "score"].includes(group.id)));
    renderFlow("critic-flow", report.groups.filter(group => ["value_hidden", "value"].includes(group.id)));
    $("critic-formula").textContent = report.semantics?.value || valueUnits(report.semantics).description;
    renderFlow("completion-flow", report.groups.filter(group => ["completion_hidden", "completion"].includes(group.id)));
    $("policy-formula").textContent = report.policy?.final_score_formula || "Policy score formula unavailable.";
  }
  function appendDefinition(container, title, explanation) {
    container.append(make("dt", "", title), make("dd", "", explanation || "Not recorded."));
  }
  function renderMethod() {
    const definitions = $("method-details");
    definitions.replaceChildren();
    appendDefinition(definitions, "Sampling", `${report.sampling?.method || "Not recorded"}. Strongest contexts are the three sampled states with the largest mean activation, not detected concepts.`);
    appendDefinition(definitions, "Activation summaries", `${report.sampling?.activation_weighting || "Not recorded"}. The band shows mean ± one standard deviation, clipped to the observed range. It is not a histogram or a confidence interval.`);
    appendDefinition(definitions, "Intervention", `Replace one unit across ${report.intervention?.scope || "applicable sites"}. Fixed sigmoid baseline ${number(report.intervention?.sigmoid_baseline)}; raw score/value baseline ${number(report.intervention?.raw_score_and_value_baseline)}. ${report.intervention?.baseline_meaning || ""}. ${report.intervention?.effect_averaging || ""}.`);
    appendDefinition(definitions, "Critic units", report.semantics?.value);
    appendDefinition(definitions, "Completion probabilities", report.semantics?.completion);
    appendDefinition(definitions, "Move probabilities", report.policy?.selection_rule);
    appendDefinition(definitions, "Interpreting effects", report.intervention?.limitation);
    $("audit-notes").replaceChildren(...(report.notes || []).map(note => make("li", "", note)));
  }
  function renderInventory() {
    if (!report) return;
    const metric = $("metric-select").value;
    const scale = metricScale(report.units, groupMap, metric, report.semantics);
    const selection = $("group-select").value;
    visible = matchingUnits(report.units, selection, $("unit-search").value);
    $("visible-count").textContent = `${number(visible.length, 0)} / ${number(report.units.length, 0)} units shown`;
    $("metric-description").textContent = metrics[metric].label;
    $("scale-min").textContent = metricText(0, metric);
    $("scale-mid").textContent = metricText(scale.maximum / 2, metric);
    $("scale-max").textContent = metricText(scale.maximum, metric);
    const grid = $("unit-grid");
    grid.replaceChildren();
    for (const group of report.groups) {
      const units = visible.filter(unit => unit.group === group.id);
      if (!units.length) continue;
      const section = make("section", "unit-group");
      const heading = make("div", "unit-group-heading");
      const title = make("h3", "", group.id);
      title.id = `layer-${group.id.replace(/[^a-z0-9_-]/gi, "-")}`;
      heading.append(title, make("span", "", `${units.length} ${group.spatial ? "shared channels" : "units"}${group.sigmoid === false ? " · raw output" : ""}`));
      const cells = make("div", "unit-cells");
      cells.setAttribute("role", "group");
      cells.setAttribute("aria-labelledby", title.id);
      for (const unit of units) {
        const value = metricValue(unit, group, metric, report.semantics);
        const button = make("button", `neuron-cell${value === null ? " metric-missing" : ""}`, unit.index);
        button.type = "button";
        button.dataset.unit = unit.id;
        const color = colorFor(value, scale.maximum);
        if (color) {
          button.style.backgroundColor = color;
          if (value / scale.maximum > 0.55) button.style.color = "#09201f";
        }
        const description = value === null ? (metric === "mean" && group.sigmoid === false ? "unbounded raw output; inspect actual activation" : "measurement unavailable") : metricText(value, metric);
        button.setAttribute("aria-label", `${unit.id}, ${metrics[metric].label}: ${description}`);
        button.title = `${unit.id} · ${description}`;
        button.setAttribute("aria-pressed", unit.id === selectedId ? "true" : "false");
        cells.append(button);
      }
      section.append(heading, cells);
      grid.append(section);
    }
    $("no-matches").hidden = visible.length > 0;
    if (!visible.some(unit => unit.id === selectedId)) selectUnit(visible[0]?.id || null);
    else selectUnit(selectedId);
  }
  function detailSection(title) {
    const section = make("section", "detail-section");
    section.append(make("h3", "", title));
    return section;
  }
  function renderActivation(unit, group) {
    const section = detailSection("Observed activations");
    const multiplier = unit.group === "value" ? valueUnits(report.semantics).multiplier : 1;
    const range = activationRange(unit.activation, group?.sigmoid === true, multiplier);
    if (!range) { section.append(make("p", "detail-copy", "Activation measurements are unavailable.")); return section; }
    const chart = make("div", "range-chart");
    chart.setAttribute("role", "img");
    chart.setAttribute("aria-label", `Activation range ${number(range.minimum)} to ${number(range.maximum)}, mean ${number(range.mean)}, standard deviation ${number(range.std)}. Axis ${number(range.lower)} to ${number(range.upper)}.`);
    const position = value => Math.max(0, Math.min(100, (value - range.lower) / (range.upper - range.lower) * 100));
    const mark = (className, minimum, maximum) => {
      const element = make("span", className);
      element.style.left = `${position(minimum)}%`;
      if (maximum !== undefined) element.style.width = `${Math.max(0, position(maximum) - position(minimum))}%`;
      return element;
    };
    chart.append(mark("observed-range", range.minimum, range.maximum), mark("std-band", range.bandMinimum, range.bandMaximum), mark("mean-marker", range.mean));
    const ticks = make("div", "range-ticks");
    ticks.append(make("span", "", number(range.lower)), make("span", "", number(range.upper)));
    const statistics = make("dl", "activation-stats");
    for (const [label, value] of [["MIN", range.minimum], ["MEAN", range.mean], ["STD", range.std], ["MAX", range.maximum]]) {
      const pair = make("div"); pair.append(make("dt", "", label), make("dd", "", number(value, 4))); statistics.append(pair);
    }
    const units = unit.group === "value" ? valueUnits(report.semantics).description : group?.sigmoid ? "Sigmoid activation on a fixed 0–1 axis." : "Raw actor score, on its own numeric axis; not a probability.";
    section.append(chart, ticks, statistics, make("p", "range-caption", `${units} Line: observed min–max. Teal band: mean ± std, clipped to that range. Orange marker: mean. No distribution shape is inferred.`));
    const saturation = group?.sigmoid ? ` · ${percent(unit.activation?.saturated_fraction)} saturated` : "";
    section.append(make("p", "detail-copy", `${number(unit.activation?.observations, 0)} observations${saturation}. ${unit.spatial ? "Shared channel measured across spatial sites." : unit.group.startsWith("rank") || unit.group === "score" ? "Measured across valid candidates." : "One scalar observation per sampled state."}`));
    return section;
  }
  function effectRow(label, value, maximum, format, axisUnit) {
    const row = make("div", "effect-row");
    const labelRow = make("div", "effect-label");
    labelRow.append(make("span", "", label), make("strong", "", format(value)));
    row.append(labelRow);
    if (finite(value)) {
      const track = make("div", "effect-track");
      const fill = make("span"); fill.style.width = `${Math.max(0, Math.min(100, value / maximum * 100))}%`; track.append(fill);
      row.append(track, make("div", "effect-axis", `axis 0 → ${format(maximum)}${axisUnit ? ` ${axisUnit}` : ""}`));
    }
    return row;
  }
  function maximumEffect(getter) {
    return Math.max(0, ...report.units.map(getter).filter(finite)) || 1;
  }
  function renderEffects(unit) {
    const section = detailSection("Measured intervention effects");
    const effect = unit.intervention || {};
    const scoreMaximum = maximumEffect(item => item.intervention?.mean_absolute_score_change);
    const units = valueUnits(report.semantics);
    const criticMaximum = maximumEffect(item => criticChange(item, report.semantics));
    const rankMaximum = maximumEffect(item => item.intervention?.mean_absolute_rank_change);
    section.append(effectRow("Greedy choice changed", effect.top1_changed_fraction, 1, percent),
      effectRow("Move probability total variation", effect.probability_total_variation, 1, value => number(value, 5)),
      effectRow("Absolute critic change", criticChange(unit, report.semantics), criticMaximum, value => number(value, 2), units.label),
      effectRow("Absolute final-score change", effect.mean_absolute_score_change, scoreMaximum, value => number(value, 5), "score units"),
      effectRow("Absolute candidate-rank change", effect.mean_absolute_rank_change, rankMaximum, value => number(value, 3), "ranks"),
      effectRow("Absolute auxiliary probability change", effect.mean_absolute_completion_probability_change, 1, percent));
    section.append(make("p", "range-caption", "Bars have separately labeled axes. Critic, score and rank axes use the largest measured effect across this audit; probability axes cover their full bounds."));
    const signed = finite(effect.signed_value_change) ? effect.signed_value_change * units.multiplier : null;
    section.append(make("p", "detail-copy", `Greedy choice changed in ${number(effect.top1_changes, 0)} / ${number(report.samples.length, 0)} sampled states. Signed mean critic change: ${finite(signed) && signed > 0 ? "+" : ""}${number(signed, 2)} ${units.label}. A value shift is a model response, not measured survival gain.`));
    return section;
  }
  function renderContexts(unit) {
    const section = detailSection("Three strongest sampled contexts");
    section.append(make("p", "range-caption", "Ranked by mean activation within the sampled states, not by survival or semantic importance."));
    const units = valueUnits(report.semantics);
    for (const context of (unit.strongest_contexts || []).slice(0, 3)) {
      const sample = report.samples.find(item => item.id === context.sample);
      const card = make("article", "context-card");
      const top = make("div", "context-top");
      top.append(make("span", "", `State ${context.sample}${sample ? ` · ${sample.piece} → ${sample.next_piece}` : ""}`), make("span", "", `mean ${number(finite(context.mean_activation) ? context.mean_activation * (unit.group === "value" ? units.multiplier : 1) : null, 4)}${unit.group === "value" ? ` ${units.label}` : ""}`));
      card.append(top);
      if (sample) {
        const meta = make("p", "context-meta", `Seed ${sample.seed} · after ${number(sample.placements_before, 0)} placements · target ${sample.effective_target} · ${sample.candidates} candidates`);
        const hash = make("p", "context-meta", "Board SHA256 "); hash.append(make("code", "", sample.board_sha256 || "Not recorded"));
        card.append(meta, hash);
      } else card.append(make("p", "context-meta", "Sample metadata unavailable."));
      section.append(card);
    }
    return section;
  }
  function selectUnit(id) {
    selectedId = id;
    $("unit-grid").querySelectorAll("button[data-unit]").forEach(button => button.setAttribute("aria-pressed", button.dataset.unit === id ? "true" : "false"));
    const unit = unitMap.get(id);
    $("detail-body").replaceChildren();
    $("unit-title").textContent = unit?.id || "No matching unit";
    $("unit-role").textContent = unit?.role || "Adjust the filter to explore a unit.";
    if (!unit) return;
    const group = groupMap.get(unit.group);
    const computation = detailSection("Exact computation");
    computation.append(make("pre", "formula", unit.computation?.formula || "Not recorded"));
    const tensors = make("ul", "tensor-list");
    for (const row of unit.computation?.checkpoint_tensor_rows || []) tensors.append(make("li", "", row));
    computation.append(tensors, make("p", "detail-copy", unit.computation?.structural_role || ""));
    const explanation = detailSection("Audit explanation");
    explanation.append(make("p", "detail-copy", unit.explanation || "No explanation recorded."));
    $("detail-body").append(renderActivation(unit, group), renderEffects(unit), computation, renderContexts(unit), explanation);
  }
  function renderReport(data) {
    if (!data || data.schema !== "tetris4d-neuron-audit-v1" || !Array.isArray(data.units) || !data.units.length || !Array.isArray(data.groups) || !Array.isArray(data.samples)) throw new Error("This file is not a supported neuron audit.");
    report = data;
    metrics.critic.label = `Mean absolute critic change · ${valueUnits(report.semantics).label}`;
    $("metric-select").querySelector('option[value="critic"]').textContent = metrics.critic.label;
    groupMap = new Map(report.groups.map(group => [group.id, group]));
    unitMap = new Map(report.units.map(unit => [unit.id, unit]));
    selectedId = null;
    $("unit-count").textContent = number(report.units.length, 0);
    $("group-count").textContent = `${number(report.groups.length, 0)} audited layers · channels counted once`;
    $("sample-count").textContent = number(report.samples.length, 0);
    const recordings = report.recordings || [];
    $("recording-count").textContent = `${recordings.length} replay recordings · ${recordings.filter(recording => recording.verified).length} verified`;
    $("parity-result").textContent = report.baseline_forward_parity_passed === true ? "Passed" : report.baseline_forward_parity_passed === false ? "Failed" : "Unknown";
    $("parity-detail").textContent = `${report.environment?.device || "Unknown device"} · ${report.environment?.backend || "unknown backend"} · rtol ${number(report.parity_tolerance?.rtol)}`;
    $("parameter-count").textContent = number(report.checkpoint?.parameters, 0);
    $("checkpoint-sha").textContent = report.checkpoint?.sha256 || "Not recorded";
    $("group-select").replaceChildren(make("option", "", "All layers"));
    $("group-select").firstChild.value = "";
    for (const group of report.groups) { const option = make("option", "", `${group.id} (${group.units})`); option.value = group.id; $("group-select").append(option); }
    renderArchitecture(); renderMethod(); renderInventory();
    $("audit-content").hidden = false; $("audit-empty").hidden = true;
    $("audit-status").textContent = "Frozen audit loaded";
  }
  async function loadAudit() {
    if (loading) return;
    loading = true; $("retry-audit").disabled = true;
    $("audit-status").textContent = "Loading audit";
    try {
      const response = await fetch("/api/neuron-audit", {cache: "no-cache", headers: {Accept: "application/json"}});
      if (!response.ok) {
        let detail = null;
        try { detail = await response.json(); } catch (_) { /* Generic message below. */ }
        if (response.status === 404) {
          const error = new Error("No neuron audit is configured for this dashboard. Start it with --neuron-audit pointing to audit/neurons.json, then retry.");
          error.unconfigured = true; throw error;
        }
        throw new Error(typeof detail?.error === "string" ? detail.error : `The audit could not be loaded (HTTP ${response.status}).`);
      }
      renderReport(await response.json());
    } catch (error) {
      $("audit-content").hidden = true; $("audit-empty").hidden = false;
      $("empty-title").textContent = error.unconfigured ? "No audit configured" : "Could not load the audit";
      $("empty-copy").textContent = error.message || "An unexpected loading error occurred. Retry to fetch the audit again.";
      $("audit-status").textContent = error.unconfigured ? "Awaiting audit" : "Audit unavailable";
    } finally { loading = false; $("retry-audit").disabled = false; }
  }
  $("metric-select").addEventListener("change", renderInventory);
  $("group-select").addEventListener("change", renderInventory);
  $("unit-search").addEventListener("input", renderInventory);
  $("retry-audit").addEventListener("click", loadAudit);
  $("back-to-unit-map").addEventListener("click", () => {
    const selected = Array.from($("unit-grid").querySelectorAll("button[data-unit]")).find(button => button.dataset.unit === selectedId);
    if (selected) { selected.scrollIntoView({block: "center"}); selected.focus({preventScroll: true}); }
  });
  $("unit-grid").addEventListener("click", event => {
    const button = event.target.closest("button[data-unit]");
    if (button) {
      selectUnit(button.dataset.unit);
      if (matchMedia("(max-width: 720px)").matches) $("unit-detail").scrollIntoView({behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start"});
    }
  });
  $("unit-grid").addEventListener("keydown", event => {
    const current = event.target.closest("button[data-unit]");
    if (!current || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
    const buttons = Array.from($("unit-grid").querySelectorAll("button[data-unit]"));
    const index = buttons.indexOf(current);
    const row = Array.from(current.parentElement.children);
    const columns = Math.max(1, row.filter(button => button.offsetTop === row[0].offsetTop).length);
    let target = index;
    if (event.key === "ArrowLeft") target--;
    if (event.key === "ArrowRight") target++;
    if (event.key === "ArrowUp") target -= columns;
    if (event.key === "ArrowDown") target += columns;
    if (event.key === "Home") target = 0;
    if (event.key === "End") target = buttons.length - 1;
    event.preventDefault();
    const next = buttons[Math.max(0, Math.min(buttons.length - 1, target))];
    if (next) { next.focus(); selectUnit(next.dataset.unit); }
  });
  loadAudit();
})();
