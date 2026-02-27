
    const panelsEl = document.getElementById("panels");
    const statsEl = document.getElementById("stats");
    const drawer = document.getElementById("drawer");
    const drawerList = document.getElementById("drawer-list");
    const drawerDetail = document.getElementById("drawer-detail");
    const drawerSource = document.getElementById("drawer-source");
    const drawerDivider = document.getElementById("drawer-divider");
    const sourcePanel = document.getElementById("source-panel");
    const sourceContent = document.getElementById("source-content");
    const routingSummaryEl = document.getElementById("routing-summary");
    const structDistributionEl = document.getElementById("struct-distribution");
    const filterFragmentationEl = document.getElementById("filter-fragmentation");
    const filterTemplateEl = document.getElementById("filter-template");
    const filterPolicyEl = document.getElementById("filter-policy");
    const filterOriginEl = document.getElementById("filter-origin");
    let currentPage = null;
    let currentCitationList = [];
    let currentCitationIndex = -1;
      let currentSourceResults = [];
      let currentSourceIndex = -1;
      let currentSourceMeta = null;
    let demoCache = null;
    let currentViewMode = "routed";
    // Resizable list/detail within drawer (vertical split)
    let isResizing = false;
    let startY = 0;
    let startListHeight = 0;
    let startDetailHeight = 0;
    const minHeight = 80;

    const onMouseMove = (e) => {
      if (!isResizing) return;
      const dy = e.clientY - startY;
      const newListH = Math.max(minHeight, startListHeight + dy);
      const newDetailH = Math.max(minHeight, startDetailHeight - dy);
      drawerList.style.height = `${newListH}px`;
      drawerDetail.style.height = `${newDetailH}px`;
    };
    const onMouseUp = () => {
      if (!isResizing) return;
      isResizing = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };
    drawerDivider.addEventListener("mousedown", (e) => {
      isResizing = true;
      startY = e.clientY;
      startListHeight = drawerList.offsetHeight;
      startDetailHeight = drawerDetail.offsetHeight;
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    });

    const escapeHtml = (s) => (s || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

    const markTerm = (text) => {
      const safe = escapeHtml(text || "");
      return safe.replace(/&lt;&lt;(.+?)&gt;&gt;/g, "<mark>$1</mark>");
    };

    const structuralOf = (r) => (r && r.structural_intelligence) ? r.structural_intelligence : null;
    const routingOf = (r) => {
      const s = structuralOf(r);
      return (s && s.routing) ? s.routing : null;
    };
    const profileOf = (r) => {
      const s = structuralOf(r);
      return (s && s.profile) ? s.profile : null;
    };

    const policyClass = (policy) => {
      const p = (policy || "").toUpperCase();
      if (p === "INTERNAL") return "internal";
      if (p === "HYBRID") return "hybrid";
      if (p === "FALLBACK") return "fallback";
      if (p === "STRONG_RERANK") return "strong_rerank";
      return "internal";
    };

    const scoreForMode = (r, mode) => {
      const rt = routingOf(r) || {};
      const internal = Number(rt.internal_score ?? r.evidence_weight ?? 0);
      const oracc = Number(rt.oracc_score ?? internal);
      const policy = (rt.selected_policy || "INTERNAL").toUpperCase();
      if (mode === "internal") {
        const roleBoost = r.source_role === "primary_text" ? 0.15 : (r.source_role === "archival_wrapper" ? 0.05 : -0.05);
        return internal + roleBoost;
      }
      if (mode === "oracc") {
        const roleBoost = r.source_role === "scholarly_commentary" ? 0.12 : (r.source_role === "archival_wrapper" ? 0.03 : 0.0);
        return oracc + roleBoost;
      }
      const policyBoost = policy === "HYBRID" ? 0.08 : (policy === "FALLBACK" ? 0.05 : (policy === "STRONG_RERANK" ? 0.04 : 0.0));
      return internal + policyBoost;
    };

    const applyViewMode = (results, mode) => {
      const arr = [...(results || [])];
      arr.sort((a, b) => scoreForMode(b, mode) - scoreForMode(a, mode));
      if (mode === "oracc") {
        const byCommentary = arr.filter(r => r.source_role === "scholarly_commentary");
        const others = arr.filter(r => r.source_role !== "scholarly_commentary");
        return [...byCommentary, ...others];
      }
      return arr;
    };

    const applyStructuralFilters = (results) => {
      const frag = (filterFragmentationEl?.value || "").trim();
      const tpl = (filterTemplateEl?.value || "").trim();
      const pol = (filterPolicyEl?.value || "").trim().toUpperCase();
      const origin = (filterOriginEl?.value || "").trim();
      return (results || []).filter((r) => {
        const p = profileOf(r) || {};
        const rt = routingOf(r) || {};
        if (frag && (p.fragmentation || "") !== frag) return false;
        if (tpl && (p.template_type || "") !== tpl) return false;
        if (pol && (rt.selected_policy || "").toUpperCase() !== pol) return false;
        if (origin && (r.source_role || "") !== origin) return false;
        return true;
      });
    };

    const selectionDelta = (base, current) => {
      const baseIds = new Set((base || []).slice(0, 3).map(x => x.page_id));
      const curIds = new Set((current || []).slice(0, 3).map(x => x.page_id));
      let changed = 0;
      curIds.forEach((id) => { if (!baseIds.has(id)) changed += 1; });
      return changed;
    };

    const compareTopSelectionV2 = (base, current) => {
      const baseTop = (base || [])[0] || null;
      const curTop = (current || [])[0] || null;
      return {
        changed: !!(baseTop && curTop && String(baseTop.page_id || "") !== String(curTop.page_id || "")),
        baseTop: baseTop,
        currentTop: curTop,
        basePage: baseTop ? baseTop.page_id : "-",
        currentPage: curTop ? curTop.page_id : "-",
        baseOrigin: baseTop ? (baseTop.source_role || "unknown") : "unknown",
        currentOrigin: curTop ? (curTop.source_role || "unknown") : "unknown",
      };
    };

    const confidenceBandV2 = (routing) => {
      const rt = routing || {};
      const policy = String(rt.selected_policy || "INTERNAL").toUpperCase();
      const internal = Number(rt.internal_score ?? 0);
      const thr = rt.thresholds_applied || {};
      const low = Number(thr.internal_low ?? 0.5);
      const high = Number(thr.internal_high ?? 0.6);
      if (policy === "FALLBACK") return { level: "Low", cls: "low", note: "Fallback route engaged due to low-confidence match." };
      if (policy === "HYBRID" || policy === "STRONG_RERANK") return { level: "Medium", cls: "medium", note: "Hybrid routing used because the match was ambiguous." };
      if (internal >= high) return { level: "High", cls: "high", note: "Internal score cleared high threshold." };
      if (internal >= low) return { level: "Medium", cls: "medium", note: "Internal score in middle band." };
      return { level: "Low", cls: "low", note: "Internal score below low threshold." };
    };

    const structuralBadgesHtmlV2 = (r) => {
      const p = profileOf(r);
      if (!p) return "";
      const badges = [];
      if (p.numeric_density === "High") badges.push("Numeric-heavy");
      if (p.fragmentation === "Fragmentary") badges.push("Fragmentary");
      if (p.template_type === "Narrative") badges.push("Narrative");
      if (p.template_type === "Slot-Structured") badges.push("Slot-structured");
      if (p.domain_intent === "Legal" || p.domain_intent === "Administrative") badges.push("Institutional");
      if (!badges.length) return "";
      return `<div class="struct-badges">${badges.map(b => `<span class="struct-badge">${b}</span>`).join("")}</div>`;
    };

    const methodExplanationV2 = (r) => {
      const si = structuralOf(r);
      if (!si || !si.method_explanation) return "Method explanation unavailable.";
      return si.method_explanation;
    };

    const routingTraceHtmlV2 = (si) => {
      if (!si) return "";
      const p = si.profile || {};
      const rf = si.raw_features || {};
      const rt = si.routing || {};
      const conf = confidenceBandV2(rt);
      return `
        <details class="trace-details">
          <summary>View Routing Trace</summary>
          <ul class="trace-list">
            <li>Fragmentation: ${escapeHtml(p.fragmentation || "Unknown")}</li>
            <li>Template: ${escapeHtml(p.template_type || "Unknown")}</li>
            <li>Numeric density: ${escapeHtml(p.numeric_density || "Unknown")}</li>
            <li>Internal score: ${Number(rt.internal_score ?? 0).toFixed(2)}</li>
            <li>ORACC score: ${Number(rt.oracc_score ?? 0).toFixed(2)}</li>
            <li>Confidence: ${conf.level}</li>
            <li>Raw features: bracket_ratio=${Number(rf.bracket_ratio ?? 0).toFixed(4)}, digit_ratio=${Number(rf.digit_ratio ?? 0).toFixed(4)}</li>
            <li>Rationale: ${escapeHtml(rt.rationale || "No rationale available.")}</li>
          </ul>
        </details>
      `;
    };

    const renderStructuralHeaderV2 = (si, opts = {}) => {
      if (!si) {
        return "<div class='struct-header'><div class='meta'>Structural profile unavailable.</div></div>";
      }
      const p = si.profile || {};
      const r = si.routing || {};
      const policy = (r.selected_policy || "INTERNAL").toUpperCase();
      const thresholds = r.thresholds_applied || {};
      const internalLow = (thresholds.internal_low ?? 0.5).toFixed(2);
      const internalHigh = (thresholds.internal_high ?? 0.6).toFixed(2);
      const conf = confidenceBandV2(r);
      const compact = opts.compact === true;
      return `
        <div class="struct-header">
          <div class="struct-title">Structural Profile
            <button class="help" type="button" data-tip-title="Structural profile" data-tip="Auto-labeled text structure used for routing decisions: fragmentation, formula density, numeric density, template type, length bucket, and domain intent.">?</button>
          </div>
          <div class="struct-grid">
            <div class="struct-item"><strong>Fragmentation:</strong> ${escapeHtml(p.fragmentation || "Unknown")}</div>
            <div class="struct-item"><strong>Formula Density:</strong> ${escapeHtml(p.formula_density || "Unknown")}</div>
            <div class="struct-item"><strong>Numeric Density:</strong> ${escapeHtml(p.numeric_density || "Unknown")}</div>
            <div class="struct-item"><strong>Template Type:</strong> ${escapeHtml(p.template_type || "Unknown")}</div>
            <div class="struct-item"><strong>Length Bucket:</strong> ${escapeHtml(p.length_bucket || "Unknown")}</div>
            <div class="struct-item"><strong>Domain Intent:</strong> ${escapeHtml(p.domain_intent || "Unknown")}</div>
          </div>
          <div class="routing-box">
            <div class="struct-title" style="font-size:13px;margin-bottom:4px;">Routing Decision
              <span class="policy-chip ${policyClass(policy)}">${escapeHtml(policy)}</span>
              <span class="confidence-pill ${conf.cls}">${conf.level} confidence</span>
              <button class="help" type="button" data-tip-title="Method explanation" data-tip="${escapeHtml(si.method_explanation || "No explanation available.")}" aria-label="Method explanation">?</button>
            </div>
            <div class="routing-line"><strong>Internal Score:</strong> ${Number(r.internal_score ?? 0).toFixed(2)} | <strong>ORACC Score:</strong> ${Number(r.oracc_score ?? 0).toFixed(2)}</div>
            <div class="routing-line"><strong>Thresholds Applied:</strong> internal_low=${internalLow}, internal_high=${internalHigh}</div>
            <div class="routing-line"><strong>Rationale:</strong> ${escapeHtml(r.rationale || "No rationale available.")}</div>
          </div>
          ${compact ? "" : routingTraceHtmlV2(si)}
        </div>
      `;
    };

    const renderDecisionPanelV2 = (sectionLabel, currentResults, internalResults, mode) => {
      const topCurrent = (currentResults || [])[0] || null;
      if (!topCurrent) {
        return `<div class="decision-panel"><div class="meta">No results after filtering.</div></div>`;
      }
      const si = structuralOf(topCurrent) || {};
      const profile = si.profile || {};
      const routing = si.routing || {};
      const conf = confidenceBandV2(routing);
      const topCompare = compareTopSelectionV2(internalResults || [], currentResults || []);
      const changedCls = topCompare.changed ? "changed" : "";
      return `
        <div class="method-sequence">
          <div class="meta"><strong>Method sequence:</strong> Artifact analyzed -> Structural context identified -> Retrieval policy selected -> Candidate pool scored -> Evidence traced</div>
        </div>
        <div class="decision-panel">
          <div class="decision-title">Routing Decision for ${escapeHtml(sectionLabel)}
            <span class="policy-chip ${policyClass(routing.selected_policy || "INTERNAL")}">${escapeHtml(String(routing.selected_policy || "INTERNAL").toUpperCase())}</span>
            <span class="confidence-pill ${conf.cls}">${conf.level}</span>
          </div>
          <div class="selection-row">
            <span class="selection-pill ${changedCls}">Selection Change: ${topCompare.changed ? "YES" : "NO"}</span>
            <span class="selection-pill">Mode: ${escapeHtml(String(mode || "routed").toUpperCase())}</span>
            <span class="selection-pill">Baseline page: ${escapeHtml(String(topCompare.basePage || "-"))}</span>
            <span class="selection-pill">Current page: ${escapeHtml(String(topCompare.currentPage || "-"))}</span>
          </div>
          <div class="decision-grid">
            <div class="decision-cell"><strong>Internal score:</strong> ${Number(routing.internal_score ?? 0).toFixed(2)}</div>
            <div class="decision-cell"><strong>ORACC score:</strong> ${Number(routing.oracc_score ?? 0).toFixed(2)}</div>
            <div class="decision-cell"><strong>Template:</strong> ${escapeHtml(profile.template_type || "Unknown")}</div>
            <div class="decision-cell"><strong>Fragmentation:</strong> ${escapeHtml(profile.fragmentation || "Unknown")}</div>
            <div class="decision-cell"><strong>Origin shift:</strong> ${escapeHtml(topCompare.baseOrigin)} -> ${escapeHtml(topCompare.currentOrigin)}</div>
            <div class="decision-cell"><strong>Reason:</strong> ${escapeHtml(methodExplanationV2(topCurrent))}</div>
          </div>
          ${routingTraceHtmlV2(si)}
        </div>
      `;
    };

    const methodExplanation = (r) => {
      const si = structuralOf(r);
      if (!si || !si.method_explanation) return "Method explanation unavailable.";
      return si.method_explanation;
    };

    const structuralBadgesHtml = (r) => {
      const p = profileOf(r);
      if (!p) return "";
      const badges = [];
      if (p.numeric_density === "High") badges.push("🔢 Numeric-heavy");
      if (p.fragmentation === "Fragmentary") badges.push("🧱 Fragmentary");
      if (p.template_type === "Narrative") badges.push("📜 Narrative");
      if (p.template_type === "Slot-Structured") badges.push("🧾 Slot-structured");
      if (p.domain_intent === "Legal" || p.domain_intent === "Administrative") badges.push("🏛 Institutional");
      if (!badges.length) return "";
      return `<div class="struct-badges">${badges.map(b => `<span class="struct-badge">${b}</span>`).join("")}</div>`;
    };

    const renderStructuralHeader = (si) => {
      if (!si) {
        return "<div class='struct-header'><div class='meta'>Structural profile unavailable.</div></div>";
      }
      const p = si.profile || {};
      const r = si.routing || {};
      const policy = (r.selected_policy || "INTERNAL").toUpperCase();
      const thresholds = r.thresholds_applied || {};
      const internalLow = (thresholds.internal_low ?? 0.5).toFixed(2);
      const internalHigh = (thresholds.internal_high ?? 0.6).toFixed(2);
      return `
        <div class="struct-header">
          <div class="struct-title">🧠 Structural Profile</div>
          <div class="struct-grid">
            <div class="struct-item"><strong>Fragmentation:</strong> ${escapeHtml(p.fragmentation || "Unknown")}</div>
            <div class="struct-item"><strong>Formula Density:</strong> ${escapeHtml(p.formula_density || "Unknown")}</div>
            <div class="struct-item"><strong>Numeric Density:</strong> ${escapeHtml(p.numeric_density || "Unknown")}</div>
            <div class="struct-item"><strong>Template Type:</strong> ${escapeHtml(p.template_type || "Unknown")}</div>
            <div class="struct-item"><strong>Length Bucket:</strong> ${escapeHtml(p.length_bucket || "Unknown")}</div>
            <div class="struct-item"><strong>Domain Intent:</strong> ${escapeHtml(p.domain_intent || "Unknown")}</div>
          </div>
          <div class="routing-box">
            <div class="struct-title" style="font-size:13px;margin-bottom:4px;">🔀 Routing Decision
              <span class="policy-chip ${policyClass(policy)}">${escapeHtml(policy)}</span>
              <button class="help" type="button" data-tip-title="Method explanation" data-tip="${escapeHtml(si.method_explanation || "No explanation available.")}" aria-label="Method explanation">?</button>
            </div>
            <div class="routing-line"><strong>Internal Score:</strong> ${Number(r.internal_score ?? 0).toFixed(2)} · <strong>ORACC Score:</strong> ${Number(r.oracc_score ?? 0).toFixed(2)}</div>
            <div class="routing-line"><strong>Thresholds Applied:</strong> internal_low=${internalLow}, internal_high=${internalHigh}</div>
            <div class="routing-line"><strong>Rationale:</strong> ${escapeHtml(r.rationale || "No rationale available.")}</div>
          </div>
        </div>
      `;
    };

    const updateRoutingAnalytics = (groups) => {
      const counts = { INTERNAL: 0, HYBRID: 0, FALLBACK: 0, STRONG_RERANK: 0 };
      const dist = { Fragmentary: 0, "Slot-Structured": 0, "Numeric-heavy": 0, Narrative: 0, total: 0 };
      (groups || []).flat().forEach((r) => {
        const rt = routingOf(r);
        const policy = ((rt && rt.selected_policy) || "INTERNAL").toUpperCase();
        counts[policy] = (counts[policy] || 0) + 1;
        const pf = profileOf(r) || {};
        if (pf.fragmentation === "Fragmentary") dist.Fragmentary += 1;
        if (pf.template_type === "Slot-Structured") dist["Slot-Structured"] += 1;
        if (pf.numeric_density === "High") dist["Numeric-heavy"] += 1;
        if (pf.template_type === "Narrative") dist.Narrative += 1;
        dist.total += 1;
      });
      const pill = (label, count) => `<span class="route-pill">${label}: ${count}</span>`;
      routingSummaryEl.innerHTML = `
        <div class="route-pills">
          ${pill("INTERNAL", counts.INTERNAL || 0)}
          ${pill("HYBRID", counts.HYBRID || 0)}
          ${pill("FALLBACK", counts.FALLBACK || 0)}
          ${pill("STRONG_RERANK", counts.STRONG_RERANK || 0)}
        </div>
      `;
      if (!dist.total) {
        structDistributionEl.innerHTML = "No structural distribution available.";
        return;
      }
      const pct = (n) => `${((100 * n) / dist.total).toFixed(1)}%`;
      structDistributionEl.innerHTML = `
        <div><strong>Structural Distribution</strong> · Fragmentary ${pct(dist.Fragmentary)} · Slot-Structured ${pct(dist["Slot-Structured"])} · Numeric-heavy ${pct(dist["Numeric-heavy"])} · Narrative ${pct(dist.Narrative)}</div>
      `;
    };

    const humanDoc = (dt) => {
      if (dt === "legal") return "Legal text";
      if (dt === "letter") return "Letter";
      if (dt === "commentary") return "Study / commentary";
      if (dt === "index" || dt === "bibliography" || dt === "front_matter") return "Reference";
      return "Unknown";
    };

    const sourceBadge = (r) => {
      if (r.source_role === "primary_text") return "PRIMARY TEXT";
      if (r.source_role === "archival_wrapper") return "ARCHIVAL WRAPPER";
      if (r.source_role === "scholarly_commentary") return "SCHOLARLY COMMENTARY";
      return "SOURCE";
    };
    const docTypeLabel = (dt) => {
      if (dt === "legal") return "Legal document";
      if (dt === "letter") return "Letter (correspondence)";
      if (dt === "commentary") return "Study / commentary";
      if (dt === "index" || dt === "bibliography" || dt === "front_matter") return "Reference";
      return "Unknown";
    };
    const sourceTypeLabel = (role) => {
      if (role === "primary_text") return { text: "Primary text", cls: "source-primary" };
      if (role === "archival_wrapper") return { text: "Archival wrapper", cls: "source-wrapper" };
      if (role === "scholarly_commentary") return { text: "Scholarly commentary", cls: "source-commentary" };
      return { text: "Scholarly commentary", cls: "source-commentary" };
    };

    const sourceRoleHeaderLabel = (role) => {
      if (role === "primary_text") return "Primary Text";
      if (role === "archival_wrapper") return "Archival Wrapper";
      return "Scholarly Commentary";
    };

    const sourceRoleSecondaryTag = (role, docType) => {
      if (role === "primary_text") {
        if (docType === "legal") return "Legal document";
        if (docType === "letter") return "Letter";
        return docTypeLabel(docType);
      }
      if (docType === "legal") return "Contains legal-language excerpts";
      if (docType === "letter") return "Contains letter excerpts";
      if (docType === "commentary") return "Scholarly analysis";
      if (docType === "index" || docType === "bibliography" || docType === "front_matter") return "Reference material";
      return "Contains cited excerpts";
    };

    const isTranslationAllowed = (source) => {
      return source && source.source_role === "primary_text";
    };

    const translationReplacementLabel = (role) => {
      if (role === "archival_wrapper") return "View metadata / seals / witnesses";
      if (role === "scholarly_commentary") return "Why scholars link these texts";
      return "Translate this text";
    };

    const metaLine = (r) => {
      const lines = [];
      if (r.divine_names && r.divine_names.length) lines.push(`Mentions: ${r.divine_names.slice(0,3).join(", ")}`);
      if (r.formula_markers && r.formula_markers.length) lines.push(`Formula: ${r.formula_markers.slice(0,3).join(", ")}`);
      if (r.institutions && r.institutions.length) lines.push(`Institution: ${r.institutions.slice(0,3).join(", ")}`);
      if (!lines.length && r.topics && r.topics.length) lines.push(`Topics: ${r.topics.join(", ")}`);
      return lines.join(" · ");
    };

    const trustLine = (r) => {
      const sigs = [];
      if (r.citations && r.citations.length) sigs.push("citations");
      if (r.formula_markers && r.formula_markers.length) sigs.push("formula");
      if (r.institutions && r.institutions.length) sigs.push("institution");
      return `Sources: ${r.citations ? r.citations.length : 0} citation(s) · Signals: ${sigs.join(" + ") || "none"}`;
    };

    let activeChip = null;

    function renderKeywordChips(summary, onChange) {
      const wrap = document.createElement("div");
      wrap.className = "chips";
      const keywords = (summary && summary.keywords) ? summary.keywords : [];
      if (!keywords.length) return wrap;

      keywords.forEach((kw) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = kw;
        chip.onclick = () => {
          activeChip = (activeChip === kw) ? null : kw;
          [...wrap.querySelectorAll(".chip")].forEach(el => {
            el.classList.toggle("active", el.textContent === activeChip);
          });
          if (typeof onChange === "function") onChange(activeChip);
        };
        wrap.appendChild(chip);
      });
      return wrap;
    }

    function matchesChip(itemText, itemKind, chip) {
      if (!chip) return true;
      const t = (itemText || "").toLowerCase();
      const k = (itemKind || "").toLowerCase();
      const c = chip.toLowerCase();

      if (c === "joins") return k.includes("reconstruction") || t.includes("hülle") || t.includes("tafel") || t.includes("tablet") || t.includes("envelope") || t.includes("join");
      if (c === "legal") return k.includes("legal") || t.includes("plaintiff") || t.includes("evidence") || t.includes("witness") || t.includes("oath") || t.includes("trial");
      if (c === "chronology") return k.includes("chronology") || t.includes("chronolog") || t.includes("eponym") || t.includes("month") || t.includes("dated");
      if (c === "lexicon" || c === "lexical") return k.includes("lexical") || t.includes("cad");

      return t.includes(c);
    }

    function sourceOriginLabel(sourceRole) {
      if (sourceRole === "primary_text") return "Internal (Primary)";
      if (sourceRole === "archival_wrapper") return "Wrapper context";
      return "Commentary context";
    }

    function evidenceSignals(ev) {
      const src = ev?.source || {};
      const si = src.structural_intelligence || {};
      const profile = si.profile || {};
      const signals = [];
      signals.push({ label: "Section compatibility", ok: !!(profile.template_type && profile.template_type !== "Unknown") });
      signals.push({ label: "Fragment compatibility", ok: (profile.fragmentation || "Unknown") !== "Unknown" });
      signals.push({ label: "Digit pattern", ok: /\d/.test(ev?.quote || "") });
      signals.push({ label: "Formula overlap", ok: /\b(igi|oath|witness|li-?tu-?la|um-ma)\b/i.test(ev?.quote || "") });
      return signals;
    }

    function appendEvidenceDrivingSelection(summary, chip, sections) {
      if (!summary?.evidence?.length || !sections) return;
      const filtered = summary.evidence.filter(ev => matchesChip(ev.quote, ev.source?.doc_type, chip));
      if (!filtered.length) return;

      const selected = filtered[0];
      const selectedSource = selected.source || {};
      const selectedRole = selectedSource.source_role || "scholarly_commentary";
      const selectedContextOnly = selectedRole !== "primary_text";
      const trace = summary.selection_trace || {};
      const block = document.createElement("div");
      block.className = "evidence-driving";
      const selectedPolicy = ((selectedSource.structural_intelligence || {}).routing || {}).selected_policy || "INTERNAL";
      const selectedWeight = Number(selected.rank?.evidence_weight ?? trace.selected_weight ?? 0).toFixed(3);
      block.innerHTML = `
        <h4>Evidence Driving Selection</h4>
        <div class="meta">Selected candidate: ${escapeHtml(selectedSource.pdf_name || "Unknown source")} (p${escapeHtml(String(selectedSource.page_number || "-"))}) | Origin: ${escapeHtml(sourceOriginLabel(selectedRole))} | Policy: ${escapeHtml(String(selectedPolicy).toUpperCase())} | Score: ${selectedWeight}</div>
      `;

      const selectedCard = document.createElement("div");
      selectedCard.className = `evidence-candidate selected ${selectedContextOnly ? "context-only" : ""}`.trim();
      const signals = evidenceSignals(selected);
      selectedCard.innerHTML = `
        <div class="candidate-title">
          <span class="tag">Selected</span>
          ${selectedContextOnly ? '<span class="tag context-only">Context only source</span>' : '<span class="tag source-primary">Translation source</span>'}
        </div>
        <div class="candidate-quote">${escapeHtml(selected.quote || "(no excerpt available)")}</div>
        <div class="match-badges">
          ${signals.map(s => `<span class="match-badge ${s.ok ? "" : "off"}">${s.ok ? "Yes" : "No"} ${escapeHtml(s.label)}</span>`).join("")}
        </div>
        <div class="candidate-actions">
          ${selectedSource.page_url ? `<a href="${selectedSource.page_url}" target="_blank" rel="noopener">View source page</a>` : ""}
          ${selectedSource.page_id ? `<a href="#" onclick="openSource('${selectedSource.page_id}', true); return false;">Open source trace</a>` : ""}
        </div>
        <div class="context-note">${escapeHtml(selected.rank?.rank_reason || trace.selected_rank_reason || "")}</div>
      `;
      block.appendChild(selectedCard);

      if (filtered.length > 1) {
        const altWrap = document.createElement("div");
        altWrap.className = "alt-candidates";
        const title = document.createElement("div");
        title.className = "meta";
        title.textContent = "Alternative candidates";
        altWrap.appendChild(title);
        filtered.slice(1, 4).forEach((ev, idx) => {
          const src = ev.source || {};
          const role = src.source_role || "scholarly_commentary";
          const contextOnly = role !== "primary_text";
          const det = document.createElement("details");
          det.innerHTML = `
            <summary>${escapeHtml(src.pdf_name || `Candidate ${idx + 2}`)} (p${escapeHtml(String(src.page_number || "-"))}) | ${escapeHtml(sourceOriginLabel(role))}</summary>
            <div class="evidence-candidate ${contextOnly ? "context-only" : ""}">
              <div class="candidate-quote">${escapeHtml(ev.quote || "(no excerpt available)")}</div>
              <div class="context-note">${escapeHtml(ev.rank?.rank_reason || "")}</div>
            </div>
          `;
          altWrap.appendChild(det);
        });
        block.appendChild(altWrap);
      }

      sections.appendChild(block);
    }

    function appendConfidenceBreakdown(summary, sections) {
      if (!summary?.confidence || !sections) return;
      const conf = summary.confidence;
      const box = document.createElement("div");
      box.className = "confidence-breakdown";
      const reasons = Array.isArray(conf.reasons) ? conf.reasons : [];
      box.innerHTML = `<h4>Confidence Components</h4><div class="meta">Overall: ${escapeHtml(String(conf.level || "Unknown"))} (${escapeHtml(String(conf.score ?? ""))})</div>`;
      if (reasons.length) {
        const ul = document.createElement("ul");
        reasons.forEach((r) => {
          const li = document.createElement("li");
          li.textContent = r;
          ul.appendChild(li);
        });
        box.appendChild(ul);
      }
      sections.appendChild(box);
    }

    function renderKeyPointsAndEvidence(summary, chip) {
      const sections = document.getElementById("evidence-sections");
      if (!sections) return;
      sections.innerHTML = "";
      renderHumanExplanation(summary, chip);
      appendEvidenceDrivingSelection(summary, chip, sections);

      if (summary.key_points && summary.key_points.length) {
        const kpWrap = document.createElement("div");
        kpWrap.className = "evidence-keypoints";
        const h4 = document.createElement("h4");
        h4.textContent = chip ? `Key Points (filtered: ${chip})` : "Key Points";
        kpWrap.appendChild(h4);

        const ul = document.createElement("ul");
        summary.key_points
          .filter(kp => matchesChip(kp.text, kp.kind || kp.label, chip))
          .forEach(kp => {
            const li = document.createElement("li");
            const badge = document.createElement("span");
            badge.className = "tag kp-badge";
            badge.textContent = kp.label || "Key point";
            li.appendChild(badge);
            li.appendChild(document.createTextNode(" "));
            const span = document.createElement("span");
            span.textContent = kp.text;
            li.appendChild(span);
            if (kp.plain_english) {
              const pe = document.createElement("div");
              pe.className = "plain-english";
              const ctx = document.createElement("div");
              ctx.className = "pe-context";
              ctx.innerHTML = `<strong>What this is about:</strong> ${kp.plain_english.context}`;
              const para = document.createElement("div");
              para.className = "pe-paraphrase";
              para.innerHTML = `<strong>Plain English:</strong> ${kp.plain_english.paraphrase}`;
              const confNote = document.createElement("div");
              confNote.className = "pe-confidence";
              confNote.textContent = kp.plain_english.confidence_note || "";
              pe.appendChild(ctx);
              pe.appendChild(para);
              pe.appendChild(confNote);
              li.appendChild(pe);
            }
            if (kp.citations && kp.citations.length) {
              const cite = kp.citations.map(c => c.pdf_name ? `${c.pdf_name} p.${c.page_number}` : c.page_id).join("; ");
              const citeSpan = document.createElement("span");
              citeSpan.className = "citation-list";
              citeSpan.textContent = ` (${cite})`;
              li.appendChild(citeSpan);
            }
            ul.appendChild(li);
          });

        kpWrap.appendChild(ul);
        sections.appendChild(kpWrap);
      }

      if (summary.evidence && summary.evidence.length) {
        const evWrap = document.createElement("div");
        evWrap.className = "evidence-quotes";
        const h4e = document.createElement("h4");
        h4e.textContent = "Supporting references";
        evWrap.appendChild(h4e);

        summary.evidence
          .filter(ev => matchesChip(ev.quote, ev.source?.doc_type, chip))
          .forEach((ev, idx) => {
            const det = document.createElement("details");
            det.className = "evidence-item";
            const sm = document.createElement("summary");
            sm.textContent = ev.source?.pdf_name ? `${ev.source.pdf_name} – page ${ev.source.page_number}` : `Source ${idx+1}`;
            det.appendChild(sm);
            const qb = document.createElement("blockquote");
            qb.textContent = ev.quote || "(no excerpt available)";
            det.appendChild(qb);
            if (ev.source?.page_url) {
              const a = document.createElement("a");
              a.href = ev.source.page_url;
              a.target = "_blank";
              a.rel = "noopener";
              a.textContent = "View source page";
              det.appendChild(a);
            }
              const sourceRole = ev.source?.source_role || "scholarly_commentary";
              det.className = `evidence-item ${sourceRole === "primary_text" ? "" : "context-only"}`.trim();
              if (sourceRole !== "primary_text") {
                const roleTag = document.createElement("div");
                roleTag.className = "context-note";
                roleTag.textContent = "Context-only source: used for routing support and linkage, not direct translation text.";
                det.appendChild(roleTag);
              }
              const canTranslate = isTranslationAllowed(ev.source);
              if (ev.source?.page_id && canTranslate) {
                const btn = document.createElement("button");
                btn.textContent = "Translate this text";
                btn.style.marginLeft = "10px";
                btn.style.background = "#1f2c4d";
                btn.style.color = "#b6c8ff";
                btn.style.border = "none";
                btn.style.padding = "4px 8px";
                btn.style.borderRadius = "4px";
                btn.style.cursor = "pointer";
                btn.onclick = () => openHandoff(ev.source.page_id, ev.quote);
                det.appendChild(btn);
              } else if (ev.source?.page_id) {
                const note = document.createElement("div");
                note.className = "meta";
                note.textContent = "This source discusses documents. It does not contain a document suitable for translation.";
                det.appendChild(note);

                const linkBtn = document.createElement("button");
                linkBtn.textContent = translationReplacementLabel(sourceRole);
                linkBtn.style.marginLeft = "10px";
                linkBtn.style.background = "#1f2c4d";
                linkBtn.style.color = "#b6c8ff";
                linkBtn.style.border = "none";
                linkBtn.style.padding = "4px 8px";
                linkBtn.style.borderRadius = "4px";
                linkBtn.style.cursor = "pointer";
                linkBtn.onclick = () => openSource(ev.source.page_id, true);
                det.appendChild(linkBtn);
              }
            evWrap.appendChild(det);
          });
        sections.appendChild(evWrap);
      }

      appendConfidenceBreakdown(summary, sections);
    }

    function renderHumanExplanation(summary, chip) {
      const box = document.getElementById("interp-box");
      if (!box) return;
      const byKeyword = (summary.human_explanation_by_keyword || (summary.human_explanation && summary.human_explanation.human_explanation_by_keyword) || {});
      const interp = chip && byKeyword[chip] ? byKeyword[chip] : summary.human_explanation;
      if (!interp) {
        box.innerHTML = "";
        return;
      }
      box.innerHTML = "";
      const h4 = document.createElement("h4");
      h4.textContent = "What you're looking at";
      box.appendChild(h4);

      const addRow = (label, value) => {
        if (!value) return;
        const row = document.createElement("div");
        row.className = "interp-row";
        const strong = document.createElement("strong");
        strong.textContent = label + ": ";
        row.appendChild(strong);
        row.appendChild(document.createTextNode(value));
        box.appendChild(row);
      };

      addRow("What you're looking at", interp.what_you_are_looking_at);
      addRow("Main claim", interp.main_claim);
      addRow("Why it matters", interp.why_it_matters);
      addRow("Uncertainty", interp.uncertainty);
      addRow("Copy note", interp.copy_note);
      addRow("Translation note", interp.translated_note);

      if (interp.next_actions && interp.next_actions.length) {
        const actions = document.createElement("div");
        actions.className = "interp-actions";
        interp.next_actions.forEach((a) => {
          const link = document.createElement("a");
          link.href = `/corpus/citation?ref=${encodeURIComponent(a.ref)}&limit=5`;
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = `${a.label} (${a.ref})`;
          actions.appendChild(link);
        });
        box.appendChild(actions);
      }
    }

    const buildCards = (results, kind, mode, baselineResults = []) => {
      const topCompare = compareTopSelectionV2(baselineResults || [], results || []);
      const currentTopId = topCompare.currentTop ? String(topCompare.currentTop.page_id || "") : "";
      return results.map((r, idx) => {
        const url = r.page_url || `/corpus/page/${r.page_id}?include_text=false`;
        const badge = sourceBadge(r);
        const docLabel = docTypeLabel(r.doc_type);
        const docTipTitle = r.doc_type === "legal" ? "Legal text" : r.doc_type === "letter" ? "Letter" : "";
        const docTipBody = r.doc_type === "legal"
          ? "A formal document type used for contracts, obligations, loans, guarantees, and settlements. Legal texts follow standardized formulas, list witnesses, and often invoke divine authority to enforce agreements."
          : r.doc_type === "letter"
          ? "A private or semi-formal correspondence between individuals. Letters often mix personal language with legal formulas, making them valuable for understanding how institutions and oaths were used in everyday practice."
          : "";
        const srcType = sourceTypeLabel(r.source_role);
        const si = structuralOf(r);
        const routing = routingOf(r) || {};
        const policy = (routing.selected_policy || "INTERNAL").toUpperCase();
        const modeScore = scoreForMode(r, mode);
        const conf = confidenceBandV2(routing);
        const isTop = String(r.page_id || "") === currentTopId;
        const changedTop = topCompare.changed && isTop;
        const cardClass = changedTop ? "card changed-card" : "card";
        const changedTag = changedTop ? `<span class="tag changed">Changed from Internal baseline</span>` : "";
        const roleContextTag = r.source_role === "primary_text" ? "" : `<span class="tag context-only">Context only</span>`;
        const whySelected = methodExplanationV2(r);
        const diffLine = (idx === 0)
          ? `<div class="result-diff">Selection change: ${topCompare.changed ? "YES" : "NO"} | baseline ${escapeHtml(String(topCompare.basePage || "-"))} -> current ${escapeHtml(String(topCompare.currentPage || "-"))}</div>`
          : "";
        return `
          <div class="${cardClass}">
            <div class="meta">
              <span class="tag" data-tip-title="Source role" data-tip="Primary text = original letters/contracts. Archival wrapper = envelopes, seals, witness lists that surround the text. Scholarly commentary = analysis, joins, or reconstructions. We show Primary first because it is closest to the original evidence.">${badge}</span>
              <span class="tag" ${docTipTitle ? `data-tip-title="${docTipTitle}"` : ""} ${docTipBody ? `data-tip="${docTipBody}"` : ""}>${docLabel}</span>
              <span class="tag ${srcType.cls}" data-tip-title="Source type" data-tip="Source role controls translation access. Primary texts are translatable; archival wrappers and scholarly commentary provide context and metadata.">${srcType.text}</span>
              <span class="tag" data-tip-title="Routing policy" data-tip="${escapeHtml(whySelected)}">Policy: ${escapeHtml(policy)}</span>
              <span class="confidence-pill ${conf.cls}">${conf.level}</span>
              ${changedTag}
              ${roleContextTag}
            </div>
            ${diffLine}
            <div class="meta">${metaLine(r)}</div>
            ${structuralBadgesHtmlV2(r)}
            <div class="snippet">${markTerm(r.snippet || "")}</div>
            <div class="meta"><a href="${url}" target="_blank">${r.pdf_name} (p${r.page_number})</a></div>
            <div class="meta" style="font-size:11px;color:#8fb0e4;">Mode score: ${Number(modeScore).toFixed(2)} | Internal ${Number(routing.internal_score ?? r.evidence_weight ?? 0).toFixed(2)} | ORACC ${Number(routing.oracc_score ?? r.evidence_weight ?? 0).toFixed(2)}</div>
            ${routingTraceHtmlV2(si)}
            <div class="meta" style="font-size:11px;color:#7ea6d6;">
              ${trustLine(r)}
              <button class="help" type="button"
                data-tip-title="Sources (${r.citations ? r.citations.length : 0} citations)"
                data-tip="The number of distinct citations detected on this page (e.g., ICK, CCT, BIN, AKT, Kt…). Higher counts usually mean the page is a strong hub connecting this topic to published primary texts."
                aria-label="Explain sources count">?</button>
              <button class="help" type="button"
                data-tip-title="Signals"
                data-tip="Signals are the features our pipeline detected on this page. Citations = links to specific editions/text IDs. Formula = repeated phrase patterns (oath/witness language). Institution = economic/legal vocabulary (e.g., naruqqum, karum). More signals generally means higher confidence."
                aria-label="Explain signals">?</button>
              · <a href="#" class="btn-view-sources" data-page-id="${r.page_id}" data-kind="${kind}">View sources</a>
            </div>
          </div>
        `;
      }).join("");
    };

    const renderStats = (data) => {
      if (!data || !data.ok) return;
      const p = data.percentages;
      const c = data.counts;
      const total = c.total || 0;
      const stat = (label, key, tipTitle, tipBody) => `
        <div class="stat-card help" data-tip-title="${tipTitle}" data-tip="${tipBody}">
          <button class="stat-help" type="button" data-tip-title="${tipTitle}" data-tip="${tipBody}">?</button>
          <div class="stat-label">${label}</div>
          <div class="stat-value">${p[key]}%</div>
          <div class="stat-label">${c[key]} pages</div>
        </div>`;
      statsEl.innerHTML = [
        `<div class="stat-card help" data-tip-title="Corpus indexed" data-tip="The total number of pages the system has scanned and analyzed. Each page is checked for names, formulas, and institutions so it can be searched and compared."><button class="stat-help" type="button" data-tip-title="Corpus indexed" data-tip="The total number of pages the system has scanned and analyzed. Each page is checked for names, formulas, and institutions so it can be searched and compared.">?</button><div class="stat-label">Corpus indexed</div><div class="stat-value">${total}</div></div>`,
        stat("Pages with traceable citations", "citations", "Pages with traceable citations", "Pages that contain clear references to specific texts or editions (like ICK, CCT, AKT). These links let us trace claims back to known sources."),
        stat("Pages with recognizable formulas", "formula_markers", "Pages with recognizable formulas", "Pages that include repeated legal phrases, such as oath or witness language. These formulas help identify contracts, promises, and official statements."),
        stat("Pages mentioning institutions", "institutions", "Pages mentioning institutions", "Pages that include words for organized systems like trade partnerships, loans, or courts (e.g., naruqqum). These show how people worked together in business and law."),
        stat("Pages classified by type", "doc_type_classified", "Pages classified by type", "Pages the system could confidently label as letters, legal documents, commentary, or reference material. Knowing the type helps us decide how strong the evidence is."),
      ].join("");
    };

    const renderModeButtons = () => {
      document.querySelectorAll("#view-mode-toggle .mode-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.mode === currentViewMode);
      });
    };

    const renderDemo = (data) => {
      demoCache = data;
      renderModeButtons();

      const deityBase = data.deity_attestations.results || [];
      const formulaBase = data.formula_examples.results || [];
      const instBase = data.institution_examples.results || [];

      const deityView = applyStructuralFilters(applyViewMode(deityBase, currentViewMode));
      const formulaView = applyStructuralFilters(applyViewMode(formulaBase, currentViewMode));
      const instView = applyStructuralFilters(applyViewMode(instBase, currentViewMode));

      const deityInternalView = applyStructuralFilters(applyViewMode(deityBase, "internal"));
      const formulaInternalView = applyStructuralFilters(applyViewMode(formulaBase, "internal"));
      const instInternalView = applyStructuralFilters(applyViewMode(instBase, "internal"));

      const deityDelta = selectionDelta(deityInternalView, deityView);
      const formulaDelta = selectionDelta(formulaInternalView, formulaView);
      const instDelta = selectionDelta(instInternalView, instView);

      updateRoutingAnalytics([deityView, formulaView, instView]);

      panelsEl.innerHTML = `
        <div class="panel">
          <h2>Belief · Deity evidence <span class="term">istarzaat <button class="help" type="button" data-tip-title="Ištar-ZA.AT (istarzaat)" data-tip="A specific way Old Assyrian texts refer to the goddess Ištar, often in formal/legal contexts. In this demo, it’s used as a “belief signal” because it shows where divine authority is invoked in real documents (oaths, contracts, witness statements)." aria-label="What is istarzaat?">?</button></span></h2>
          <div class="meta" style="margin-bottom:6px;">Showing top ${data.deity_attestations.count_returned} (mode: ${currentViewMode.toUpperCase()})</div>
          <div class="delta-note">Δ exemplar selection vs Internal: ${deityDelta}</div>
          ${renderDecisionPanelV2("Belief", deityView, deityInternalView, currentViewMode)}
          ${buildCards(deityView, "deity", currentViewMode, deityInternalView)}
        </div>
        <div class="panel">
          <h2>Speech · Oath / witness formulas <span class="term">li-tù-la <button class="help" type="button" data-tip-title="li-tù-la" data-tip="A common Old Assyrian “formula marker” meaning something like “may X be witnesses / may X see.” In this demo, it’s a “speech signal” because it highlights standardized legal language used in letters and contracts." aria-label="What is li-tù-la?">?</button></span></h2>
          <div class="meta" style="margin-bottom:6px;">Showing top ${data.formula_examples.count_returned} (mode: ${currentViewMode.toUpperCase()})</div>
          <div class="delta-note">Δ exemplar selection vs Internal: ${formulaDelta}</div>
          ${renderDecisionPanelV2("Speech", formulaView, formulaInternalView, currentViewMode)}
          ${buildCards(formulaView, "formula", currentViewMode, formulaInternalView)}
        </div>
        <div class="panel">
          <h2>Behavior · Economic institutions <span class="term">naruqqum <button class="help" type="button" data-tip-title="naruqqum" data-tip="A joint-stock / pooled-capital partnership used in Old Assyrian commerce. In this demo, it’s a “behavior signal” because it points to how trade was organized (investment, risk, obligations), not just what people said." aria-label="What is naruqqum?">?</button></span></h2>
          <div class="meta" style="margin-bottom:6px;">Showing top ${data.institution_examples.count_returned} (mode: ${currentViewMode.toUpperCase()})</div>
          <div class="delta-note">Δ exemplar selection vs Internal: ${instDelta}</div>
          ${renderDecisionPanelV2("Behavior", instView, instInternalView, currentViewMode)}
          ${buildCards(instView, "institution", currentViewMode, instInternalView)}
        </div>
        <div class="panel">
          <h2>User Provided</h2>
          <p class="meta">Paste text (preview). Demo runs lightweight extraction only.</p>
          <textarea id="userText" placeholder="Paste text..."></textarea>
          <button onclick="analyze()">Analyze</button>
          <div id="userResults" class="card" style="margin-top:8px; display:none;"></div>
        </div>
      `;
      panelsEl.querySelectorAll(".btn-view-sources").forEach((el) => {
        el.addEventListener("click", (evt) => {
          evt.preventDefault();
          evt.stopPropagation();
          const pageId = el.getAttribute("data-page-id") || "";
          const kind = el.getAttribute("data-kind") || "";
          if (!pageId) return;
          openDrawer(pageId, kind);
        });
      });
    };

    document.querySelectorAll("#view-mode-toggle .mode-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const nextMode = btn.dataset.mode || "routed";
        if (!nextMode || nextMode === currentViewMode) return;
        currentViewMode = nextMode;
        renderModeButtons();
        if (demoCache) renderDemo(demoCache);
      });
    });

    [filterFragmentationEl, filterTemplateEl, filterPolicyEl, filterOriginEl].forEach((el) => {
      if (!el) return;
      el.addEventListener("change", () => {
        if (demoCache) renderDemo(demoCache);
      });
    });

    function analyze() {
      const txt = document.getElementById("userText").value || "";
      const box = document.getElementById("userResults");
      if (!txt.trim()) { box.style.display='none'; return; }
      box.style.display='block';
      box.innerHTML = `<div class="meta">What we recognize in this text</div><div class="snippet">${markTerm(txt.substring(0,300))}${txt.length>300?'...':''}</div><div class="meta" style="font-size:11px;color:#7ea6d6;">Status: not yet matched to the corpus</div>`;
    }

    // Evidence summary renderer
    function renderEvidenceSummary(summary) {
      const root = document.getElementById("evidence-summary");
      if (!root) return;
      root.innerHTML = "";
      if (!summary || !summary.summary) {
        root.innerHTML = "<p class='meta'>No evidence summary available.</p>";
        return;
      }
      const block = document.createElement("div");
      block.className = "evidence-summary-block";
      const title = document.createElement("h3");
      title.textContent = "Evidence Summary - Decision Trace";
      block.appendChild(title);
      const p = document.createElement("p");
      p.textContent = summary.summary;
      block.appendChild(p);
      const sub = document.createElement("div");
      sub.className = "meta";
      sub.textContent = "This section explains which source was selected, why it was selected, and what alternatives were considered.";
      block.appendChild(sub);
      root.appendChild(block);
      const interpBox = document.createElement("div");
      interpBox.id = "interp-box";
      interpBox.className = "interp-card";
      root.appendChild(interpBox);
      renderHumanExplanation(summary, null);
      const chipsEl = renderKeywordChips(summary, (chip) => {
        renderKeyPointsAndEvidence(summary, chip);
      });
      root.appendChild(chipsEl);

      const sections = document.createElement("div");
      sections.id = "evidence-sections";
      root.appendChild(sections);

      renderKeyPointsAndEvidence(summary, null);
    }

    function openDrawer(pageId, kind) {
      currentPage = pageId;
      drawer.classList.add("open");
      drawerList.innerHTML = "<div class='meta'>Loading citations...</div>";
      drawerDetail.innerHTML = "Select a citation to see context.";
      currentCitationList = [];
      currentCitationIndex = -1;
      fetch(`/corpus/page/${pageId}/citations`)
        .then(r => r.json())
        .then(data => {
          if (!data.ok) throw new Error("No citations");
          const srcMethod = data.source_page?.structural_intelligence?.routing || {};
          const srcPolicy = (srcMethod.selected_policy || "INTERNAL").toUpperCase();
          drawerSource.innerHTML = `${data.source_page.pdf_name} (p${data.source_page.page_number}) | ${data.source_page.doc_type || "unknown"} | weight ${data.source_page.evidence_weight} | policy ${srcPolicy} <button class="help" type="button" data-tip-title="Source line" data-tip="Current page context for this citation list: document, page, evidence weight, and active routing policy.">?</button>`;
          currentCitationList = data.citations || [];
          drawerList.innerHTML = currentCitationList.map((c, idx) => {
            return `<div class="list-item" onclick="loadCitation('${encodeURIComponent(c.ref)}', ${idx})">
              <div><strong>${c.ref}</strong></div>
              <div class="meta">${(c.ref_type || "unknown").toUpperCase()} · ${c.doc_hint || ""}</div>
            </div>`;
          }).join("");
        })
        .catch(err => {
          drawerList.innerHTML = `<div class='meta'>Citation explorer not available. Try another item or open the source page.</div>`;
        });
    }

    function closeDrawer() {
      drawer.classList.remove("open");
      currentPage = null;
      drawer.style.right = "-480px";
      closeSource();
    }

      function closeSource() {
        sourcePanel.classList.remove("open");
        sourcePanel.style.right = "-520px";
        drawer.style.right = "-480px";
        sourceContent.innerHTML = "Select a source to view here.";
        currentSourceResults = [];
        currentSourceIndex = -1;
        currentSourceMeta = null;
      }

      function buildPaperTrailHeader(sourceRole, hasCitations) {
        const role = sourceRole || "scholarly_commentary";
        const roleMeta = {
          primary_text: {
            title: "Primary Text (Tablet)",
            blurb: "This is the original Old Assyrian document.",
          },
          archival_wrapper: {
            title: "Archival Wrapper (Envelope / Seal)",
            blurb: "Contains seals, witnesses, and metadata.",
          },
          scholarly_commentary: {
            title: "Scholarly Commentary",
            blurb: "You are reading analysis, not an original document.",
          },
        };
        const meta = roleMeta[role] || roleMeta.scholarly_commentary;
        const gateCopy = role === "primary_text"
          ? ""
          : '<div class="meta gate-copy">This source discusses documents. It does not contain a document suitable for translation.</div>';
        let ctaHtml = "";
        if (role === "primary_text") {
          ctaHtml = '<div class="meta ladder-cta ok">Translation available</div>';
        } else if (hasCitations) {
          const label = role === "archival_wrapper" ? "View enclosed tablet" : "Jump to linked primary text";
          ctaHtml = `<button class="ladder-cta" type="button" onclick="jumpToPrimaryFromCurrentSource()">${label}</button>`;
        } else {
          ctaHtml = '<div class="meta ladder-cta muted">No linked primary text available.</div>';
        }
        const ladderHtml = `
          <div class="paper-ladder">
            <div class="ladder-step ${role === "scholarly_commentary" ? "active" : ""}">Scholarly Commentary</div>
            <div class="ladder-arrow">&darr;</div>
            <div class="ladder-step ${role === "archival_wrapper" ? "active" : ""}">Archival Wrapper (Envelope / Seal)</div>
            <div class="ladder-arrow">&darr;</div>
            <div class="ladder-step ${role === "primary_text" ? "active" : ""}">Primary Text (Tablet / Letter)</div>
          </div>
        `;
        return `
          <div class="source-header">
            <div class="role-title">${meta.title}</div>
            <div class="meta">${meta.blurb}</div>
            ${gateCopy}
            ${ctaHtml}
            ${ladderHtml}
          </div>
        `;
      }

      function openSource(pageId, fromList = false, afterRender = null) {
        sourcePanel.classList.add("open");
        sourcePanel.style.right = "0";
        drawer.style.right = "500px";
      sourceContent.innerHTML = "<div class='meta'>Loading source...</div>";
      fetch(`/corpus/page/${pageId}/story`)
        .then(r => r.json())
        .then(data => {
          if (!data.ok) { sourceContent.innerHTML = `<div class='meta'>${data.error || "Could not load source."}</div>`; return; }
          const p = data.page || {};
          const sourceRole = data.source_role || "scholarly_commentary";
          const anchorHint = (data.snippets && data.snippets.length) ? (data.snippets[0].text || "") : "";
          currentSourceMeta = {
            source_role: sourceRole,
            citations: data.citations || [],
            anchor: anchorHint,
          };
          const highlights = (data.highlights || []).map(h => `<div class="meta"><strong>${h.label}:</strong> ${markTerm(h.value || "")}</div>`).join("");
          const why = (data.why_this_matters || []).map(w => `<div class="meta">• ${w}</div>`).join("");
          const cits = (data.citations || []).map(c => `<span class="badge">${c.ref}</span>`).join(" ");
          const snips = (data.snippets || []).map(s => `<div class="section"><div class="meta"><strong>${s.topic || ""}</strong></div><div class="snippet">${markTerm(s.text || "")}</div></div>`).join("");
          const breadcrumb = `Belief · Evidence → ${p.title || "Source"} (p${p.page_number || ""})`;
          const header = buildPaperTrailHeader(sourceRole, (data.citations || []).length);
          const roleBadge = sourceRoleHeaderLabel(sourceRole);
          const docTag = sourceRoleSecondaryTag(sourceRole, p.doc_type || "");
          const headerMeta = `<span class="badge">${roleBadge}</span>${docTag ? ` <span class="badge">${docTag}</span>` : ""}`;
          const structuralHtml = renderStructuralHeaderV2(data.structural_intelligence || null);
          sourceContent.innerHTML = `
            ${header}
                <div class="section">
                  <h4>${breadcrumb}</h4>
                  <div class="meta">${headerMeta}</div>
                </div>
            <div class="section">
              ${structuralHtml}
            </div>
            <div class="section">
              <h4>Highlights</h4>
              ${highlights || "<div class='meta'>None listed.</div>"}
            </div>
            <div class="section">
              <h4>Why this matters</h4>
              ${why || "<div class='meta'>Contextual support.</div>"}
            </div>
            <div class="section">
              <h4>Citations</h4>
              ${cits || "<div class='meta'>None listed.</div>"}
            </div>
            <div class="section">
              <h4>Snippets</h4>
              ${snips || "<div class='meta'>No snippets.</div>"}
            </div>
            <div class="section">
              <h4>Raw view</h4>
              <a href="/corpus/page/${p.page_id}?include_text=false" target="_blank">Developer view</a>
            </div>
            <div class="section">
              <button onclick="stepSource(-1)" style="background:#1f2c4d;color:#b6c8ff;border:none;padding:6px 8px;border-radius:4px;cursor:pointer;margin-right:6px;">Prev source</button>
              <button onclick="stepSource(1)" style="background:#1f2c4d;color:#b6c8ff;border:none;padding:6px 8px;border-radius:4px;cursor:pointer;">Next source</button>
            </div>
          `;
          if (typeof afterRender === "function") afterRender();
        })
          .catch(() => {
            sourceContent.innerHTML = "<div class='meta'>Could not load source.</div>";
          });
      }

      function pickBestRefFromCitations(citations) {
        if (!citations || !citations.length) return null;
        return citations[0].ref || citations[0];
      }

      function appendPrimaryExcerptBlock(handoffData, anchor) {
        if (!handoffData || !handoffData.page_id) return;
        const blockId = `primary-excerpt-${handoffData.page_id}`;
        if (document.getElementById(blockId)) return;
        const excerptFocus = handoffData.excerpt_focus || handoffData.excerpt_short || "";
        const focusLabel = handoffData.anchor_matched ? "Focused excerpt (matched)" : "Focused excerpt (default)";

        const block = document.createElement("div");
        block.className = "section";
        block.id = blockId;
        block.innerHTML = `
          <h4>Primary excerpt (from ${handoffData.pdf_name} p${handoffData.page_number})</h4>
          <div class="meta">${docTypeLabel(handoffData.doc_type)} · Translation available</div>
          <div class="meta" style="margin-top:6px;"><strong>${focusLabel}</strong></div>
          <div class="snippet" style="max-height:none; display:block; white-space:pre-wrap;">${escapeHtml(excerptFocus)}</div>
          <div class="meta" style="margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;">
            <button id="btn-open-primary-${handoffData.page_id}" style="background:#1f2c4d;color:#b6c8ff;">Open primary source</button>
            <button id="btn-translate-primary-${handoffData.page_id}" style="background:#1f2c4d;color:#b6c8ff;">Translate this text</button>
          </div>
        `;
        sourceContent.appendChild(block);

        document.getElementById(`btn-open-primary-${handoffData.page_id}`).onclick = () => {
          openSource(handoffData.page_id, true);
        };
        document.getElementById(`btn-translate-primary-${handoffData.page_id}`).onclick = () => {
          openHandoff(handoffData.page_id, anchor || "");
        };
      }

      function fetchPrimaryExcerptFromRef(ref, anchor) {
        if (!ref) return;
        const anchorParam = anchor ? `&anchor=${encodeURIComponent(anchor)}` : "";
        fetch(`/corpus/resolve/${encodeURIComponent(ref)}?limit=10${anchorParam}`)
          .then(r => r.json())
          .then(data => {
            if (!data.ok || !data.best) return;
            const primary = data.best;
            const handoffAnchor = anchor || "";
            const handoffParam = handoffAnchor ? `?anchor=${encodeURIComponent(handoffAnchor)}` : "";
            fetch(`/corpus/handoff/${primary.page_id}${handoffParam}`)
              .then(r => r.json())
              .then(handoff => {
                if (!handoff.ok) return;
                appendPrimaryExcerptBlock(handoff, handoffAnchor);
              })
              .catch(() => {});
          })
          .catch(() => {});
      }

      function jumpToPrimaryFromCurrentSource() {
        if (!currentSourceMeta) return;
        const ref = pickBestRefFromCitations(currentSourceMeta.citations);
        if (!ref) return;
        const anchorParam = currentSourceMeta.anchor ? `&anchor=${encodeURIComponent(currentSourceMeta.anchor)}` : "";
        fetch(`/corpus/resolve/${encodeURIComponent(ref)}?limit=10${anchorParam}`)
          .then(r => r.json())
          .then(data => {
            if (!data.ok || !data.best) return;
            const primary = data.best;
            openSource(primary.page_id, true);
            openHandoff(primary.page_id, primary.snippet || "");
          })
          .catch(() => {});
      }

      function copyToClipboard(text) {
      if (!text) return;
      if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).catch(() => {});
        return;
      }
      const tmp = document.createElement("textarea");
      tmp.value = text;
      document.body.appendChild(tmp);
      tmp.select();
      try { document.execCommand("copy"); } catch (e) {}
      document.body.removeChild(tmp);
    }

    function openHandoff(pageId, anchor) {
      openSource(pageId, true, () => {
        if (anchor && currentSourceMeta) {
          currentSourceMeta.anchor = anchor;
        }
        const anchorParam = anchor ? `?anchor=${encodeURIComponent(anchor)}` : "";
        fetch(`/corpus/handoff/${pageId}${anchorParam}`)
          .then(r => r.json())
          .then(data => {
            if (!data.ok) return;

            const block = document.createElement("div");
            block.className = "section";

            const summary = data.handoff_summary || {};
            const keywords = (data.keywords || []).map(k => `<span class="badge">${k}</span>`).join(" ");
            const structural = data.structural_intelligence || summary.structural_intelligence || null;
            const structuralHtml = renderStructuralHeaderV2(structural);

            const sourceRoleCode = summary.source_role_code || "scholarly_commentary";
            const translationAllowed = summary.translation_allowed === true || sourceRoleCode === "primary_text";
            const context = summary.context_capsule || {};
            const contextRows = [
              ["Document type", context.document_type],
              ["Parties involved", context.parties],
              ["Commodity", context.commodity],
              ["Source chain", context.source_chain],
              ["Confidence", context.confidence],
            ].filter(([, value]) => value);
            const contextHtml = translationAllowed
              ? (contextRows.length
                  ? `<ul class="context-list">${contextRows
                      .map(([label, value]) => `<li><strong>${label}:</strong> ${escapeHtml(value)}</li>`)
                      .join("")}</ul>`
                  : "<div class='meta'>Context unavailable.</div>")
              : "<div class='meta'>Context extraction is available only for primary texts. Jump to linked primary text to translate.</div>";
            const gateCopyHtml = translationAllowed
              ? ""
              : `<div class="meta gate-copy">${escapeHtml(summary.translation_gate_copy || "")}</div>`;
            const modes = translationAllowed ? (summary.translation_modes || []) : [];
            const modesHtml = modes.length
              ? modes
                  .map((m) => {
                    const label = m.mode || m.label || "";
                    const status = m.status || "";
                    const cls = status === "available" ? "mode-on" : "mode-off";
                    const suffix = status && status !== "available" ? ` (${status})` : "";
                    return `<span class="badge ${cls}">${escapeHtml(label + suffix)}</span>`;
                  })
                  .join(" ")
              : "<span class='meta'>not configured</span>";
            const translationBlock = translationAllowed
              ? `<div class="meta" style="margin-top:8px;"><strong>Translation scope:</strong> ${escapeHtml(summary.translation_scope || "")}</div>
                 <div class="meta" style="margin-top:6px;"><strong>Translation mode:</strong> ${modesHtml}</div>`
              : "";
            const canJump = !translationAllowed && (data.citations_preview || []).length;
            const jumpHtml = !translationAllowed
              ? (canJump
                  ? `<div class="meta" style="margin-top:6px;"><strong>What you can do:</strong> Jump to linked primary text.</div>
                     <div class="meta" style="margin-top:6px;"><button id="btn-jump-primary-${pageId}" style="background:#1f2c4d;color:#b6c8ff;">Jump now</button></div>`
                  : "<div class='meta' style='margin-top:6px;'>No linked primary text available.</div>")
              : "";

            const snippets = data.snippets_structured || [];
            const bestRef = pickBestRefFromCitations(data.citations_preview || data.citations || []);
            const showSnippetList = !translationAllowed && snippets.length;
            const snippetText = showSnippetList
              ? snippets.map((s) => `${s.marker || "Snippet"}: ${s.snippet || ""}`).join("\n")
              : "";
            const snippetsHtml = showSnippetList
              ? `<div class="meta" style="margin-top:8px;"><strong>Evidence snippets</strong></div>` +
                snippets.map((s) => {
                  const snippetText = s.snippet || "";
                  const label = s.marker ? `Snippet about ${s.marker}` : "Snippet";
                  return `
                    <div class="snippet-card">
                      <div class="meta"><strong>${escapeHtml(label)}</strong></div>
                      <div class="snippet" style="max-height:none; display:block; white-space:pre-wrap;">${escapeHtml(snippetText)}</div>
                      <div class="meta" style="margin-top:6px;">
                        <button class="btn-snippet-jump" data-ref="${bestRef ? encodeURIComponent(bestRef) : ""}" data-anchor="${encodeURIComponent(snippetText)}" style="background:#1f2c4d;color:#b6c8ff;" ${bestRef ? "" : "disabled"}>Open primary</button>
                      </div>
                    </div>
                  `;
                }).join("")
              : "";
            const excerptFocus = data.excerpt_focus || data.excerpt_short || "";
            const excerptFull = data.excerpt_full || "";
            const focusLabel = data.anchor_matched ? "Focused excerpt (matched)" : "Focused excerpt (default)";
            let expanded = false;
            const excerptId = `handoff-excerpt-${pageId}`;
            const excerptHtml = showSnippetList
              ? `${snippetsHtml}<div id="${excerptId}" class="snippet" style="display:none;">${escapeHtml(snippetText)}</div>`
              : `<div class="meta" style="margin-top:8px;"><strong>${focusLabel}</strong></div>
                 <div id="${excerptId}" class="snippet" style="max-height:none; display:block; white-space:pre-wrap;">${escapeHtml(excerptFocus)}</div>`;
            const copyFocusText = showSnippetList ? snippetText : excerptFocus;
            const copyFullText = showSnippetList ? snippetText : excerptFull;
            const toggleStyle = showSnippetList ? "display:none;" : "";

            block.innerHTML = `
              <h4>Translation handoff</h4>
              ${structuralHtml}
              <div class="meta"><strong>Source:</strong> ${data.pdf_name} (p${data.page_number}) ? ${docTypeLabel(data.doc_type)}</div>
              <div class="meta"><strong>Handoff summary:</strong> ${escapeHtml(summary.source_role || "-")} ? ${escapeHtml(summary.doc_type_label || "-")}</div>
              <div class="meta">${escapeHtml(summary.why_opened || "")}</div>
              ${gateCopyHtml}
              <div class="meta"><strong>Signals:</strong> ${keywords || "<span class='meta'>none</span>"}</div>

              <div class="meta" style="margin-top:8px;"><strong>Context</strong></div>
              ${contextHtml}
              ${jumpHtml}
              ${translationBlock}

              <div class="meta" style="margin-top:8px;"><strong>Paper trail:</strong> ${(data.citations_preview || []).join("; ") || "No citations listed"}</div>

              <div class="meta" style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
                <button id="btn-copy-excerpt-${pageId}" style="background:#1f2c4d;color:#b6c8ff;">Copy excerpt</button>
                <button id="btn-copy-cits-${pageId}" style="background:#1f2c4d;color:#b6c8ff;">Copy citations</button>
                <button id="btn-copy-all-${pageId}" style="background:#1f2c4d;color:#b6c8ff;">Copy all</button>
                <button id="btn-toggle-${pageId}" style="background:#1f2c4d;color:#b6c8ff;${toggleStyle}">Show more</button>
              </div>

              ${excerptHtml}

              <div class="meta" style="margin-top:10px;"><em>${escapeHtml(summary.note || "")}</em></div>
            `;

            sourceContent.appendChild(block);

            const jumpBtn = document.getElementById(`btn-jump-primary-${pageId}`);
            if (jumpBtn) {
              jumpBtn.onclick = () => {
                const ref = pickBestRefFromCitations(data.citations_preview || data.citations || []);
                fetchPrimaryExcerptFromRef(ref, anchor || "");
              };
            }

            const snippetButtons = block.querySelectorAll(".btn-snippet-jump");
            snippetButtons.forEach((btn) => {
              const refEncoded = btn.dataset.ref || "";
              const anchorEncoded = btn.dataset.anchor || "";
              if (!refEncoded) return;
              const ref = decodeURIComponent(refEncoded);
              const anchorText = decodeURIComponent(anchorEncoded || "");
              btn.onclick = () => fetchPrimaryExcerptFromRef(ref, anchorText);
            });

            document.getElementById(`btn-copy-excerpt-${pageId}`).onclick = () => {
              copyToClipboard(expanded ? copyFullText : copyFocusText);
            };
            document.getElementById(`btn-copy-cits-${pageId}`).onclick = () => {
              copyToClipboard((data.citations || []).join("; "));
            };
            document.getElementById(`btn-copy-all-${pageId}`).onclick = () => {
              copyToClipboard(data.copy_payload || "");
            };
            document.getElementById(`btn-toggle-${pageId}`).onclick = () => {
              expanded = !expanded;
              const excerptEl = document.getElementById(excerptId);
              if (!excerptEl) return;
              excerptEl.textContent = expanded ? copyFullText : copyFocusText;
              document.getElementById(`btn-toggle-${pageId}`).textContent = expanded ? "Show less" : "Show more";
            };
          });
      });
    }

    function loadCitation(ref, idx = -1) {
      if (idx >= 0) currentCitationIndex = idx;
      drawerDetail.innerHTML = "<div class='meta'>Loading...</div>";
      fetch(`/corpus/citation?ref=${ref}&limit=5`)
        .then(r => r.json())
        .then(data => {
          if (!data.ok) {
            drawerDetail.innerHTML = `<div class='meta'>${data.error || "Could not load citation."}</div>`;
            return;
          }
          if (!data.results || !data.results.length) {
            drawerDetail.innerHTML = "<div class='meta'>No matches found for this citation.</div>";
            return;
          }
          if (data.evidence_summary) {
            renderEvidenceSummary(data.evidence_summary);
          }
          currentSourceResults = data.results;
          currentSourceIndex = 0;
          const r = data.results[0];
          const srcType = sourceTypeLabel(r.source_role || "scholarly_commentary");
          const structuralHtml = renderStructuralHeaderV2(r.structural_intelligence || null);
          const openLink = r.page_id ? `<a href="#" onclick="openSource('${r.page_id}');return false;">Open source</a>` : `<span style="color:var(--muted)">Source link unavailable</span>`;
          drawerDetail.innerHTML = `
            <div class="meta"><span class="badge">${docTypeLabel(r.doc_type)}</span> <span class="badge">${r.ref_type ? r.ref_type.toUpperCase() : "SOURCE"}</span> <span class="badge">${r.doc_hint || ""}</span> <span class="badge ${srcType.cls}">${srcType.text}</span> <button class="help" type="button" data-tip-title="Citation detail badges" data-tip="These badges summarize document type, citation class, hint label, and source role (primary/wrapper/commentary) for the currently selected citation result.">?</button></div>
            ${structuralHtml}
            <div class="snippet">${markTerm(r.excerpt && r.excerpt.text ? r.excerpt.text : (r.snippet || ""))}</div>
            <div class="meta" style="font-size:11px;color:#7ea6d6;">${r.rank && r.rank.rank_reason ? "Why it matters: " + r.rank.rank_reason : ""}</div>
            <div class="meta" style="font-size:11px;color:#7ea6d6;">${openLink}</div>
            <div class="meta" style="font-size:11px;color:#7ea6d6;">
              <button onclick="stepCitation(-1)" style="background:#1f2c4d;color:#b6c8ff;border:none;padding:4px 6px;border-radius:4px;cursor:pointer;">Prev</button>
              <button onclick="stepCitation(1)" style="background:#1f2c4d;color:#b6c8ff;border:none;padding:4px 6px;border-radius:4px;cursor:pointer;">Next</button>
            </div>
          `;
        })
        .catch(() => {
          drawerDetail.innerHTML = "<div class='meta'>Citation lookup failed. Try another or open the source page.</div>";
        });
    }

    function stepCitation(delta) {
      if (!currentCitationList.length) return;
      let idx = currentCitationIndex >= 0 ? currentCitationIndex + delta : delta;
      if (idx < 0) idx = currentCitationList.length - 1;
      if (idx >= currentCitationList.length) idx = 0;
      currentCitationIndex = idx;
      const ref = currentCitationList[idx].ref;
      loadCitation(encodeURIComponent(ref), idx);
    }

    function stepSource(delta) {
      if (!currentSourceResults.length) return;
      let idx = currentSourceIndex >= 0 ? currentSourceIndex + delta : delta;
      if (idx < 0) idx = currentSourceResults.length - 1;
      if (idx >= currentSourceResults.length) idx = 0;
      currentSourceIndex = idx;
      const r = currentSourceResults[idx];
      openSource(r.page_id, true);
    }

    Promise.all([fetch("/corpus/demo"), fetch("/corpus/stats")])
      .then(([d, s]) => Promise.all([d.json(), s.json()]))
      .then(([data, stats]) => {
        if (!data.ok) throw new Error("Demo not ok");
        renderStats(stats);
        renderDemo(data);
      })
      .catch(err => {
        panelsEl.innerHTML = '<div class="panel"><h2>Error</h2><div class="snippet">' + err + '</div></div>';
      });

    // Tooltip logic
    let tipEl = null;
    let tipAnchor = null;
    function closeTip() {
      if (tipEl) tipEl.remove();
      tipEl = null;
      tipAnchor = null;
    }
    function showTip(btn) {
      if (tipAnchor === btn && tipEl) return;
      closeTip();
      tipAnchor = btn;
      const title = btn.dataset.tipTitle || "Info";
      const body = btn.dataset.tip || "";
      tipEl = document.createElement("div");
      tipEl.className = "tooltip";
      tipEl.innerHTML = `
        <div class="tooltip-title">${title}</div>
        <div class="tooltip-body">${body}</div>
      `;
      document.body.appendChild(tipEl);
      const r = btn.getBoundingClientRect();
      const pad = 10;
      let x = r.right + pad;
      let y = r.top;
      const w = tipEl.offsetWidth;
      const h = tipEl.offsetHeight;
      if (x + w > window.innerWidth - 12) x = r.left - w - pad;
      if (y + h > window.innerHeight - 12) y = window.innerHeight - h - 12;
      if (y < 12) y = 12;
      tipEl.style.left = `${x}px`;
      tipEl.style.top = `${y}px`;
    }
    const findTipTarget = (el) => {
      if (!el) return null;
      let node = el.nodeType === 3 ? el.parentNode : el; // text node guard
      return node && node.closest ? node.closest("[data-tip]") : null;
    };

    // Click to pin tooltip
    document.addEventListener("click", (e) => {
      const btn = findTipTarget(e.target);
      if (btn) {
        e.preventDefault();
        showTip(btn);
        return;
      }
      if (tipEl && !e.target.closest(".tooltip")) closeTip();
    });
    // Hover support for stats/cards
    document.addEventListener("mouseenter", (e) => {
      const btn = findTipTarget(e.target);
      if (btn) showTip(btn);
    }, true);
    document.addEventListener("mouseleave", (e) => {
      const btn = findTipTarget(e.target);
      if (!btn) return;
      const related = e.relatedTarget;
      const relTip = findTipTarget(related);
      if (related && (relTip === btn || (related.closest && related.closest(".tooltip")))) return;
      closeTip();
    }, true);
    document.addEventListener("mouseleave", (e) => {
      if (e.target.classList && e.target.classList.contains("tooltip")) {
        closeTip();
      }
    }, true);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeTip();
    });
  