// Tool-Edit Review Phim Studio — Full A–Z Frontend Engine (v2.1)

const state = {
  hardware: null,
  languages: [],
  models: [],
  fonts: [],
  voices: [],
  presets: [],
  projects: [],
  activeProjectId: null,
  activeLayer: "subtitle",
  cues: [],
  saveTimer: null,
  pollTimer: null,
  batchPollTimer: null,
  dlPollTimer: null,
};

const $ = (id) => document.getElementById(id);

const FIELD_NAMES = {
  batch_size: "Số dòng mỗi lô (Batch) — Tab Dịch SRT",
  concurrency: "Số luồng song song",
  length_tolerance_pct: "Độ chênh độ dài (%)",
  font_size: "Cỡ chữ phụ đề",
  outline_width: "Viền chữ",
  ocr_sample_fps: "Tần suất quét OCR",
  blur_strength: "Độ mờ khung",
  speed: "Tốc độ đọc",
  max_atempo: "Tua tối đa",
};

function formatApiError(status, txt) {
  try {
    const data = JSON.parse(txt);
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail
        .map((d) => {
          const field = (d.loc || []).filter((x) => typeof x === "string" && x !== "body").pop() || "";
          const name = FIELD_NAMES[field] || field;
          const lim = d.ctx && (d.ctx.le ?? d.ctx.ge ?? d.ctx.lt ?? d.ctx.gt);
          return `• ${name}: giá trị không hợp lệ${lim !== undefined ? ` (giới hạn ${lim})` : ""} — ${d.msg}`;
        })
        .join("\n");
    }
  } catch {}
  return txt || `HTTP ${status}`;
}

async function api(url, opts = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const txt = await res.text();
    const err = new Error(formatApiError(res.status, txt));
    err.status = res.status;
    throw err;
  }
  return res.json();
}

function showToast(msg, kind = "err", ms = 9000) {
  const box = document.getElementById("toast-box");
  if (!box) return alert(msg);
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  el.title = "Bấm để đóng";
  el.addEventListener("click", () => el.remove());
  box.appendChild(el);
  setTimeout(() => el.remove(), ms);
}

// Every button handler is async; a failed request (e.g. 422 when saving settings, 409 when the video is busy)
// used to end as an unhandled rejection: nothing happened on screen. Show it.
window.addEventListener("unhandledrejection", (ev) => {
  const msg = ev.reason && ev.reason.message ? ev.reason.message : String(ev.reason || "Lỗi không xác định");
  showToast(`⚠ ${msg}`, "err");
});

function activeProject() {
  return state.projects.find((p) => p.id === state.activeProjectId) || null;
}

function fmtTime(sec) {
  if (!sec || isNaN(sec)) return "00:00.0";
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, "0");
  return `${String(m).padStart(2, "0")}:${s}`;
}

function mediaUrl(path) {
  return `/api/media/stream?path=${encodeURIComponent(path)}`;
}

// ---------------- INIT ----------------

function startHeartbeat() {
  // Tells the hidden launcher this window is still open; the server shuts itself down a few minutes
  // after the window is closed (never while a job is running).
  const beat = () => fetch("/api/system/heartbeat", { method: "POST" }).catch(() => {});
  beat();
  setInterval(beat, 10000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) beat(); });
}

async function init() {
  startHeartbeat();
  bindStaticEvents();
  await Promise.all([
    loadHardware(),
    loadConfigs(),
    loadTranslateBackends(),
    loadPresets(),
    loadProjects(),
  ]);
  await loadVoices("capcut_tiktok", "vi", true);
  if (state.projects.length > 0) {
    selectProject(state.projects[0].id);
  }
  window.addEventListener("resize", fitStageCanvas);
  setInterval(renderJobTimer, 500); // live stopwatch of the selected video
  initGeminiWeb();
  watchCodeUpdates();
}

// The page is read from disk on every load, the Python server is not: after a code update the UI looks new while
// the old code still runs (style guide sent the old way). Tell the user to restart the tool.
function watchCodeUpdates() {
  let warned = false;
  const check = async () => {
    if (warned) return;
    try {
      const r = await api("/api/system/code-status");
      if (r.stale) {
        warned = true;
        showToast(
          "⚠ Code của tool vừa được cập nhật nhưng tool vẫn đang chạy bản cũ. Hãy bấm Dừng (nếu đang chạy), " +
            "TẮT HẲN tool rồi mở lại để dùng bản mới. File đổi: " + r.changed.join(", "),
          "warn",
          600000
        );
      }
    } catch {}
  };
  check();
  setInterval(check, 30000);
}

async function loadHardware() {
  try {
    const hw = await api("/api/hardware");
    state.hardware = hw;
    $("hw-enc-pill").textContent = hw.recommended_encoder.toUpperCase();
    const dualTxt = hw.dual_gpu_capable ? "Dual-GPU" : `${hw.cpu_cores}C/${hw.cpu_threads}T`;
    $("hw-text").textContent = `${dualTxt} • ${hw.ram_gb}GB RAM`;
    $("hw-badge").title = `${hw.summary}\nGPUs: ${(hw.gpus || []).join(", ")}\nFFmpeg: ${hw.ffmpeg_version}`;
    // Encoder / performance dropdowns: show what "Tự động" resolves to on THIS machine, disable what it lacks
    const encNames = { h264_nvenc: "NVIDIA NVENC", h264_qsv: "Intel QSV", libx264: "CPU libx264" };
    const encSel = $("cfg-encoder");
    if (encSel) {
      for (const opt of encSel.options) {
        if (!opt.dataset.label) opt.dataset.label = opt.textContent;
        if (opt.value === "auto") {
          opt.textContent = `Tự động — máy này: ${encNames[hw.recommended_encoder] || hw.recommended_encoder} (khuyên dùng)`;
        } else {
          const ok = hw.encoders && hw.encoders[opt.value];
          opt.disabled = !ok;
          opt.textContent = opt.dataset.label + (ok ? "" : " — máy này không có");
        }
      }
    }
    const tierNames = { workstation: "Workstation (RTX)", dual_gpu_laptop: "Dual-GPU Laptop", igpu_laptop: "Laptop iGPU", cpu_only: "Chỉ CPU" };
    const perfAuto = $("cfg-perf") && $("cfg-perf").querySelector('option[value="auto"]');
    if (perfAuto) perfAuto.textContent = `Tự động — máy này: ${tierNames[hw.tier] || hw.tier} (khuyên dùng)`;
  } catch {
    $("hw-text").textContent = "Lỗi kiểm tra phần cứng";
  }
}

async function loadTranslateBackends() {
  try {
    const b = await api("/api/translate/backends");
    const tag = $("tr-backend-tag");
    if (b.agentapi?.available) {
      tag.textContent = "Antigravity Sẵn sàng (Đã đăng nhập)";
      tag.className = "tag ok";
      tag.title = `Dùng tài khoản Antigravity tại: ${b.agentapi.path}`;
    } else if (b.agy_cli?.available) {
      tag.textContent = "agy CLI Sẵn sàng";
      tag.className = "tag ok";
    } else if (b.antigravity_sdk?.available) {
      tag.textContent = "Antigravity SDK Sẵn sàng";
      tag.className = "tag ok";
    } else if (b.gemini_api_env?.available) {
      tag.textContent = "Gemini API ENV";
      tag.className = "tag ok";
    } else {
      tag.textContent = "Cần Key / agy";
      tag.className = "tag warn";
    }
  } catch {}
}

async function loadConfigs() {
  const [langRes, modelRes, fontRes, gemRes] = await Promise.all([
    api("/api/config/languages"),
    api("/api/config/models"),
    api("/api/config/fonts"),
    api("/api/config/gem"),
  ]);
  state.languages = langRes.languages || [];
  state.models = modelRes.models || [];
  state.fonts = fontRes.fonts || [];
  $("gem-preview").value = gemRes.content || "";

  // Populate latest Antigravity model selectors everywhere
  const modelOptions = state.models
    .map((m) => `<option value="${m.id}">${m.label}</option>`)
    .join("");
  $("tr-model").innerHTML = modelOptions;
  $("tr-model").value = "gemini-4-neon";
  $("clean-srt-model").innerHTML = modelOptions;
  $("dl-clean-model").innerHTML = modelOptions;
  // default for cleaning: a fast model that exists in this Antigravity (Hydrogen, else Flash, else the first)
  const fastDefault = ["gemini-4-hydrogen", "gemini-3.8-flash"].find((id) => state.models.some((m) => m.id === id)) || (state.models[0] || {}).id;
  $("clean-srt-model").value = fastDefault;
  $("dl-clean-model").value = fastDefault;
  // show which exact Antigravity model (model + thinking level) will run
  [["tr-model", "tr-thinking", "tr-model-resolved"], ["clean-srt-model", "clean-srt-thinking", "clean-model-resolved"],
   ["dl-clean-model", "dl-clean-thinking", "dl-model-resolved"]].forEach(([m, t, out]) => {
    const upd = () => showResolvedModel($(m).value, $(t).value, out);
    $(m).addEventListener("change", upd);
    $(t).addEventListener("change", upd);
    upd();
  });

  const fillLang = (sel, defVal, includeAll = false) => {
    sel.innerHTML =
      (includeAll ? `<option value="">-- Tất cả ngôn ngữ --</option>` : "") +
      state.languages
        .map((l) => `<option value="${l.code}">${l.name} (${l.native})</option>`)
        .join("");
    sel.value = defVal;
  };
  fillLang($("tr-src-lang"), "zh");
  fillLang($("tr-tgt-lang"), "vi");
  fillLang($("voice-lang"), "vi", true);

  const fontSel = $("sub-font-select");
  fontSel.innerHTML =
    `<option value="">Mặc định (Arial Bold)</option>` +
    state.fonts.map((f) => `<option value="${f.path}">${f.name}</option>`).join("");
  $("ch-font").innerHTML = fontSel.innerHTML;
}

async function loadVoices(provider, language, includeMultilingual = true) {
  try {
    const res = await api(
      `/api/voices?provider=${encodeURIComponent(provider)}&language=${encodeURIComponent(
        language || ""
      )}&include_multilingual=${includeMultilingual ? "true" : "false"}`
    );
    state.voices = res.voices || [];
    $("voice-list-num").textContent = String(state.voices.length);
    const sel = $("voice-select");
    const notice = $("voice-empty-notice");

    if (!state.voices.length) {
      sel.innerHTML = `<option value="">(Không có giọng phù hợp ở nguồn này)</option>`;
      notice.classList.remove("hidden");
      return;
    }

    notice.classList.add("hidden");
    sel.innerHTML = state.voices
      .map(
        (v) =>
          `<option value="${v.id}">[${v.lang.toUpperCase()}] ${v.name} — ${v.style || v.gender}</option>`
      )
      .join("");

    const proj = activeProject();
    if (
      proj &&
      proj.config.voice.voice_id &&
      state.voices.some((v) => v.id === proj.config.voice.voice_id)
    ) {
      sel.value = proj.config.voice.voice_id;
    } else if (state.voices.length) {
      sel.value = state.voices[0].id;
      if (proj) {
        proj.config.voice.voice_id = state.voices[0].id;
        proj.config.voice.voice_name = state.voices[0].name;
      }
    }
  } catch (e) {
    console.error("Load voices error:", e);
  }
}

async function loadPresets() {
  const res = await api("/api/presets");
  state.presets = res.presets || [];
  const sel = $("preset-select");
  sel.innerHTML = state.presets.map((p) => `<option value="${p.id}">${p.name}</option>`).join("");
}

async function loadProjects() {
  const res = await api("/api/projects");
  state.projects = res.projects || [];
  renderVideoList();
}

// Old saved ids -> Antigravity family ids ("claude-opus-5.5" -> "claude-opus-5-5", "orchestrator" -> neon orchestrator)
function normalizeModelId(id) {
  if (!id) return id;
  if (id === "orchestrator") return "gemini-pro-orchestrator-neon";
  if (id.startsWith("claude")) return id.replace(/(\d)\.(\d)/, "$1-$2");
  return id;
}

async function showResolvedModel(model, thinking, outId) {
  const el = $(outId);
  if (!el || !model) return;
  try {
    const r = await api(`/api/config/resolve-model?model=${encodeURIComponent(model)}&thinking=${encodeURIComponent(thinking || "medium")}`);
    el.textContent = `Antigravity sẽ chạy: ${r.model_id}${r.note ? ` (${r.note})` : ""}`;
  } catch {
    el.textContent = "";
  }
}

// ---------------- JOB STOPWATCH (per video: total + each step) ----------------

function escHtml(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtDur(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(sec / 3600);
  const m = String(Math.floor((sec % 3600) / 60)).padStart(2, "0");
  const s = String(sec % 60).padStart(2, "0");
  return h ? `${h}:${m}:${s}` : `${m}:${s}`;
}

function timingTotal(t) {
  if (!t || !t.started) return 0;
  return (t.ended || Date.now() / 1000) - t.started;
}

// Whole-file translation: Batch / thread settings only matter for the (rare) gap-filling pass -> hide them.
function syncWholeFileUi(on) {
  $("tr-batch-field").classList.toggle("hidden", on);
  $("tr-conc-field").classList.toggle("hidden", on);
  $("tr-whole-note").classList.toggle("hidden", !on);
}

// Provider-specific fields: Gemini Web has no model / thinking / batch / threads / key in the tool.
function syncProviderUi() {
  const web = $("tr-provider").value === "gemini_web";
  ["tr-thinking-field", "tr-model-field", "tr-model-resolved", "tr-apikey-field"].forEach((id) =>
    $(id).classList.toggle("hidden", web)
  );
  $("tr-whole-file").closest("label").classList.toggle("hidden", web);
  $("tr-web-box").classList.toggle("hidden", !web);
  $("tr-web-auto-row").classList.toggle("hidden", $("tr-web-mode").value !== "auto");
  if (web) {
    $("tr-batch-field").classList.add("hidden");
    $("tr-conc-field").classList.add("hidden");
    $("tr-whole-note").classList.add("hidden");
    updateWebPlan();
  } else {
    syncWholeFileUi($("tr-whole-file").checked);
  }
}

// "21740 dòng → 87 phần × ~250 dòng": same split rule as gemini_web.chunk_size_for
function updateWebPlan() {
  const el = $("tr-web-plan");
  if (!el) return;
  const total = state.cpTotal || 0;
  const max = Math.max(20, Number($("tr-web-chunk").value) || 250);
  if (!total) {
    el.textContent = `File ≤ ${max} dòng: gửi nguyên 1 phần · dài hơn: chia đều nhiều phần (≤ ${max} dòng/phần).`;
    return;
  }
  const parts = Math.ceil(total / max);
  const size = Math.ceil(total / parts);
  const left = Math.max(0, total - (state.cpDone || 0));
  el.textContent = parts === 1
    ? `File ${total} dòng → gửi nguyên 1 phần.`
    : `File ${total} dòng → ${parts} phần × ~${size} dòng` + (state.cpDone ? ` (còn ${left} dòng chưa dịch).` : ".");
}

// ---------------- GEMINI WEB RELAY PANEL ----------------
const GW = { id: null, info: null, min: false, lastReport: null };

function initGeminiWeb() {
  $("gw-min").addEventListener("click", () => {
    GW.min = !GW.min;
    $("gw-panel").classList.toggle("gw-min", GW.min);
    $("gw-min").textContent = GW.min ? "+" : "–";
  });
  $("gw-copy").addEventListener("click", async () => {
    const txt = await gwPrompt();
    if (txt == null) return;
    await gwCopy(txt);
    showToast(`Đã copy prompt (${txt.length.toLocaleString()} ký tự). Dán vào Gemini rồi gửi.`, "info", 4000);
  });
  $("gw-download").addEventListener("click", async () => {
    const txt = await gwPrompt();
    if (txt == null) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([txt], { type: "text/plain;charset=utf-8" }));
    const first = GW.info && GW.info.first ? `_dong_${GW.info.first}-${GW.info.last}` : "_style_guide";
    a.download = `gemini_prompt${first}.txt`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    $("gw-file-hint").classList.remove("hidden");
  });
  $("gw-copy-short").addEventListener("click", () => gwCopy("Làm đúng theo hướng dẫn trong file đính kèm."));
  $("gw-open").addEventListener("click", () => window.open("https://gemini.google.com/app", "gemini_web"));
  $("gw-paste").addEventListener("click", async () => {
    try {
      const txt = await navigator.clipboard.readText();
      if (!txt.trim()) return showToast("Clipboard đang trống. Bấm Copy dưới câu trả lời của Gemini trước.", "err", 5000);
      $("gw-result").value = txt;
    } catch {
      return showToast("Trình duyệt không cho đọc clipboard: dán (Ctrl+V) vào ô rồi bấm Kiểm tra & nhận.", "err", 6000);
    }
    gwSubmit(false);
  });
  $("gw-submit").addEventListener("click", () => gwSubmit(false));
  $("gw-partial").addEventListener("click", () => gwSubmit(true));

  $("btn-web-open").addEventListener("click", async () => {
    try {
      const r = await api("/api/gemini-web/browser/open", { method: "POST" });
      $("tr-web-status").textContent = r.already
        ? "Cửa sổ Chrome Gemini của tool đang mở."
        : "Đã mở cửa sổ Chrome Gemini. Lần đầu: đăng nhập Google trong cửa sổ đó, rồi bấm Kiểm tra kết nối.";
    } catch (e) { $("tr-web-status").textContent = e.message; }
  });
  $("btn-web-check").addEventListener("click", async () => {
    $("tr-web-status").textContent = "Đang kiểm tra...";
    try {
      const r = await api("/api/gemini-web/browser/status");
      $("tr-web-status").textContent = (r.ready ? "✅ " : "⚠ ") + r.message;
    } catch (e) { $("tr-web-status").textContent = e.message; }
  });

  const poll = async () => {
    try {
      const r = await api("/api/gemini-web/pending");
      renderGeminiWeb(r.pending);
    } catch {}
  };
  poll();
  setInterval(poll, 2000);
}

async function gwPrompt() {
  if (!GW.id) return null;
  try {
    const res = await fetch(`/api/gemini-web/prompt/${GW.id}`);
    if (!res.ok) throw new Error("Yêu cầu này không còn chờ.");
    return await res.text();
  } catch (e) {
    showToast(e.message, "err", 5000);
    return null;
  }
}

async function gwCopy(txt) {
  try {
    await navigator.clipboard.writeText(txt);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = txt;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

async function gwSubmit(partial) {
  if (!GW.id) return;
  const text = $("gw-result").value;
  if (!text.trim()) return showToast("Chưa có câu trả lời để kiểm tra.", "err", 4000);
  try {
    const r = await api("/api/gemini-web/submit", {
      method: "POST",
      body: JSON.stringify({ id: GW.id, text, accept_partial: partial }),
    });
    gwShowReport(r);
    if (r.ok) {
      $("gw-result").value = "";
      showToast(r.message, "info", 4000);
    }
  } catch (e) {
    showToast(e.message, "err", 6000);
  }
}

function gwShowReport(r) {
  GW.lastReport = r;
  const box = $("gw-report");
  box.textContent = r ? r.message : "";
  box.className = "gw-report" + (r ? (r.ok ? " ok" : " bad") : "");
  $("gw-partial").classList.toggle("hidden", !(r && !r.ok && r.good > 0));
}

function renderGeminiWeb(p) {
  const panel = $("gw-panel");
  if (!p) {
    panel.classList.add("hidden");
    GW.id = null;
    return;
  }
  const isNew = p.id !== GW.id;
  GW.id = p.id;
  GW.info = p;
  panel.classList.remove("hidden");
  $("gw-title").textContent = p.label;
  const lines = p.n_lines ? ` · ${p.n_lines} dòng (${p.first}–${p.last})` : "";
  $("gw-meta").textContent = p.attachment_name
    ? `${p.project_name} · gửi file ${p.attachment_name} + câu lệnh ${p.chars.toLocaleString()} ký tự`
    : `${p.project_name}${lines} · prompt ${p.chars.toLocaleString()} ký tự` +
      (p.chars > 60000 ? " — dài: nên dùng Tải file prompt" : "");
  const att = !!p.attachment_name;
  $("gw-attach").classList.toggle("hidden", !att);
  $("gw-download").classList.toggle("hidden", att);
  $("gw-copy").textContent = att ? "📋 Copy câu lệnh" : "📋 Copy prompt";
  if (att && isNew) {
    $("gw-attach-dl").href = `/api/gemini-web/attachment/${p.id}`;
    $("gw-attach-dl").setAttribute("download", p.attachment_name);
    $("gw-attach-dl").textContent = `⬇ Tải file ${p.attachment_name}`;
    $("gw-attach-dl-txt").href = `/api/gemini-web/attachment/${p.id}?txt=1`;
    $("gw-attach-dl-txt").setAttribute("download", `${p.attachment_name}.txt`);
  }
  const auto = p.state === "auto";
  $("gw-status").textContent = p.status || "";
  $("gw-status").className = "gw-status" + (auto ? " auto" : "");
  ["gw-copy", "gw-download", "gw-paste", "gw-submit"].forEach((id) => ($(id).disabled = auto));
  $("gw-result").disabled = auto;
  if (isNew) {
    $("gw-result").value = "";
    $("gw-file-hint").classList.add("hidden");
    gwShowReport(null);
    if (!auto) {
      if (GW.min) $("gw-min").click();
      panel.classList.remove("gw-attn");
      void panel.offsetWidth;
      panel.classList.add("gw-attn");
    }
  } else if (p.report && !GW.lastReport && !auto) {
    gwShowReport(p.report);
  }
}

function renderJobTimer() {
  const box = $("job-timer");
  if (!box) return;
  const p = activeProject();
  const t = p && p.timing;
  if (!t || !t.started) {
    box.classList.add("hidden");
    return;
  }
  const running = !t.ended;
  const now = Date.now() / 1000;
  const label = running ? "đang chạy" : t.ok ? "xong" : t.note || "lỗi";
  const steps = (t.steps || [])
    .map((st) => {
      const end = st.end || (running ? now : st.start);
      const ico = st.status === "running" ? "●" : st.status === "error" ? "✕" : "✓";
      const cls = `${st.status || "done"}${st.key === "wait" ? " wait" : ""}`;
      return `<div class="jt-step ${cls}"><span class="jt-ico">${ico}</span><span>${escHtml(st.label)}</span><span class="jt-dur">${fmtDur(end - st.start)}</span>${
        st.note ? `<span class="jt-note">${escHtml(st.note)}</span>` : ""
      }</div>`;
    })
    .join("");
  const html =
    `<div class="jt-head"><span class="jt-title">⏱ ${escHtml(t.title)} · ${escHtml(label)}</span>` +
    `<span class="jt-total" title="Tổng thời gian chạy 1 video (tính cả các bước chạy song song)">${fmtDur(timingTotal(t))}</span></div>${steps}`;
  box.classList.remove("hidden");
  box.classList.toggle("running", running);
  box.classList.toggle("failed", !running && t.ok === false);
  if (box._last !== html) {
    box.innerHTML = html;
    box._last = html;
  }
}

// ---------------- VIDEO LIST ----------------

const BUSY_STATUSES = ["downloading", "translating", "voicing", "cleaning", "rendering"];

function updateStopButton() {
  const btn = $("btn-stop-job");
  if (!btn) return;
  const cur = state.projects.find((p) => p.id === state.activeProjectId);
  const show = state.batchRunning || (cur && BUSY_STATUSES.includes(cur.status));
  btn.classList.toggle("hidden", !show);
  if (!show) btn.disabled = false;
}

function renderVideoList() {
  updateStopButton();
  $("video-count").textContent = String(state.projects.length);
  const container = $("video-list");
  if (!state.projects.length) {
    container.innerHTML = `<div style="padding:20px 10px;text-align:center;color:var(--text-2);font-size:12px;">Chưa có video nào.<br/>Bấm "+ Thêm File" hoặc "⬇ Link YT/Bili".</div>`;
    return;
  }

  container.innerHTML = state.projects
    .map((p) => {
      const isActive = p.id === state.activeProjectId;
      const srtTag = p.translated_srt_path
        ? `<span class="tag ok">Sub đã dịch</span>`
        : p.srt_path
        ? `<span class="tag">SRT gốc</span>`
        : `<span class="tag warn">Chưa SRT</span>`;
      const voiceTag = p.dub_audio_path ? `<span class="tag ok">Có Voice</span>` : "";
      const bgmTag = p.config?.bgm?.enabled && p.config?.bgm?.file_paths?.length ? `<span class="tag">🎵 BGM</span>` : "";
      const busy = ["downloading", "translating", "voicing", "cleaning", "rendering"].includes(p.status);
      const timeTag =
        p.timing && p.timing.ended
          ? `<span class="tag ${p.timing.ok ? "ok" : "err"}" title="${escHtml(p.timing.title)}: tổng thời gian lần chạy gần nhất">⏱ ${fmtDur(timingTotal(p.timing))}</span>`
          : "";
      const stTag =
        p.status === "done"
          ? `<span class="tag ok">Đã xuất</span>`
          : busy
          ? `<span class="tag warn">${p.status} ${Math.round(p.progress)}%</span>`
          : p.status === "error"
          ? `<span class="tag err">Lỗi</span>`
          : "";
      return `
      <div class="video-card ${isActive ? "active" : ""}" data-id="${p.id}">
        <div class="video-card-title" title="${p.video_path}">${p.name}</div>
        <div class="video-card-meta">
          <span class="tag">${p.width}×${p.height}</span>
          <span class="tag">${fmtTime(p.duration)}</span>
          ${srtTag}
          ${voiceTag}
          ${bgmTag}
          ${stTag}
          ${timeTag}
        </div>
        ${
          busy || p.progress > 0
            ? `<div class="progress-mini"><div class="progress-mini-fill" style="width:${p.progress}%"></div></div>`
            : ""
        }
        <div class="video-card-actions">
          <button class="btn btn-sm btn-open-cfg" data-id="${p.id}">⚙ Cấu hình</button>
          <button class="btn btn-sm btn-danger btn-del-proj" data-id="${p.id}">Xóa</button>
        </div>
      </div>`;
    })
    .join("");

  container.querySelectorAll(".video-card").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (e.target.closest(".btn-del-proj")) return;
      selectProject(el.dataset.id);
    });
  });

  container.querySelectorAll(".btn-del-proj").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      await api(`/api/projects/${id}`, { method: "DELETE" });
      state.projects = state.projects.filter((p) => p.id !== id);
      if (state.activeProjectId === id) {
        state.activeProjectId = state.projects[0]?.id || null;
        if (state.activeProjectId) selectProject(state.activeProjectId);
        else clearStage();
      }
      renderVideoList();
    });
  });
}

function clearStage() {
  $("empty-stage").classList.remove("hidden");
  $("video-canvas-wrap").classList.add("hidden");
  $("stage-meta").textContent = "Chưa chọn video";
  $("preview-video").src = "";
}

async function selectProject(id) {
  state.activeProjectId = id;
  renderVideoList();
  const proj = activeProject();
  if (!proj) return clearStage();

  // Ensure bgm & new voice fields exist on older saved projects
  if (!proj.config.bgm) {
    proj.config.bgm = {
      enabled: false,
      file_paths: [],
      volume: 0.22,
      crossfade_sec: 3.5,
      auto_ducking: true,
      duck_level: 0.08,
      fade_in_sec: 1.5,
      fade_out_sec: 2.5,
    };
  }
  if (!proj.config.voice.concurrency) proj.config.voice.concurrency = 12;
  if (proj.config.voice.include_multilingual === undefined) proj.config.voice.include_multilingual = true;

  $("empty-stage").classList.add("hidden");
  $("video-canvas-wrap").classList.remove("hidden");
  $("stage-meta").textContent = `${proj.name} • ${proj.width}×${proj.height} • ${proj.fps}fps • ${fmtTime(proj.duration)}`;

  const vid = $("preview-video");
  vid.src = mediaUrl(proj.video_path);
  vid.load();
  $("btn-play").textContent = "▶ Phát";

  await Promise.all([loadProjectSubtitles(proj.id), loadStyleGuideInfo(proj.id)]);
  populateInspectorFromProject(proj);
  await loadVoices(
    proj.config.voice.provider,
    proj.config.voice.language,
    proj.config.voice.include_multilingual
  );
  fitStageCanvas();
  syncAllOverlays();

  if (["downloading", "translating", "voicing", "cleaning", "rendering"].includes(proj.status)) {
    startStatusPolling(proj.id);
  } else {
    $("render-status-text").textContent = proj.status_message || "Sẵn sàng chỉnh sửa & xuất video.";
    $("render-pct").textContent = proj.status === "done" ? "100%" : "";
  }
}

async function loadProjectSubtitles(projectId) {
  try {
    const res = await api(`/api/projects/${projectId}/subtitles`);
    state.cues = res.cues || [];
  } catch {
    state.cues = [];
  }
}

async function loadStyleGuideInfo(projectId) {
  try {
    const res = await api(`/api/projects/${projectId}/style-guide`);
    $("style-guide-editor").value = res.content || "";
    state.cpTotal = res.checkpoint_total || 0;
    state.cpDone = res.checkpoint_done || 0;
    updateWebPlan();
    $("tr-checkpoint-info").textContent =
      res.checkpoint_total > 0
        ? `Checkpoint: Đã dịch ${res.checkpoint_done}/${res.checkpoint_total} dòng`
        : "Checkpoint: Chưa dịch";
  } catch {}
}

// ---------------- BGM PLAYLIST RENDERER ----------------

function renderBgmList() {
  const proj = activeProject();
  const box = $("bgm-list");
  if (!proj || !proj.config.bgm?.file_paths?.length) {
    box.innerHTML = `<div style="font-size:11px;color:var(--text-2);padding:4px 2px;">Chưa chọn bài nhạc nền nào.</div>`;
    return;
  }
  box.innerHTML = proj.config.bgm.file_paths
    .map((fp, idx) => {
      const fname = fp.split(/[\\/]/).pop();
      return `
      <div class="bgm-item">
        <span title="${fp}">${idx + 1}. ${fname}</span>
        <button class="btn-rm-bgm" data-idx="${idx}" title="Bỏ bài này">✕</button>
      </div>`;
    })
    .join("");

  box.querySelectorAll(".btn-rm-bgm").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.idx);
      proj.config.bgm.file_paths.splice(idx, 1);
      renderBgmList();
      renderVideoList();
      scheduleAutoSave();
    });
  });
}

// ---------------- STAGE SIZING & OVERLAY RENDERING ----------------

function fitStageCanvas() {
  const proj = activeProject();
  if (!proj) return;
  const vp = $("stage-viewport");
  const wrap = $("video-canvas-wrap");
  const maxW = Math.max(200, vp.clientWidth - 36);
  const maxH = Math.max(200, vp.clientHeight - 36);
  const ratio = proj.width / Math.max(1, proj.height);

  let w = maxW;
  let h = w / ratio;
  if (h > maxH) {
    h = maxH;
    w = h * ratio;
  }
  wrap.style.width = `${Math.round(w)}px`;
  wrap.style.height = `${Math.round(h)}px`;
  syncSubtitleVisual();
}

function applyBoxStyle(el, box) {
  el.style.left = `${(box.x * 100).toFixed(3)}%`;
  el.style.top = `${(box.y * 100).toFixed(3)}%`;
  el.style.width = `${(box.w * 100).toFixed(3)}%`;
  el.style.height = `${(box.h * 100).toFixed(3)}%`;
}

function hexToRgba(hex, alpha) {
  const h = (hex || "#000000").replace("#", "");
  const r = parseInt(h.substring(0, 2), 16) || 0;
  const g = parseInt(h.substring(2, 4), 16) || 0;
  const b = parseInt(h.substring(4, 6), 16) || 0;
  return `rgba(${r},${g},${b},${alpha})`;
}

function getCurrentSubtitleText() {
  const proj = activeProject();
  if (!proj) return "";
  const t = $("preview-video").currentTime || 0;
  const cue = state.cues.find((c) => t >= c.start_sec && t <= c.end_sec);
  if (cue) return cue.text;
  if (state.cues.length > 0 && t < 0.2) return state.cues[0].text;
  return proj.config.subtitle.sample_text || "Phụ đề xem trước (kéo khung W×H tự do)";
}

const loadedFontFaceMap = new Map();

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function cleanUiText(text) {
  if (!text) return "";
  return String(text)
    .replace(/\x1b\[[0-9;]*[a-zA-Z]?/g, "")
    .replace(/\[[0-9;]+m/g, "")
    .replace(/^\s*ERROR:\s*/i, "")
    .trim();
}

async function applySubtitleFont(fontPath) {
  return applyFontTo($("ov-subtitle-inner"), fontPath);
}

async function applyFontTo(inner, fontPath) {
  if (!inner) return;

  if (!fontPath) {
    inner.style.fontFamily = '"Arial", "Segoe UI", -apple-system, sans-serif';
    return;
  }

  const match = (state.fonts || []).find((f) => f.path.toLowerCase() === fontPath.toLowerCase());
  const friendlyName = match ? match.name.replace(/\s+(Bold|Regular|Italic)$/i, "") : "";
  const fontKey = "CustomSubFont_" + hashString(fontPath);

  if (!loadedFontFaceMap.has(fontPath)) {
    try {
      const url = `/api/media/stream?path=${encodeURIComponent(fontPath)}`;
      const fontFace = new FontFace(fontKey, `url("${url}")`);
      await fontFace.load();
      document.fonts.add(fontFace);
      loadedFontFaceMap.set(fontPath, fontKey);
    } catch (e) {
      console.warn("Could not load dynamic FontFace:", fontPath, e);
      loadedFontFaceMap.set(fontPath, friendlyName || "Arial");
    }
  }

  const fam = loadedFontFaceMap.get(fontPath) || fontKey;
  inner.style.fontFamily = `"${fam}", "${friendlyName}", "Arial", "Segoe UI", sans-serif`;
}

function syncSubtitleVisual() {
  const proj = activeProject();
  if (!proj) return;
  const cfg = proj.config.subtitle;
  const boxEl = $("ov-subtitle");
  const inner = $("ov-subtitle-inner");

  boxEl.classList.toggle("hidden", !cfg.enabled);
  if (!cfg.enabled) return;

  applyBoxStyle(boxEl, cfg.box);

  const stageH = $("video-canvas-wrap").clientHeight || 540;
  const scale = stageH / 1080;

  const txt = getCurrentSubtitleText();
  if (inner.textContent !== txt || inner.dataset.fx !== (cfg.effect || "none")) {
    inner.textContent = txt;
    inner.dataset.fx = cfg.effect || "none";
    inner.className = "";
    if (cfg.effect && cfg.effect !== "none") {
      void inner.offsetWidth; // restart the CSS animation
      inner.style.setProperty("--fx-dur", `${cfg.effect_duration || 0.3}s`);
      inner.className = `sub-fx-${cfg.effect}`;
    }
  }
  inner.style.fontSize = `${Math.max(10, Math.round(cfg.font_size * scale))}px`;
  inner.style.fontWeight = cfg.bold ? "700" : "400";
  inner.style.color = cfg.text_color;
  inner.style.textAlign = cfg.align;
  inner.style.justifyContent =
    cfg.valign === "top" ? "flex-start" : cfg.valign === "bottom" ? "flex-end" : "center";
  inner.style.padding = `${Math.round(cfg.padding_y * scale)}px ${Math.round(cfg.padding_x * scale)}px`;
  inner.style.borderRadius = `${Math.round(cfg.border_radius * scale)}px`;
  inner.style.background = cfg.bg_enabled ? hexToRgba(cfg.bg_color, cfg.bg_opacity) : "transparent";

  const ow = Math.max(0, cfg.outline_width * scale);
  inner.style.webkitTextStroke = ow > 0 ? `${ow.toFixed(1)}px ${cfg.outline_color}` : "0px transparent";
  inner.style.paintOrder = "stroke fill";

  // Apply selected font family to preview canvas
  applySubtitleFont(cfg.font_path);
}

function syncLogoVisual() {
  const proj = activeProject();
  if (!proj) return;
  const cfg = proj.config.logo;
  const el = $("ov-logo");
  const img = $("ov-logo-img");
  const show = cfg.enabled && Boolean(cfg.file_path);
  el.classList.toggle("hidden", !show);
  if (!show) return;
  applyBoxStyle(el, cfg.box);
  img.style.opacity = String(cfg.opacity);
  const srcUrl = mediaUrl(cfg.file_path);
  if (img.dataset.rawPath !== cfg.file_path) {
    img.dataset.rawPath = cfg.file_path;
    img.src = srcUrl;
  }
}

function syncSubRemovalVisual() {
  const proj = activeProject();
  if (!proj) return;
  const cfg = proj.config.sub_removal;
  const el = $("ov-subremoval");
  el.classList.toggle("hidden", cfg.mode === "none");
  el.classList.toggle("mode-blur", cfg.mode === "blur");
  el.classList.toggle("mode-ai", cfg.mode === "ai_clean");
  if (cfg.mode !== "none") {
    applyBoxStyle(el, cfg.box);
    if (cfg.mode === "blur") {
      const px = Math.max(2, Math.round(cfg.blur_strength * 0.45));
      el.style.backdropFilter = `blur(${px}px)`;
      el.style.webkitBackdropFilter = `blur(${px}px)`;
    } else {
      el.style.backdropFilter = "none";
    }
  }
}

function syncGreenScreenVisual() {
  const proj = activeProject();
  if (!proj) return;
  const cfg = proj.config.green_screen;
  const el = $("ov-greenscreen");
  const gsVid = $("gs-source-video");
  const show = cfg.enabled && Boolean(cfg.file_path);
  el.classList.toggle("hidden", !show);
  if (!show) {
    gsVid.pause();
    return;
  }
  applyBoxStyle(el, cfg.box);
  if (gsVid.dataset.rawPath !== cfg.file_path) {
    gsVid.dataset.rawPath = cfg.file_path;
    gsVid.src = mediaUrl(cfg.file_path);
    gsVid.loop = true;
    gsVid.load();
    gsVid.onloadeddata = () => {
      gsVid.currentTime = Math.min(0.8, gsVid.duration / 2 || 0);
      drawGreenScreenFrame();
    };
  } else {
    drawGreenScreenFrame();
  }
}

function drawGreenScreenFrame() {
  const proj = activeProject();
  if (!proj || !proj.config.green_screen.enabled) return;
  const cfg = proj.config.green_screen;
  const gsVid = $("gs-source-video");
  const canvas = $("ov-gs-canvas");
  if (gsVid.readyState < 2) return;

  const w = 320;
  const h = Math.max(120, Math.round((gsVid.videoHeight / Math.max(1, gsVid.videoWidth)) * w));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(gsVid, 0, 0, w, h);

  const imgData = ctx.getImageData(0, 0, w, h);
  const data = imgData.data;
  const hex = (cfg.chroma_color || "#00ff00").replace("#", "");
  const kr = parseInt(hex.substring(0, 2), 16) || 0;
  const kg = parseInt(hex.substring(2, 4), 16) || 255;
  const kb = parseInt(hex.substring(4, 6), 16) || 0;
  const maxDist = cfg.similarity * 442;
  const blendRange = Math.max(1, cfg.blend * 442);

  for (let i = 0; i < data.length; i += 4) {
    const dr = data[i] - kr;
    const dg = data[i + 1] - kg;
    const db = data[i + 2] - kb;
    const dist = Math.sqrt(dr * dr + dg * dg + db * db);
    if (dist < maxDist) {
      data[i + 3] = 0;
    } else if (dist < maxDist + blendRange) {
      data[i + 3] = Math.round(((dist - maxDist) / blendRange) * 255);
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

function syncAudioPreview() {
  const proj = activeProject();
  if (!proj) return;
  const vid = $("preview-video");
  const a = proj.config.audio;
  vid.muted = a.mute_original || a.original_volume <= 0.001;
  vid.volume = Math.min(1.0, Math.max(0.0, a.original_volume));
}

function syncChannelVisual() {
  const proj = activeProject();
  if (!proj) return;
  const cfg = proj.config.channel_text;
  const el = $("ov-channel");
  const inner = $("ov-channel-inner");
  if (!cfg) return;
  el.classList.toggle("hidden", !cfg.enabled);
  if (!cfg.enabled) return;
  applyBoxStyle(el, cfg.box);
  const scale = ($("video-canvas-wrap").clientHeight || 540) / 1080;
  inner.textContent = cfg.text || " ";
  inner.style.fontSize = `${Math.max(8, Math.round(cfg.font_size * scale))}px`;
  inner.style.fontWeight = cfg.bold ? "700" : "400";
  inner.style.color = cfg.text_color;
  inner.style.textAlign = cfg.align || "center";
  inner.style.padding = `${Math.round(cfg.padding_y * scale)}px ${Math.round(cfg.padding_x * scale)}px`;
  inner.style.borderRadius = `${Math.round(cfg.border_radius * scale)}px`;
  inner.style.background = cfg.bg_enabled ? hexToRgba(cfg.bg_color, cfg.bg_opacity) : "transparent";
  const ow = Math.max(0, cfg.outline_width * scale);
  inner.style.webkitTextStroke = ow > 0 ? `${ow.toFixed(1)}px ${cfg.outline_color}` : "0px transparent";
  inner.style.paintOrder = "stroke fill";
  inner.style.setProperty("--fx-period", `${cfg.effect_period_sec || 8}s`);
  const fxClass = cfg.effect && cfg.effect !== "none" ? `ch-fx-${cfg.effect}` : "";
  if (inner.dataset.fx !== fxClass) {
    inner.className = "";
    void inner.offsetWidth;
    inner.className = fxClass;
    inner.dataset.fx = fxClass;
  }
  applyFontTo(inner, cfg.font_path);
}

function syncWmVisual() {
  const proj = activeProject();
  const cont = $("ov-wm-container");
  if (!proj || !cont) return;
  const cfg = proj.config.wm_removal;
  const boxes = cfg && cfg.enabled ? cfg.boxes || [] : [];
  if (cont.children.length !== boxes.length) {
    cont.innerHTML = boxes
      .map(
        (_, i) => `<div class="ov-box ov-wm" data-layer="wm" data-idx="${i}">
          <span class="ov-label">Chữ logo #${i + 1}</span>
          <div class="handle nw" data-dir="nw"></div><div class="handle n" data-dir="n"></div><div class="handle ne" data-dir="ne"></div>
          <div class="handle w" data-dir="w"></div><div class="handle e" data-dir="e"></div>
          <div class="handle sw" data-dir="sw"></div><div class="handle s" data-dir="s"></div><div class="handle se" data-dir="se"></div>
        </div>`
      )
      .join("");
    selectLayer(state.activeLayer, state.activeWmIdx);
  }
  Array.from(cont.children).forEach((el, i) => {
    applyBoxStyle(el, boxes[i]);
    el.classList.toggle("mode-blur", cfg.mode === "blur");
  });
}

function renderWmList() {
  const proj = activeProject();
  const list = $("wm-list");
  if (!proj || !list) return;
  const boxes = proj.config.wm_removal.boxes || [];
  list.innerHTML = boxes.length
    ? boxes
        .map(
          (b, i) => `<div class="wm-row ${state.activeLayer === "wm" && state.activeWmIdx === i ? "active" : ""}" data-idx="${i}">
            <span>Vùng #${i + 1} — X ${b.x.toFixed(2)} · Y ${b.y.toFixed(2)} · ${b.w.toFixed(2)}×${b.h.toFixed(2)}</span>
            <button class="btn btn-sm btn-danger btn-wm-del" data-idx="${i}">Xóa</button>
          </div>`
        )
        .join("")
    : `<div style="font-size:11.5px;color:var(--text-2);margin-bottom:6px;">Chưa có vùng nào — bấm "+ Thêm vùng chữ logo".</div>`;
}

function syncAllOverlays() {
  syncSubtitleVisual();
  syncLogoVisual();
  syncGreenScreenVisual();
  syncSubRemovalVisual();
  syncWmVisual();
  syncChannelVisual();
  syncAudioPreview();
}

function selectLayer(layerName, idx) {
  state.activeLayer = layerName;
  if (layerName === "wm") state.activeWmIdx = Number.isInteger(idx) ? idx : state.activeWmIdx || 0;
  document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c.dataset.layer === layerName));
  document.querySelectorAll(".ov-box").forEach((b) =>
    b.classList.toggle(
      "selected",
      b.dataset.layer === layerName && (layerName !== "wm" || Number(b.dataset.idx) === state.activeWmIdx)
    )
  );
  if (layerName === "wm") renderWmList();
}

// ---------------- DRAG & 8-HANDLE INDEPENDENT WxH RESIZE ----------------

function initOverlayDragResize() {
  let dragState = null;

  const getBoxRef = (layer, idx) => {
    const proj = activeProject();
    if (!proj) return null;
    if (layer === "subtitle") return proj.config.subtitle.box;
    if (layer === "logo") return proj.config.logo.box;
    if (layer === "greenscreen") return proj.config.green_screen.box;
    if (layer === "subremoval") return proj.config.sub_removal.box;
    if (layer === "channel") return proj.config.channel_text.box;
    if (layer === "wm") return (proj.config.wm_removal.boxes || [])[idx] || null;
    return null;
  };

  // one delegated listener: also covers boxes added later (text-logo regions)
  $("video-canvas-wrap").addEventListener("mousedown", (e) => {
    const boxEl = e.target.closest(".ov-box");
    if (!boxEl) return;
    e.preventDefault();
    e.stopPropagation();
    const layer = boxEl.dataset.layer;
    const idx = boxEl.dataset.idx !== undefined ? Number(boxEl.dataset.idx) : undefined;
    selectLayer(layer, idx);
    const boxRef = getBoxRef(layer, idx);
    if (!boxRef) return;

    const wrap = $("video-canvas-wrap").getBoundingClientRect();
    const handle = e.target.closest(".handle");
    dragState = {
      layer,
      idx,
      mode: handle ? "resize" : "move",
      dir: handle ? handle.dataset.dir : "",
      startX: e.clientX,
      startY: e.clientY,
      orig: { ...boxRef },
      wrapW: wrap.width,
      wrapH: wrap.height,
    };
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragState) return;
    const boxRef = getBoxRef(dragState.layer, dragState.idx);
    if (!boxRef) return;

    const dx = (e.clientX - dragState.startX) / dragState.wrapW;
    const dy = (e.clientY - dragState.startY) / dragState.wrapH;
    const o = dragState.orig;
    const minW = 0.04;
    const minH = 0.04;

    if (dragState.mode === "move") {
      boxRef.x = Math.max(0, Math.min(1 - o.w, o.x + dx));
      boxRef.y = Math.max(0, Math.min(1 - o.h, o.y + dy));
    } else {
      let nx = o.x, ny = o.y, nw = o.w, nh = o.h;
      const d = dragState.dir;
      if (d.includes("e")) nw = Math.max(minW, Math.min(1 - o.x, o.w + dx));
      if (d.includes("s")) nh = Math.max(minH, Math.min(1 - o.y, o.h + dy));
      if (d.includes("w")) {
        const clampedDx = Math.max(-o.x, Math.min(o.w - minW, dx));
        nx = o.x + clampedDx;
        nw = o.w - clampedDx;
      }
      if (d.includes("n")) {
        const clampedDy = Math.max(-o.y, Math.min(o.h - minH, dy));
        ny = o.y + clampedDy;
        nh = o.h - clampedDy;
      }
      boxRef.x = Number(nx.toFixed(4));
      boxRef.y = Number(ny.toFixed(4));
      boxRef.w = Number(nw.toFixed(4));
      boxRef.h = Number(nh.toFixed(4));
    }

    if (dragState.layer === "subtitle") {
      updateSubCoordInputs(boxRef);
      syncSubtitleVisual();
    } else if (dragState.layer === "logo") {
      syncLogoVisual();
    } else if (dragState.layer === "greenscreen") {
      syncGreenScreenVisual();
    } else if (dragState.layer === "subremoval") {
      syncSubRemovalVisual();
    } else if (dragState.layer === "channel") {
      syncChannelVisual();
    } else if (dragState.layer === "wm") {
      syncWmVisual();
      renderWmList();
    }
  });

  window.addEventListener("mouseup", () => {
    if (dragState) {
      dragState = null;
      scheduleAutoSave();
    }
  });
}

function updateSubCoordInputs(box) {
  $("sub-x").value = box.x.toFixed(3);
  $("sub-y").value = box.y.toFixed(3);
  $("sub-w").value = box.w.toFixed(3);
  $("sub-h").value = box.h.toFixed(3);
}

// ---------------- INSPECTOR BINDING ----------------

function populateInspectorFromProject(proj) {
  const c = proj.config;
  $("proj-srt-path").value = proj.translated_srt_path || proj.srt_path || "";

  // Subtitle
  $("sub-enabled").checked = c.subtitle.enabled;
  if (c.subtitle.font_path) {
    let opt = Array.from($("sub-font-select").options).find(
      (o) => o.value.toLowerCase() === c.subtitle.font_path.toLowerCase()
    );
    if (!opt) {
      opt = document.createElement("option");
      opt.value = c.subtitle.font_path;
      opt.textContent = c.subtitle.font_path.split(/[\\/]/).pop();
      $("sub-font-select").appendChild(opt);
    }
    $("sub-font-select").value = opt.value;
  } else {
    $("sub-font-select").value = "";
  }
  $("sub-size").value = c.subtitle.font_size;
  $("val-sub-size").textContent = `${c.subtitle.font_size}px`;
  $("sub-outline").value = c.subtitle.outline_width;
  $("val-sub-outline").textContent = `${c.subtitle.outline_width}px`;
  $("sub-color").value = c.subtitle.text_color;
  $("sub-outline-color").value = c.subtitle.outline_color;
  $("sub-bg-enabled").checked = c.subtitle.bg_enabled;
  $("sub-bold").checked = c.subtitle.bold;
  $("sub-bg-color").value = c.subtitle.bg_color;
  $("sub-bg-opacity").value = c.subtitle.bg_opacity;
  $("val-sub-bg-op").textContent = `${Math.round(c.subtitle.bg_opacity * 100)}%`;
  $("sub-radius").value = c.subtitle.border_radius;
  $("val-sub-radius").textContent = `${c.subtitle.border_radius}px`;
  $("sub-align").value = c.subtitle.align;
  updateSubCoordInputs(c.subtitle.box);

  // BGM
  $("bgm-enabled").checked = Boolean(c.bgm?.enabled);
  $("bgm-volume").value = c.bgm?.volume ?? 0.22;
  $("val-bgm-vol").textContent = `${Math.round((c.bgm?.volume ?? 0.22) * 100)}%`;
  $("bgm-crossfade").value = c.bgm?.crossfade_sec ?? 3.5;
  $("val-bgm-xfade").textContent = `${Number(c.bgm?.crossfade_sec ?? 3.5).toFixed(1)}s`;
  $("bgm-ducking").checked = c.bgm?.auto_ducking ?? true;
  renderBgmList();

  // Logo
  $("logo-enabled").checked = c.logo.enabled;
  $("logo-file-path").value = c.logo.file_path || "";
  $("logo-folder-path").value = c.logo.folder_path || "";
  $("logo-opacity").value = c.logo.opacity;
  $("val-logo-op").textContent = `${Math.round(c.logo.opacity * 100)}%`;
  if (c.logo.folder_path) loadLogoFolderGrid(c.logo.folder_path);
  else $("logo-folder-grid").classList.add("hidden");

  // Green screen
  $("gs-enabled").checked = c.green_screen.enabled;
  $("gs-file-path").value = c.green_screen.file_path || "";
  $("gs-color").value = c.green_screen.chroma_color;
  $("gs-start").value = c.green_screen.start_sec;
  $("gs-sim").value = c.green_screen.similarity;
  $("val-gs-sim").textContent = Number(c.green_screen.similarity).toFixed(2);
  $("gs-blend").value = c.green_screen.blend;
  $("val-gs-blend").textContent = Number(c.green_screen.blend).toFixed(2);
  $("gs-loop").checked = c.green_screen.loop_enabled;

  // Subtitle effect
  $("sub-effect").value = c.subtitle.effect || "none";
  $("sub-effect-dur").value = c.subtitle.effect_duration || 0.3;
  $("val-sub-fx-dur").textContent = `${Number(c.subtitle.effect_duration || 0.3).toFixed(2)}s`;

  // Text-logo / watermark removal
  $("wm-enabled").checked = !!c.wm_removal.enabled;
  $("wm-mode").value = c.wm_removal.mode;
  $("wm-blur").value = c.wm_removal.blur_strength;
  $("val-wm-blur").textContent = String(c.wm_removal.blur_strength);
  $("wm-blur-opts").classList.toggle("hidden", c.wm_removal.mode !== "blur");
  renderWmList();

  // Channel name
  {
    const ch = c.channel_text;
    $("ch-enabled").checked = !!ch.enabled;
    $("ch-text").value = ch.text || "";
    if (ch.font_path && !Array.from($("ch-font").options).some((o) => o.value === ch.font_path)) {
      const o = document.createElement("option");
      o.value = ch.font_path;
      o.textContent = ch.font_path.split(/[\\/]/).pop();
      $("ch-font").appendChild(o);
    }
    $("ch-font").value = ch.font_path || "";
    $("ch-size").value = ch.font_size;
    $("val-ch-size").textContent = `${ch.font_size}px`;
    $("ch-outline").value = ch.outline_width;
    $("val-ch-outline").textContent = `${ch.outline_width}px`;
    $("ch-color").value = ch.text_color;
    $("ch-outline-color").value = ch.outline_color;
    $("ch-bg-enabled").checked = !!ch.bg_enabled;
    $("ch-bold").checked = !!ch.bold;
    $("ch-bg-color").value = ch.bg_color;
    $("ch-bg-opacity").value = ch.bg_opacity;
    $("val-ch-bg-op").textContent = `${Math.round(ch.bg_opacity * 100)}%`;
    $("ch-radius").value = ch.border_radius;
    $("val-ch-radius").textContent = `${ch.border_radius}px`;
    $("ch-effect").value = ch.effect || "none";
    $("ch-period").value = ch.effect_period_sec || 8;
    $("ch-period-field").classList.toggle("hidden", !["shimmer", "slide_in", "typewriter"].includes(ch.effect));
  }

  // Sub removal
  $("sr-mode").value = c.sub_removal.mode;
  $("sr-blur-strength").value = c.sub_removal.blur_strength;
  $("val-sr-blur").textContent = String(c.sub_removal.blur_strength);
  $("sr-ai-engine").value = c.sub_removal.ai_engine;
  $("sr-auto-ocr").checked = c.sub_removal.auto_detect_ocr;
  {
    const fpsOpts = ["2", "3", "4", "6"];
    const cur = String(Math.round(c.sub_removal.ocr_sample_fps || 4));
    $("sr-ocr-fps").value = fpsOpts.includes(cur) ? cur : "4";
  }
  $("sr-blur-opts").classList.toggle("hidden", c.sub_removal.mode !== "blur");
  $("sr-ai-opts").classList.toggle("hidden", c.sub_removal.mode !== "ai_clean");

  // Audio & Performance
  $("audio-edit-mode").value = c.audio.edit_mode;
  $("audio-orig-vol").value = c.audio.original_volume;
  $("val-orig-vol").textContent = `${Math.round(c.audio.original_volume * 100)}%`;
  $("audio-max-atempo").value = c.audio.max_atempo;
  $("val-max-atempo").textContent = `${Number(c.audio.max_atempo).toFixed(2)}x`;
  $("audio-mute").checked = c.audio.mute_original;
  $("audio-ducking").checked = c.audio.auto_ducking;
  $("cfg-encoder").value = c.encoder_override;
  $("cfg-perf").value = c.performance_profile;

  // Translate
  $("tr-src-lang").value = c.translate.source_lang;
  $("tr-tgt-lang").value = c.translate.target_lang === "tl" ? "fil" : c.translate.target_lang;
  $("tr-provider").value = c.translate.provider;
  c.translate.model = normalizeModelId(c.translate.model);
  if (state.models.some((m) => m.id === c.translate.model)) {
    $("tr-model").value = c.translate.model;
  } else {
    $("tr-model").value = "gemini-4-neon";
    c.translate.model = "gemini-4-neon";
  }
  $("tr-thinking").value = c.translate.thinking_level === "max" ? "xhigh" : c.translate.thinking_level;
  showResolvedModel($("tr-model").value, $("tr-thinking").value, "tr-model-resolved");
  $("tr-batch").value = c.translate.batch_size;
  $("tr-whole-file").checked = !!c.translate.whole_file;
  syncWholeFileUi(!!c.translate.whole_file);
  $("tr-tol").value = c.translate.length_tolerance_pct;
  $("tr-concurrency").value = c.translate.concurrency || 4;
  $("tr-api-key").value = c.translate.api_key || "";
  $("tr-web-mode").value = c.translate.web_mode || "manual";
  $("tr-web-chunk").value = c.translate.web_chunk_lines || 250;
  $("tr-web-newchat").checked = c.translate.web_new_chat ?? true;
  syncProviderUi();

  // Voice
  $("voice-provider").value = c.voice.provider;
  $("voice-lang").value = c.voice.language === "tl" ? "fil" : (c.voice.language || "");
  $("voice-include-multi").checked = c.voice.include_multilingual ?? true;
  $("voice-speed").value = c.voice.speed;
  $("val-voice-speed").textContent = `${Number(c.voice.speed).toFixed(2)}x`;
  $("voice-concurrency").value = c.voice.concurrency || 12;
  $("val-voice-conc").textContent = `${c.voice.concurrency || 12} luồng`;
  $("voice-tiktok-sid").value = c.voice.tiktok_session_id || "";
  $("voice-fallback-edge").checked = c.voice.fallback_edge_tts;
  applyHiggsConfig(c);
  if (proj) {
    refreshLengthReport(proj.id);
    refreshReviewButton(proj.id);
  }
}

async function loadLogoFolderGrid(folderPath) {
  const grid = $("logo-folder-grid");
  if (!folderPath) {
    grid.classList.add("hidden");
    return;
  }
  const res = await api(`/api/system/logos?folder=${encodeURIComponent(folderPath)}`);
  const logos = res.logos || [];
  if (!logos.length) {
    grid.classList.add("hidden");
    return;
  }
  const proj = activeProject();
  const curFile = proj?.config.logo.file_path || "";
  grid.classList.remove("hidden");
  grid.innerHTML = logos
    .map(
      (lg) => `
      <div class="logo-thumb ${lg.path === curFile ? "active" : ""}" data-path="${lg.path}" title="${lg.name}">
        <img src="${mediaUrl(lg.path)}" alt="${lg.name}" />
      </div>`
    )
    .join("");
  grid.querySelectorAll(".logo-thumb").forEach((el) => {
    el.addEventListener("click", () => {
      const p = activeProject();
      if (!p) return;
      p.config.logo.file_path = el.dataset.path;
      p.config.logo.enabled = true;
      $("logo-file-path").value = el.dataset.path;
      $("logo-enabled").checked = true;
      grid.querySelectorAll(".logo-thumb").forEach((x) => x.classList.remove("active"));
      el.classList.add("active");
      syncLogoVisual();
      selectLayer("logo");
      scheduleAutoSave();
    });
  });
}

async function flushSave(proj) {
  if (!proj) return;
  await api(`/api/projects/${proj.id}`, {
    method: "PUT",
    body: JSON.stringify({
      srt_path: proj.srt_path,
      translated_srt_path: proj.translated_srt_path,
      preset_id: proj.preset_id,
      config: proj.config,
    }),
  });
  state.lastSaveErr = null; // next distinct error will be shown again
}

function scheduleAutoSave() {
  const proj = activeProject();
  if (!proj) return;
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(
    () =>
      flushSave(proj).catch((e) => {
        if (state.lastSaveErr !== e.message) showToast(`⚠ Không lưu được cấu hình:\n${e.message}`, "err");
        state.lastSaveErr = e.message;
      }),
    250
  );
}

// ---------------- ACTION TRIGGERS & POLLING ----------------

async function triggerRender(previewSeconds) {
  const proj = activeProject();
  if (!proj) return alert("Hãy thêm và chọn một video trước!");
  await flushSave(proj);
  $("render-status-text").textContent = previewSeconds
    ? `Đang xuất thử ${previewSeconds} giây...`
    : "Đang bắt đầu xuất video đầy đủ...";
  $("render-pct").textContent = "1%";

  await api("/api/render", {
    method: "POST",
    body: JSON.stringify({ project_id: proj.id, preview_seconds: previewSeconds }),
  });
  startStatusPolling(proj.id);
}

function startStatusPolling(projectId) {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const st = await api(`/api/projects/${projectId}/status`);
      const p = state.projects.find((x) => x.id === projectId);
      if (p) {
        p.status = st.status;
        p.progress = st.progress;
        p.status_message = st.status_message;
        p.output_path = st.output_path;
        p.srt_path = st.srt_path || p.srt_path;
        p.translated_srt_path = st.translated_srt_path || p.translated_srt_path;
        p.style_guide_path = st.style_guide_path || p.style_guide_path;
        p.dub_audio_path = st.dub_audio_path || p.dub_audio_path;
        p.timing = st.timing || p.timing;
      }
      renderVideoList();
      if (state.activeProjectId === projectId) {
        $("render-status-text").textContent = st.status_message || st.status;
        $("render-pct").textContent = `${Math.round(st.progress)}%`;
      }
      if (["ready", "done", "error"].includes(st.status)) {
        clearInterval(state.pollTimer);
        if (state.activeProjectId === projectId) {
          await Promise.all([loadProjectSubtitles(projectId), loadStyleGuideInfo(projectId)]);
          if (p) $("proj-srt-path").value = p.translated_srt_path || p.srt_path || "";
          syncSubtitleVisual();
          afterJobRefresh(projectId, st);
        }
      }
    } catch {
      clearInterval(state.pollTimer);
    }
  }, 600);
}

function startBatchPolling() {
  clearInterval(state.batchPollTimer);
  const banner = $("batch-banner");
  banner.classList.remove("hidden");
  state.batchRunning = true;
  updateStopButton();
  state.batchPollTimer = setInterval(async () => {
    try {
      const st = await api("/api/pipeline/status");
      banner.textContent = st.message;
      state.batchRunning = !!st.running;
      await loadProjects();
      const cur = activeProject();
      if (cur) {
        $("render-status-text").textContent = cur.status_message || st.message;
        $("render-pct").textContent = `${Math.round(cur.progress)}%`;
      }
      if (!st.running) {
        clearInterval(state.batchPollTimer);
        updateStopButton();
        if (cur) afterJobRefresh(cur.id, cur);
      }
    } catch {
      clearInterval(state.batchPollTimer);
    }
  }, 1000);
}

// ---------------- EVENT WIRING ----------------

function bindStaticEvents() {
  initOverlayDragResize();

  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.dataset.tab;
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.add("hidden"));
      $(`tab-${target}`).classList.remove("hidden");
    });
  });

  // Layer chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => selectLayer(chip.dataset.layer));
  });

  // Hardware info modal
  $("hw-badge").addEventListener("click", async () => {
    await loadHardware();
    if (state.hardware) {
      alert(
        `Cấu hình phần cứng phát hiện:\n\n` +
          `• CPU: ${state.hardware.cpu_cores} nhân / ${state.hardware.cpu_threads} luồng\n` +
          `• RAM: ${state.hardware.ram_gb} GB\n` +
          `• GPUs: ${(state.hardware.gpus || []).join(", ")}\n` +
          `• Encoder khả dụng: QSV=${state.hardware.encoders.h264_qsv}, NVENC=${state.hardware.encoders.h264_nvenc}\n` +
          `• Đề xuất: ${state.hardware.recommended_encoder}\n\n` +
          `${state.hardware.summary}`
      );
    }
  });

  // Add videos from disk
  const addVideosHandler = async () => {
    const res = await api("/api/system/pick", {
      method: "POST",
      body: JSON.stringify({ mode: "files", kind: "video" }),
    });
    const presetId = $("preset-select").value || null;
    for (const vp of res.paths || []) {
      const added = await api("/api/projects", {
        method: "POST",
        body: JSON.stringify({ video_path: vp, preset_id: presetId }),
      });
      state.projects.push(added.project);
    }
    renderVideoList();
    if (res.paths && res.paths.length) {
      selectProject(state.projects[state.projects.length - 1].id);
    }
  };
  $("btn-add-videos").addEventListener("click", addVideosHandler);
  $("btn-add-videos-hero").addEventListener("click", addVideosHandler);

  // Download Modal (YouTube / BiliBili). The download runs on the server: closing the dialog does not stop it,
  // reopening re-attaches to it (GET /api/download/active), "Dừng tải" cancels it.
  const DL_START_LABEL = "⬇ Tải Video + Sub gốc về danh sách";
  const setDlRunning = (running) => {
    $("btn-start-dl").disabled = running;
    $("btn-start-dl").textContent = running ? "⏳ Đang tải..." : DL_START_LABEL;
    $("btn-stop-dl").classList.toggle("hidden", !running);
  };
  const showDlStatus = (st) => {
    const stBox = $("dl-status-box");
    stBox.classList.remove("hidden");
    const took = st.started ? ((st.ended || st.now || Date.now() / 1000) - st.started) : 0;
    stBox.textContent = `[${Math.round(st.progress)}%] ${cleanUiText(st.message)}${st.started ? `  ⏱ ${fmtDur(took)}` : ""}`;
  };
  const attachDlJob = (jobId) => {
    state.dlJobId = jobId;
    setDlRunning(true);
    clearInterval(state.dlPollTimer);
    state.dlPollTimer = setInterval(async () => {
      try {
        const st = await api(`/api/download/${jobId}/status`);
        showDlStatus(st);
        if (st.status === "done") {
          clearInterval(state.dlPollTimer);
          setDlRunning(false);
          state.dlJobId = null;
          if (st.project && !state.projects.some((p) => p.id === st.project.id)) {
            state.projects.push(st.project);
            renderVideoList();
            selectProject(st.project.id);
          }
        } else if (st.status === "error" || st.status === "stopped") {
          clearInterval(state.dlPollTimer);
          setDlRunning(false);
          state.dlJobId = null;
        }
      } catch {
        clearInterval(state.dlPollTimer);
        setDlRunning(false);
      }
    }, 700);
  };
  const openDlModal = async () => {
    $("dl-modal").classList.remove("hidden");
    try {
      const { job } = await api("/api/download/active");
      if (job) {
        showDlStatus(job);
        attachDlJob(job.job_id);
        return;
      }
    } catch {}
    if (!state.dlJobId) {
      setDlRunning(false);
      $("dl-status-box").classList.add("hidden");
    }
  };
  const closeDlModal = () => $("dl-modal").classList.add("hidden");
  $("btn-open-dl-modal").addEventListener("click", openDlModal);
  $("btn-open-dl-hero").addEventListener("click", openDlModal);
  $("btn-close-dl-modal").addEventListener("click", closeDlModal);
  $("btn-cancel-dl").addEventListener("click", closeDlModal);
  $("dl-clean-ai").addEventListener("change", (e) => {
    $("dl-ai-opts").classList.toggle("hidden", !e.target.checked);
    $("dl-ai-batch").classList.toggle("hidden", !e.target.checked);
  });
  $("btn-stop-dl").addEventListener("click", async () => {
    if (!state.dlJobId) return;
    $("btn-stop-dl").disabled = true;
    try {
      await api(`/api/download/${state.dlJobId}/cancel`, { method: "POST" });
    } finally {
      $("btn-stop-dl").disabled = false;
    }
  });

  $("btn-start-dl").addEventListener("click", async () => {
    const url = $("dl-url").value.trim();
    if (!url) return alert("Vui lòng dán link YouTube hoặc BiliBili!");
    const stBox = $("dl-status-box");
    stBox.classList.remove("hidden");
    stBox.textContent = "Đang phân tích link video...";
    setDlRunning(true);
    try {
      const job = await api("/api/download", {
        method: "POST",
        body: JSON.stringify({
          url,
          custom_name: $("dl-name").value.trim(),
          sub_lang: $("dl-sub-lang").value,
          threads: Number($("dl-threads").value) || 16,
          browser_cookies: $("dl-cookies").value,
          clean_srt_rules: $("dl-clean-rules").checked,
          clean_srt_ai: $("dl-clean-ai").checked,
          clean_model: $("dl-clean-model").value,
          clean_thinking_level: $("dl-clean-thinking").value,
          clean_batch_lines: Number($("dl-clean-batch").value) || 0,
          api_key: $("tr-api-key").value || "",
          preset_id: $("preset-select").value || null,
        }),
      });
      attachDlJob(job.job_id);
    } catch (e) {
      stBox.textContent = `Lỗi: ${cleanUiText(e.message)}`;
      setDlRunning(false);
    }
  });

  // Batch A-Z all videos
  $("btn-run-batch-az").addEventListener("click", async () => {
    if (!state.projects.length) return alert("Chưa có video nào trong danh sách!");
    const proj = activeProject();
    if (proj) await flushSave(proj);
    const prof = proj?.config.performance_profile || "auto";
    await api("/api/pipeline/run-batch", {
      method: "POST",
      body: JSON.stringify({
        project_ids: state.projects.map((p) => p.id),
        do_translate: true,
        do_voice: true,
        do_render: true,
        profile: prof,
      }),
    });
    startBatchPolling();
  });

  // Stop running job (single video, or the whole A–Z batch if one is running)
  $("btn-stop-job").addEventListener("click", async () => {
    const proj = activeProject();
    const btn = $("btn-stop-job");
    let stopAll = false;
    if (state.batchRunning) {
      const st = await api("/api/pipeline/status").catch(() => null);
      const running = (st && st.running_projects) || [];
      if (running.length > 1) {
        stopAll = confirm(`Đang chạy hàng loạt ${running.length} video.\n\nOK = Dừng TẤT CẢ\nHủy = Chỉ dừng video đang chọn`);
      }
    }
    btn.disabled = true;
    btn.textContent = "⏳ Đang dừng...";
    try {
      if (stopAll || !proj) {
        await api("/api/pipeline/cancel", { method: "POST" });
      } else {
        await api(`/api/projects/${proj.id}/cancel`, { method: "POST" });
      }
      await loadProjects();
      const cur = activeProject();
      if (cur) {
        $("render-status-text").textContent = cur.status_message || "Đã dừng.";
        $("render-pct").textContent = "";
      }
    } catch (e) {
      alert(`Không dừng được: ${e.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "⏹ Dừng Tiến Trình";
      updateStopButton();
    }
  });

  // Single video A-Z
  $("btn-run-one-az").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return alert("Hãy chọn một video trước!");
    await flushSave(proj);
    await api("/api/pipeline/run-batch", {
      method: "POST",
      body: JSON.stringify({
        project_ids: [proj.id],
        do_translate: true,
        do_voice: true,
        do_render: true,
        profile: proj.config.performance_profile || "auto",
      }),
    });
    startBatchPolling();
    startStatusPolling(proj.id);
  });

  // Open output folder
  $("btn-open-output").addEventListener("click", () => {
    const proj = activeProject();
    api("/api/system/open-folder", {
      method: "POST",
      body: JSON.stringify({ path: proj?.output_path || "" }),
    });
  });

  // Presets
  $("btn-save-preset").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return alert("Hãy chọn một video để lưu cấu hình thành Preset!");
    const name = prompt("Nhập tên Preset kênh:", "Preset Kênh Mới");
    if (!name) return;
    const res = await api("/api/presets", {
      method: "POST",
      body: JSON.stringify({ name, config: proj.config }),
    });
    await loadPresets();
    $("preset-select").value = res.preset.id;
  });

  $("btn-apply-preset").addEventListener("click", () => {
    const proj = activeProject();
    const pid = $("preset-select").value;
    const preset = state.presets.find((p) => p.id === pid);
    if (!proj || !preset) return;
    proj.config = JSON.parse(JSON.stringify(preset.config));
    proj.preset_id = preset.id;
    populateInspectorFromProject(proj);
    syncAllOverlays();
    scheduleAutoSave();
  });

  // Video transport
  const vid = $("preview-video");
  const gsVid = $("gs-source-video");
  $("btn-play").addEventListener("click", () => {
    if (!vid.src) return;
    if (vid.paused) {
      vid.play();
      $("btn-play").textContent = "⏸ Dừng";
      const proj = activeProject();
      if (proj?.config.green_screen.enabled && gsVid.src) gsVid.play().catch(() => {});
    } else {
      vid.pause();
      gsVid.pause();
      $("btn-play").textContent = "▶ Phát";
    }
  });

  vid.addEventListener("timeupdate", () => {
    if (vid.duration) {
      $("seek-bar").max = String(vid.duration);
      $("seek-bar").value = String(vid.currentTime);
      $("timecode").textContent = `${fmtTime(vid.currentTime)} / ${fmtTime(vid.duration)}`;
    }
    syncSubtitleVisual();
    if (!vid.paused) drawGreenScreenFrame();
  });

  vid.addEventListener("loadedmetadata", () => {
    $("seek-bar").max = String(vid.duration || 100);
    $("timecode").textContent = `00:00.0 / ${fmtTime(vid.duration)}`;
    fitStageCanvas();
    syncAllOverlays();
  });

  $("seek-bar").addEventListener("input", (e) => {
    vid.currentTime = parseFloat(e.target.value) || 0;
    syncSubtitleVisual();
  });

  // File pickers
  $("btn-pick-srt").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return;
    const res = await api("/api/system/pick", {
      method: "POST",
      body: JSON.stringify({ mode: "file", kind: "srt" }),
    });
    if (res.paths?.[0]) {
      proj.srt_path = res.paths[0];
      $("proj-srt-path").value = proj.srt_path;
      await flushSave(proj);
      await loadProjectSubtitles(proj.id);
      syncSubtitleVisual();
      renderVideoList();
    }
  });

  $("btn-pick-font").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return;
    const res = await api("/api/system/pick", {
      method: "POST",
      body: JSON.stringify({ mode: "file", kind: "font" }),
    });
    if (res.paths?.[0]) {
      const fp = res.paths[0];
      proj.config.subtitle.font_path = fp;
      let opt = Array.from($("sub-font-select").options).find(
        (o) => o.value.toLowerCase() === fp.toLowerCase()
      );
      if (!opt) {
        opt = document.createElement("option");
        opt.value = fp;
        opt.textContent = fp.split(/[\\/]/).pop();
        $("sub-font-select").appendChild(opt);
      }
      $("sub-font-select").value = opt.value;
      syncSubtitleVisual();
      scheduleAutoSave();
    }
  });

  // BGM file & folder pickers
  $("btn-pick-bgm-files").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return alert("Hãy chọn một video trước!");
    const res = await api("/api/system/pick", {
      method: "POST",
      body: JSON.stringify({ mode: "files", kind: "audio" }),
    });
    if (res.paths?.length) {
      proj.config.bgm.file_paths.push(...res.paths);
      proj.config.bgm.enabled = true;
      $("bgm-enabled").checked = true;
      renderBgmList();
      renderVideoList();
      scheduleAutoSave();
    }
  });

  $("btn-pick-bgm-folder").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return alert("Hãy chọn một video trước!");
    const res = await api("/api/system/pick", {
      method: "POST",
      body: JSON.stringify({ mode: "folder", kind: "any" }),
    });
    if (res.paths?.[0]) {
      const list = await api(`/api/system/audio-folder?folder=${encodeURIComponent(res.paths[0])}`);
      const paths = (list.tracks || []).map((t) => t.path);
      if (paths.length) {
        proj.config.bgm.file_paths = paths;
        proj.config.bgm.enabled = true;
        $("bgm-enabled").checked = true;
        renderBgmList();
        renderVideoList();
        scheduleAutoSave();
      } else {
        alert("Không tìm thấy file nhạc (.mp3/.wav/.m4a...) trong thư mục này.");
      }
    }
  });

  $("btn-clear-bgm").addEventListener("click", () => {
    const proj = activeProject();
    if (!proj) return;
    proj.config.bgm.file_paths = [];
    renderBgmList();
    renderVideoList();
    scheduleAutoSave();
  });

  $("btn-pick-logo").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return;
    const res = await api("/api/system/pick", {
      method: "POST",
      body: JSON.stringify({ mode: "file", kind: "image" }),
    });
    if (res.paths?.[0]) {
      proj.config.logo.file_path = res.paths[0];
      proj.config.logo.enabled = true;
      $("logo-file-path").value = res.paths[0];
      $("logo-enabled").checked = true;
      syncLogoVisual();
      selectLayer("logo");
      scheduleAutoSave();
    }
  });

  $("btn-pick-logo-folder").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return;
    const res = await api("/api/system/pick", {
      method: "POST",
      body: JSON.stringify({ mode: "folder", kind: "any" }),
    });
    if (res.paths?.[0]) {
      proj.config.logo.folder_path = res.paths[0];
      $("logo-folder-path").value = res.paths[0];
      await loadLogoFolderGrid(res.paths[0]);
      scheduleAutoSave();
    }
  });

  $("btn-pick-gs").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return;
    const res = await api("/api/system/pick", {
      method: "POST",
      body: JSON.stringify({ mode: "file", kind: "video" }),
    });
    if (res.paths?.[0]) {
      proj.config.green_screen.file_path = res.paths[0];
      proj.config.green_screen.enabled = true;
      $("gs-file-path").value = res.paths[0];
      $("gs-enabled").checked = true;
      syncGreenScreenVisual();
      selectLayer("greenscreen");
      scheduleAutoSave();
    }
  });

  // Phase 2 buttons + SRT Garbage Cleaner
  $("btn-analyze-style").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return alert("Hãy chọn một video có file SRT trước!");
    await flushSave(proj);
    await api(`/api/projects/${proj.id}/analyze-style`, { method: "POST" });
    startStatusPolling(proj.id);
  });

  $("btn-translate-srt").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return alert("Hãy chọn một video có file SRT trước!");
    await flushSave(proj);
    await api(`/api/projects/${proj.id}/translate`, {
      method: "POST",
      body: JSON.stringify({ reset_checkpoint: false }),
    });
    startStatusPolling(proj.id);
  });

  $("btn-translate-reset").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return;
    if (!confirm("Xóa tiến trình dịch cũ và dịch lại từ dòng 1?")) return;
    await flushSave(proj);
    await api(`/api/projects/${proj.id}/translate`, {
      method: "POST",
      body: JSON.stringify({ reset_checkpoint: true }),
    });
    startStatusPolling(proj.id);
  });

  $("btn-translate-clear").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return;
    if (!confirm(`Xóa tiến trình dịch của "${proj.name}"?\nLần dịch sau sẽ bắt đầu lại từ dòng 1. (Không tự chạy dịch.)`)) return;
    const sg = confirm("Xóa luôn Style Guide (nhân vật, xưng hô, văn phong)?\n\nOK = xóa (lần sau phân tích lại)\nCancel = giữ Style Guide");
    try {
      const r = await api(`/api/projects/${proj.id}/translate-checkpoint?style_guide=${sg}`, { method: "DELETE" });
      showToast(`Đã xóa tiến trình dịch (${r.removed_lines} dòng)` + (r.style_guide_removed ? " và Style Guide." : ", giữ Style Guide."), "info", 5000);
      await loadProjects();
      await loadStyleGuideInfo(proj.id);
    } catch (e) {
      showToast(e.message, "err", 8000);
    }
  });

  $("btn-clean-srt-rules").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return alert("Hãy chọn video có file SRT gốc!");
    await api(`/api/projects/${proj.id}/clean-srt`, {
      method: "POST",
      body: JSON.stringify({ use_ai: false }),
    });
    startStatusPolling(proj.id);
  });

  $("btn-clean-srt-ai").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return alert("Hãy chọn video có file SRT gốc!");
    await flushSave(proj);
    await api(`/api/projects/${proj.id}/clean-srt`, {
      method: "POST",
      body: JSON.stringify({
        use_ai: true,
        model: $("clean-srt-model").value,
        thinking_level: $("clean-srt-thinking").value,
        batch_lines: Number($("clean-srt-batch").value) || 0,
        api_key: $("tr-api-key").value || "",
      }),
    });
    startStatusPolling(proj.id);
  });

  $("btn-save-style-guide").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return;
    await api(`/api/projects/${proj.id}/style-guide`, {
      method: "PUT",
      body: JSON.stringify({ content: $("style-guide-editor").value }),
    });
    $("render-status-text").textContent = "Đã lưu chỉnh sửa Style Guide!";
  });

  // Phase 3 Voice buttons & quick switchers
  $("btn-switch-edge").addEventListener("click", async () => {
    $("voice-provider").value = "edge_tts";
    const p = activeProject();
    if (p) {
      p.config.voice.provider = "edge_tts";
      scheduleAutoSave();
    }
    await loadVoices("edge_tts", $("voice-lang").value, $("voice-include-multi").checked);
  });

  $("btn-switch-gtts").addEventListener("click", async () => {
    $("voice-provider").value = "gtts";
    const p = activeProject();
    if (p) {
      p.config.voice.provider = "gtts";
      scheduleAutoSave();
    }
    await loadVoices("gtts", $("voice-lang").value, false);
  });

  $("btn-preview-voice").addEventListener("click", async () => {
    const proj = activeProject();
    const cfg = proj
      ? proj.config.voice
      : {
          provider: $("voice-provider").value,
          language: $("voice-lang").value || "vi",
          voice_id: $("voice-select").value,
          speed: Number($("voice-speed").value) || 1.0,
          concurrency: Number($("voice-concurrency").value) || 12,
          tiktok_session_id: $("voice-tiktok-sid").value || "",
          fallback_edge_tts: $("voice-fallback-edge").checked,
        };
    const higgs = cfg.provider === "higgs3";
    if (higgs && !cfg.higgs_ref_audio) return showToast("Higgs TTS 3: hãy chọn file giọng mẫu trước.", "warn");
    if (!higgs && !cfg.voice_id) return alert("Hãy chọn một giọng đọc trong danh sách trước!");
    const btn = $("btn-preview-voice");
    btn.textContent = higgs ? "⏳ Higgs: chuẩn bị giọng mẫu + bật máy chủ (lần đầu ~1–2 phút)..." : "⏳ Đang tạo mẫu...";
    btn.disabled = true;
    try {
      const sampleByLang = {
        vi: "Xin chào các bạn! Hôm nay chúng ta sẽ cùng review một bộ phim cực kỳ hấp dẫn.",
        fil: "Magandang araw sa inyong lahat! Ngayon ay ire-review natin ang isang napakagandang pelikula.",
        tl: "Magandang araw sa inyong lahat! Ngayon ay ire-review natin ang isang napakagandang pelikula.",
        en: "Welcome back! Today we are recapping an incredible action movie.",
        zh: "大家好，今天我们一起来看一部非常精彩的电影解说。",
        ja: "こんにちは、今日はとても面白い映画を紹介します。",
        ko: "안녕하세요! 오늘은 정말 흥미진진한 영화를 리뷰해 보겠습니다.",
      };
      const text = sampleByLang[cfg.language] || sampleByLang.en;
      const res = await fetch("/api/voices/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: cfg, text }),
      });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const player = $("voice-audition-player");
      player.src = URL.createObjectURL(blob);
      player.classList.remove("hidden");
      player.play();
      const used = res.headers.get("X-TTS-Provider-Used") || "";
      $("render-status-text").textContent = `Đang phát thử giọng (${used})`;
      if (higgs) refreshHiggsStatus();
    } catch (e) {
      alert(`Lỗi nghe thử giọng: ${e.message}`);
    } finally {
      btn.textContent = "🔊 Nghe thử giọng";
      btn.disabled = false;
    }
  });

  $("btn-generate-voices").addEventListener("click", async () => {
    const proj = activeProject();
    if (!proj) return alert("Hãy chọn một video có file SRT trước!");
    await flushSave(proj);
    await api(`/api/projects/${proj.id}/voices`, { method: "POST" });
    startStatusPolling(proj.id);
  });

  // Live control bindings
  const bind = (id, ev, fn) => {
    $(id).addEventListener(ev, (e) => {
      const proj = activeProject();
      if (!proj) return;
      const el = e.target;
      // number inputs do not stop typing above max (e.g. 500 in a max=120 field): clamp, so the saved config
      // is always valid (an out-of-range value made EVERY save fail with HTTP 422 and blocked all buttons).
      if (el.type === "number" && el.value !== "") {
        const v = Number(el.value);
        const lo = el.min !== "" ? Number(el.min) : -Infinity;
        const hi = el.max !== "" ? Number(el.max) : Infinity;
        if (!Number.isNaN(v) && (v < lo || v > hi)) {
          const c = Math.min(hi, Math.max(lo, v));
          if (ev === "change" || v > hi) {
            el.value = String(c);
            showToast(`Giá trị ${v} vượt giới hạn (${el.min}–${el.max}), đã chỉnh thành ${c}.`, "warn", 5000);
          }
        }
      }
      fn(proj, el);
      scheduleAutoSave();
    });
  };

  bind("sub-enabled", "change", (p, el) => { p.config.subtitle.enabled = el.checked; syncSubtitleVisual(); });
  bind("sub-font-select", "change", (p, el) => { p.config.subtitle.font_path = el.value; syncSubtitleVisual(); });
  bind("sub-effect", "change", (p, el) => { p.config.subtitle.effect = el.value; syncSubtitleVisual(); });
  bind("sub-effect-dur", "input", (p, el) => {
    p.config.subtitle.effect_duration = Number(el.value);
    $("val-sub-fx-dur").textContent = `${Number(el.value).toFixed(2)}s`;
  });

  // text-logo / watermark regions
  bind("wm-enabled", "change", (p, el) => {
    p.config.wm_removal.enabled = el.checked;
    if (el.checked) selectLayer("wm", 0);
    syncWmVisual();
  });
  bind("wm-mode", "change", (p, el) => {
    p.config.wm_removal.mode = el.value;
    $("wm-blur-opts").classList.toggle("hidden", el.value !== "blur");
    syncWmVisual();
  });
  bind("wm-blur", "input", (p, el) => { p.config.wm_removal.blur_strength = Number(el.value); $("val-wm-blur").textContent = el.value; });
  $("btn-wm-add").addEventListener("click", () => {
    const p = activeProject();
    if (!p) return;
    const boxes = (p.config.wm_removal.boxes = p.config.wm_removal.boxes || []);
    // new box: top-right first, then top-left, then stacked below
    const presets = [
      { x: 0.7, y: 0.02, w: 0.28, h: 0.07 },
      { x: 0.02, y: 0.02, w: 0.28, h: 0.07 },
    ];
    boxes.push({ ...(presets[boxes.length] || { x: 0.36, y: 0.02 + 0.08 * (boxes.length - 2), w: 0.28, h: 0.07 }) });
    p.config.wm_removal.enabled = true;
    $("wm-enabled").checked = true;
    state.activeWmIdx = boxes.length - 1;
    syncWmVisual();
    selectLayer("wm", boxes.length - 1);
    scheduleAutoSave();
  });
  $("wm-list").addEventListener("click", (e) => {
    const p = activeProject();
    if (!p) return;
    const del = e.target.closest(".btn-wm-del");
    if (del) {
      p.config.wm_removal.boxes.splice(Number(del.dataset.idx), 1);
      state.activeWmIdx = 0;
      syncWmVisual();
      renderWmList();
      scheduleAutoSave();
      return;
    }
    const row = e.target.closest(".wm-row");
    if (row) selectLayer("wm", Number(row.dataset.idx));
  });

  // channel name
  const chBind = (id, ev, fn) => bind(id, ev, (p, el) => { fn(p.config.channel_text, el); syncChannelVisual(); });
  chBind("ch-enabled", "change", (c, el) => { c.enabled = el.checked; if (el.checked) selectLayer("channel"); });
  chBind("ch-text", "input", (c, el) => { c.text = el.value; });
  chBind("ch-font", "change", (c, el) => { c.font_path = el.value; });
  chBind("ch-size", "input", (c, el) => { c.font_size = Number(el.value); $("val-ch-size").textContent = `${el.value}px`; });
  chBind("ch-outline", "input", (c, el) => { c.outline_width = Number(el.value); $("val-ch-outline").textContent = `${el.value}px`; });
  chBind("ch-color", "input", (c, el) => { c.text_color = el.value; });
  chBind("ch-outline-color", "input", (c, el) => { c.outline_color = el.value; });
  chBind("ch-bg-enabled", "change", (c, el) => { c.bg_enabled = el.checked; });
  chBind("ch-bold", "change", (c, el) => { c.bold = el.checked; });
  chBind("ch-bg-color", "input", (c, el) => { c.bg_color = el.value; });
  chBind("ch-bg-opacity", "input", (c, el) => { c.bg_opacity = Number(el.value); $("val-ch-bg-op").textContent = `${Math.round(el.value * 100)}%`; });
  chBind("ch-radius", "input", (c, el) => { c.border_radius = Number(el.value); $("val-ch-radius").textContent = `${el.value}px`; });
  chBind("ch-effect", "change", (c, el) => {
    c.effect = el.value;
    $("ch-period-field").classList.toggle("hidden", !["shimmer", "slide_in", "typewriter"].includes(el.value));
  });
  chBind("ch-period", "change", (c, el) => { c.effect_period_sec = Math.max(2, Math.min(120, Number(el.value) || 8)); });
  bind("sub-size", "input", (p, el) => { p.config.subtitle.font_size = Number(el.value); $("val-sub-size").textContent = `${el.value}px`; syncSubtitleVisual(); });
  bind("sub-outline", "input", (p, el) => { p.config.subtitle.outline_width = Number(el.value); $("val-sub-outline").textContent = `${el.value}px`; syncSubtitleVisual(); });
  bind("sub-color", "input", (p, el) => { p.config.subtitle.text_color = el.value; syncSubtitleVisual(); });
  bind("sub-outline-color", "input", (p, el) => { p.config.subtitle.outline_color = el.value; syncSubtitleVisual(); });
  bind("sub-bg-enabled", "change", (p, el) => { p.config.subtitle.bg_enabled = el.checked; syncSubtitleVisual(); });
  bind("sub-bold", "change", (p, el) => { p.config.subtitle.bold = el.checked; syncSubtitleVisual(); });
  bind("sub-bg-color", "input", (p, el) => { p.config.subtitle.bg_color = el.value; syncSubtitleVisual(); });
  bind("sub-bg-opacity", "input", (p, el) => { p.config.subtitle.bg_opacity = Number(el.value); $("val-sub-bg-op").textContent = `${Math.round(el.value * 100)}%`; syncSubtitleVisual(); });
  bind("sub-radius", "input", (p, el) => { p.config.subtitle.border_radius = Number(el.value); $("val-sub-radius").textContent = `${el.value}px`; syncSubtitleVisual(); });
  bind("sub-align", "change", (p, el) => { p.config.subtitle.align = el.value; syncSubtitleVisual(); });
  ["x", "y", "w", "h"].forEach((k) => {
    bind(`sub-${k}`, "input", (p, el) => {
      const v = parseFloat(el.value);
      if (!isNaN(v)) {
        p.config.subtitle.box[k] = Math.max(0, Math.min(1, v));
        syncSubtitleVisual();
      }
    });
  });

  bind("bgm-enabled", "change", (p, el) => { p.config.bgm.enabled = el.checked; renderVideoList(); });
  bind("bgm-volume", "input", (p, el) => { p.config.bgm.volume = Number(el.value); $("val-bgm-vol").textContent = `${Math.round(el.value * 100)}%`; });
  bind("bgm-crossfade", "input", (p, el) => { p.config.bgm.crossfade_sec = Number(el.value); $("val-bgm-xfade").textContent = `${Number(el.value).toFixed(1)}s`; });
  bind("bgm-ducking", "change", (p, el) => { p.config.bgm.auto_ducking = el.checked; });

  bind("logo-enabled", "change", (p, el) => { p.config.logo.enabled = el.checked; syncLogoVisual(); });
  bind("logo-opacity", "input", (p, el) => { p.config.logo.opacity = Number(el.value); $("val-logo-op").textContent = `${Math.round(el.value * 100)}%`; syncLogoVisual(); });

  bind("gs-enabled", "change", (p, el) => { p.config.green_screen.enabled = el.checked; syncGreenScreenVisual(); });
  bind("gs-color", "input", (p, el) => { p.config.green_screen.chroma_color = el.value; drawGreenScreenFrame(); });
  bind("gs-start", "input", (p, el) => { p.config.green_screen.start_sec = Math.max(0, parseFloat(el.value) || 0); });
  bind("gs-sim", "input", (p, el) => { p.config.green_screen.similarity = Number(el.value); $("val-gs-sim").textContent = Number(el.value).toFixed(2); drawGreenScreenFrame(); });
  bind("gs-blend", "input", (p, el) => { p.config.green_screen.blend = Number(el.value); $("val-gs-blend").textContent = Number(el.value).toFixed(2); drawGreenScreenFrame(); });
  bind("gs-loop", "change", (p, el) => { p.config.green_screen.loop_enabled = el.checked; });

  bind("sr-mode", "change", (p, el) => {
    p.config.sub_removal.mode = el.value;
    $("sr-blur-opts").classList.toggle("hidden", el.value !== "blur");
    $("sr-ai-opts").classList.toggle("hidden", el.value !== "ai_clean");
    if (el.value !== "none") selectLayer("subremoval");
    syncSubRemovalVisual();
  });
  bind("sr-blur-strength", "input", (p, el) => { p.config.sub_removal.blur_strength = Number(el.value); $("val-sr-blur").textContent = el.value; syncSubRemovalVisual(); });
  bind("sr-ai-engine", "change", (p, el) => { p.config.sub_removal.ai_engine = el.value; });
  bind("sr-auto-ocr", "change", (p, el) => { p.config.sub_removal.auto_detect_ocr = el.checked; });
  bind("sr-ocr-fps", "change", (p, el) => { p.config.sub_removal.ocr_sample_fps = Number(el.value) || 4; });

  bind("audio-edit-mode", "change", (p, el) => {
    p.config.audio.edit_mode = el.value;
    if (el.value === "dubbing") {
      p.config.audio.mute_original = true;
      $("audio-mute").checked = true;
    } else {
      p.config.audio.mute_original = false;
      $("audio-mute").checked = false;
    }
    syncAudioPreview();
  });
  bind("audio-orig-vol", "input", (p, el) => { p.config.audio.original_volume = Number(el.value); $("val-orig-vol").textContent = `${Math.round(el.value * 100)}%`; syncAudioPreview(); });
  bind("audio-max-atempo", "input", (p, el) => { p.config.audio.max_atempo = Number(el.value); $("val-max-atempo").textContent = `${Number(el.value).toFixed(2)}x`; });
  bind("audio-mute", "change", (p, el) => { p.config.audio.mute_original = el.checked; syncAudioPreview(); });
  bind("audio-ducking", "change", (p, el) => { p.config.audio.auto_ducking = el.checked; });
  bind("cfg-encoder", "change", (p, el) => { p.config.encoder_override = el.value; });
  bind("cfg-perf", "change", (p, el) => { p.config.performance_profile = el.value; });

  bind("tr-src-lang", "change", (p, el) => { p.config.translate.source_lang = el.value; });
  bind("tr-tgt-lang", "change", (p, el) => { p.config.translate.target_lang = el.value; });
  bind("tr-provider", "change", (p, el) => { p.config.translate.provider = el.value; syncProviderUi(); });
  bind("tr-web-mode", "change", (p, el) => { p.config.translate.web_mode = el.value; syncProviderUi(); });
  bind("tr-web-chunk", "input", (p, el) => {
    p.config.translate.web_chunk_lines = Math.max(20, Math.min(3000, Number(el.value) || 250));
    updateWebPlan();
  });
  bind("tr-web-newchat", "change", (p, el) => { p.config.translate.web_new_chat = el.checked; });
  bind("tr-model", "change", (p, el) => { p.config.translate.model = el.value; });
  bind("tr-thinking", "change", (p, el) => { p.config.translate.thinking_level = el.value; });
  bind("tr-whole-file", "change", (p, el) => {
    p.config.translate.whole_file = el.checked;
    syncProviderUi();
  });
  bind("tr-batch", "input", (p, el) => { p.config.translate.batch_size = Math.max(10, Math.min(2000, Number(el.value) || 45)); });
  bind("tr-tol", "input", (p, el) => { p.config.translate.length_tolerance_pct = Number(el.value) || 15; });
  bind("tr-api-key", "input", (p, el) => { p.config.translate.api_key = el.value; });
  bind("tr-concurrency", "input", (p, el) => {
    p.config.translate.concurrency = Math.max(1, Math.min(16, Number(el.value) || 4));
  });

  const refreshVoiceList = async () => {
    const prov = $("voice-provider").value;
    const lang = $("voice-lang").value;
    const multi = $("voice-include-multi").checked;
    const p = activeProject();
    if (p) {
      p.config.voice.provider = prov;
      p.config.voice.language = lang;
      p.config.voice.include_multilingual = multi;
      scheduleAutoSave();
    }
    syncVoiceProviderUi(prov);
    if (prov === "higgs3") {
      if (p) refreshReviewButton(p.id);
      return;
    }
    await loadVoices(prov, lang, multi);
  };
  $("voice-provider").addEventListener("change", refreshVoiceList);
  $("voice-lang").addEventListener("change", refreshVoiceList);
  $("voice-include-multi").addEventListener("change", refreshVoiceList);

  bind("voice-select", "change", (p, el) => {
    p.config.voice.voice_id = el.value;
    const found = state.voices.find((v) => v.id === el.value);
    if (found) p.config.voice.voice_name = found.name;
  });
  bind("voice-speed", "input", (p, el) => { p.config.voice.speed = Number(el.value); $("val-voice-speed").textContent = `${Number(el.value).toFixed(2)}x`; });
  bind("voice-concurrency", "input", (p, el) => {
    const n = Math.max(1, Math.min(500, Number(el.value) || 1));
    p.config.voice.concurrency = n;
    const cap = { capcut_tiktok: 32, gtts: 16 }[p.config.voice.provider];
    $("val-voice-conc").textContent = cap && n > cap ? `${n} → dùng ${cap} luồng (giới hạn nguồn này)` : `${n} luồng`;
  });
  bind("voice-tiktok-sid", "input", (p, el) => { p.config.voice.tiktok_session_id = el.value; });
  bind("voice-fallback-edge", "change", (p, el) => { p.config.voice.fallback_edge_tts = el.checked; });

  bindHiggsEvents(bind);

  // Render buttons
  $("btn-render-sample").addEventListener("click", () => triggerRender(8.0));
  $("btn-render-full").addEventListener("click", () => triggerRender(null));
}


// ---------------- LENGTH FIT + HIGGS TTS 3 + VOICE REVIEW TABLE ----------------

const HIGGS_FOLDER_KEY = "higgsRefFolder";
const VR = { rows: [], page: 0, per: 100, edits: {}, pollTimer: null, projectId: null, counts: null };

function escHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtTime(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = (s % 60).toFixed(1);
  return `${h ? h + ":" : ""}${String(m).padStart(h ? 2 : 1, "0")}:${r.padStart(4, "0")}`;
}

function mediaUrl(path) {
  return `/api/media/stream?path=${encodeURIComponent(path)}&t=${Date.now()}`;
}

function syncVoiceProviderUi(prov) {
  const higgs = prov === "higgs3";
  $("voice-std-box").classList.toggle("hidden", higgs);
  $("higgs-box").classList.toggle("hidden", !higgs);
  if (higgs) refreshHiggsStatus();
}

function applyHiggsConfig(c) {
  const v = c.voice;
  $("tr-fit-timing").checked = c.translate.fit_timing ?? true;
  $("voice-auto-fit").checked = v.auto_fit ?? true;
  $("higgs-ref").value = v.higgs_ref_audio || "";
  $("higgs-ref-text").value = v.higgs_ref_text || "";
  $("higgs-clean").checked = v.higgs_clean_ref ?? true;
  $("higgs-clean-strong").checked = !!v.higgs_clean_strong;
  $("higgs-review").checked = v.higgs_review ?? true;
  $("higgs-retry").value = String(v.higgs_auto_retry ?? 1);
  $("higgs-pause").checked = v.higgs_pause_review ?? true;
  $("higgs-ref-result").classList.add("hidden");
  syncVoiceProviderUi(v.provider);
}

async function refreshLengthReport(projectId) {
  const el = $("tr-length-report");
  try {
    const r = await api(`/api/projects/${projectId}/length-report`);
    if (!r.checked) {
      el.textContent = "Độ dài bản dịch: chưa có bản dịch";
      el.className = "fit-report";
      return;
    }
    const pct = Math.round((100 * r.over) / r.checked);
    el.textContent = r.over
      ? `⚠ ${r.over}/${r.checked} câu (${pct}%) dài quá khung · cả bài cần đọc nhanh ${r.ratio}x`
      : `✓ Cả ${r.checked} câu vừa khung thời gian (${r.ratio}x)`;
    el.className = `fit-report ${r.over ? "bad" : "good"}`;
  } catch {
    el.textContent = "";
  }
}

async function refreshHiggsStatus(refresh = false) {
  const el = $("higgs-status");
  try {
    const s = await api(`/api/higgs/status${refresh ? "?refresh=true" : ""}`);
    if (!s.ok) {
      el.textContent = !s.scripts
        ? "⚠ Không thấy script Higgs ở D:\\Download\\Model"
        : "⚠ Chưa cài Higgs TTS 3 trong WSL (Ubuntu: /root/higgs3)";
      el.className = "fit-report bad";
      return;
    }
    const env = s.env || {};
    const envTxt = env.python_whisper ? "" : " · ⚠ thiếu môi trường Whisper (envs\\giong_sach)";
    el.textContent = (s.alive ? `● Máy chủ Higgs đang chạy (${s.config || "?"}, giữ ~12GB VRAM)` : "○ Máy chủ Higgs đang tắt (tự bật khi tạo voice, ~1 phút)")
      + (s.gpu ? ` · ${s.gpu}` : "") + envTxt;
    el.className = `fit-report ${s.alive ? "good" : ""}`;
  } catch (e) {
    el.textContent = `Không đọc được trạng thái Higgs: ${e.message}`;
  }
}

async function loadHiggsRefFolder(folder) {
  const list = $("higgs-ref-list");
  if (!folder) return;
  localStorage.setItem(HIGGS_FOLDER_KEY, folder);
  const r = await api(`/api/higgs/refs?folder=${encodeURIComponent(folder)}`);
  state.higgsRefs = r.refs || [];
  $("higgs-ref-count").textContent = String(state.higgsRefs.length);
  list.innerHTML = state.higgsRefs
    .map((x) => `<option value="${escHtml(x.path)}">${escHtml(x.name)}${x.text ? " — có lời" : " — chưa có lời (.txt)"}</option>`)
    .join("");
  const cur = $("higgs-ref").value;
  if (cur && state.higgsRefs.some((x) => x.path === cur)) list.value = cur;
}

// lines shortened / edited after the voice was made: their voice still reads the OLD text
async function refreshVoicePending(projectId) {
  const row = $("voice-pending-row");
  try {
    const r = await api(`/api/projects/${projectId}/voice-pending`);
    if (state.activeProjectId !== projectId) return;
    const show = r.has_voice && r.pending > 0;
    row.classList.toggle("hidden", !show);
    if (show) {
      $("voice-pending-text").textContent =
        `⚠ ${r.pending} câu đã đổi chữ (rút gọn / sửa) sau lần tạo voice trước — voice vẫn đọc chữ cũ`;
      $("btn-voice-pending").textContent = `🔄 Tạo voice cho ${r.pending} câu đã đổi chữ`;
    }
  } catch {
    row.classList.add("hidden");
  }
}

async function runPendingVoices(projectId) {
  const p = state.projects.find((x) => x.id === projectId);
  if (!p) return;
  try {
    await flushSave(p);
    if (p.config.voice.provider === "higgs3") {
      // Higgs: only those lines (server stays on for <= 40 lines, Whisper hears the new versions)
      const r = await api(`/api/projects/${projectId}/voice-review/regen`, {
        method: "POST",
        body: JSON.stringify({ indices: [], pending: true }),
      });
      showToast(`Đang tạo voice cho ${r.count} câu đã đổi chữ...`, "info", 5000);
      if (!$("vr-modal").classList.contains("hidden") && VR.projectId === projectId) watchReviewJob();
    } else {
      // other voices: the whole step runs again, unchanged lines come from the cache (no new request)
      await api(`/api/projects/${projectId}/voices`, { method: "POST" });
      showToast("Đang tạo voice cho các câu đã đổi chữ (câu khác lấy lại từ cache)...", "info", 5000);
    }
    startStatusPolling(projectId);
  } catch (e) {
    showToast(`Không tạo được voice: ${e.message}`);
  }
}

async function refreshReviewButton(projectId) {
  refreshVoicePending(projectId);
  const p = state.projects.find((x) => x.id === projectId);
  const btn = $("btn-voice-review");
  if (!p || p.config.voice.provider !== "higgs3") {
    btn.textContent = "📋 Bảng kiểm tra voice";
    btn.classList.remove("has-bad");
    return;
  }
  try {
    const r = await api(`/api/projects/${projectId}/voice-review?show=flagged`);
    const c = r.counts || {};
    btn.textContent = c.total
      ? `📋 Bảng kiểm tra voice — ${c.flagged} câu nghi ngờ / ${c.total} câu`
      : "📋 Bảng kiểm tra voice (chưa tạo voice Higgs)";
    btn.classList.toggle("has-bad", (c.flagged || 0) > 0);
  } catch {
    /* ignore */
  }
}

// called when a job of a project ends (single-job polling or batch polling)
function afterJobRefresh(projectId, st) {
  if (state.activeProjectId !== projectId) return;
  refreshLengthReport(projectId);
  refreshReviewButton(projectId);
  if (!$("vr-modal").classList.contains("hidden") && VR.projectId === projectId) loadReviewRows();
  const msg = (st && st.status_message) || "";
  if (msg.startsWith("⏸")) {
    showToast(msg, "warn", 12000);
    openReviewModal();
  }
}

// ---------- review modal ----------
function reviewFiltered() {
  const q = ($("vr-search").value || "").trim().toLowerCase();
  if (!q) return VR.rows;
  return VR.rows.filter((r) => `${r.text} ${r.heard || ""} ${r.source || ""} ${r.index}`.toLowerCase().includes(q));
}

function markHeard(text, heard) {
  if (!heard) return '<span class="vr-time">(chưa nghe / không nghe ra lời)</span>';
  const norm = (w) => w.toLowerCase().replace(/[^\p{L}\p{N}']+/gu, "");
  const ref = new Set(text.split(/[\s-]+/).map(norm).filter(Boolean));
  return heard
    .split(/(\s+)/)
    .map((w) => (w.trim() && !ref.has(norm(w)) && norm(w) ? `<mark>${escHtml(w)}</mark>` : escHtml(w)))
    .join("");
}

function renderReviewTable() {
  const rows = reviewFiltered();
  const pages = Math.max(1, Math.ceil(rows.length / VR.per));
  VR.page = Math.min(VR.page, pages - 1);
  const slice = rows.slice(VR.page * VR.per, (VR.page + 1) * VR.per);
  $("vr-tbody").innerHTML = slice.length
    ? slice
        .map((r) => {
          const ed = VR.edits[r.index];
          const cls = `${r.flag ? "vr-flag" : ""} ${r.ok && !r.stale ? "vr-ok" : ""} ${r.stale ? "vr-stale" : ""}`;
          const versions = (r.tries || [])
            .map((t) => `<option value="${t.t}" ${t.t === r.sel ? "selected" : ""}>Bản ${t.t + 1}${t.flag ? " ⚠" : " ✓"}</option>`)
            .join("");
          const okTxt = r.ok_by === "ai" ? "✓ AI đánh dấu OK (Whisper nghe nhầm)" : "Đã đánh dấu OK";
          let reasons = r.reasons && r.reasons.length
            ? `<ul class="vr-reasons">${r.reasons.map((x) => `<li>${escHtml(x)}</li>`).join("")}</ul>`
            : (r.ok ? `<span class="vr-time">${okTxt}</span>` : '<span class="vr-time">—</span>');
          if (r.ok && r.reasons && r.reasons.length) reasons += `<div class="vr-time">${okTxt}</div>`;
          if (r.ai && !r.stale) {
            const k = { OK: "ok", SAI: "bad" }[r.ai.verdict] || "unsure";
            const lbl = { OK: "Whisper nghe nhầm", SAI: "Đọc sai thật", NGHI: "Cần nghe" }[r.ai.verdict] || r.ai.verdict;
            reasons += `<div class="vr-ai ${k}">🤖 ${lbl}${r.ai.note ? ` — ${escHtml(r.ai.note)}` : ""}</div>`;
          }
          if (r.stale) {
            reasons = `<div class="vr-stale-note">✏ Chữ đã đổi sau khi tạo voice — voice hiện tại vẫn đọc: «${escHtml(r.voiced_text)}». Bấm 🔄 hoặc ↻ để tạo voice mới.</div>`;
          }
          return `<tr class="${cls}" data-idx="${r.index}">
            <td><input type="checkbox" class="vr-chk" data-idx="${r.index}" /></td>
            <td><b>${r.index}</b><div class="vr-time">${fmtTime(r.start)}</div><div class="vr-time">${(r.dur || 0).toFixed(1)}s / ${(r.allow || 0).toFixed(1)}s</div></td>
            <td><textarea class="vr-text ${ed !== undefined ? "edited" : ""}" data-idx="${r.index}">${escHtml(ed !== undefined ? ed : r.text)}</textarea>
                ${r.source ? `<div class="vr-src">${escHtml(r.source)}</div>` : ""}</td>
            <td class="vr-heard">${markHeard(r.voiced_text || r.text, r.heard)}</td>
            <td>${reasons}</td>
            <td><button class="btn btn-sm vr-play" data-wav="${escHtml(r.wav || "")}" ${r.wav ? "" : "disabled"}>▶</button></td>
            <td>${versions ? `<select class="vr-ver" data-idx="${r.index}">${versions}</select>` : ""}</td>
            <td><button class="btn btn-sm vr-regen" data-idx="${r.index}">↻</button></td>
          </tr>`;
        })
        .join("")
    : `<tr><td colspan="8" style="text-align:center;padding:18px;color:var(--text-2);">${
        VR.rows.length ? "Không có câu khớp ô tìm." : "Không có câu nghi ngờ nào 🎉"
      }</td></tr>`;
  $("vr-page-info").textContent = `${rows.length} câu · trang ${VR.page + 1}/${pages}`;
  const c = VR.counts || {};
  $("vr-counts").textContent = `${c.flagged ?? 0} nghi ngờ · ${c.accepted ?? 0} đã OK · ${c.total ?? 0} câu`
    + (c.unchecked ? ` · ${c.unchecked} chưa nghe` : "")
    + (c.stale ? ` · ${c.stale} câu đổi chữ chưa có voice mới` : "");
  $("vr-check-all").checked = false;
  const pb = $("btn-vr-pending");
  pb.classList.toggle("hidden", !c.stale);
  pb.textContent = `🔄 Tạo voice ${c.stale || 0} câu đã đổi chữ`;
  const nBad = VR.rows.filter((r) => !r.stale && r.flag && r.ai && r.ai.verdict === "SAI").length;
  const bb = $("btn-vr-ai-bad");
  bb.classList.toggle("hidden", !nBad);
  bb.textContent = `↻ Tạo lại ${nBad} câu AI báo sai`;
}

async function loadReviewRows() {
  if (!VR.projectId) return;
  const r = await api(`/api/projects/${VR.projectId}/voice-review?show=${$("vr-show").value}`);
  VR.rows = r.rows || [];
  VR.counts = r.counts || {};
  renderReviewTable();
}

async function openReviewModal() {
  const p = activeProject();
  if (!p) return showToast("Hãy chọn một video trước.");
  VR.projectId = p.id;
  VR.edits = {};
  VR.page = 0;
  $("vr-title").textContent = p.name;
  $("vr-modal").classList.remove("hidden");
  $("vr-tbody").innerHTML = '<tr><td colspan="8" style="padding:18px;">Đang tải...</td></tr>';
  try {
    await loadReviewRows();
  } catch (e) {
    showToast(`Không tải được bảng kiểm tra: ${e.message}`);
  }
  if (["voicing"].includes(p.status)) watchReviewJob();
}

function selectedReviewIdx() {
  return [...document.querySelectorAll(".vr-chk:checked")].map((x) => Number(x.dataset.idx));
}

function watchReviewJob() {
  clearInterval(VR.pollTimer);
  const box = $("vr-progress");
  box.classList.remove("hidden");
  VR.pollTimer = setInterval(async () => {
    try {
      const st = await api(`/api/projects/${VR.projectId}/status`);
      box.textContent = `⏳ ${st.status_message || st.status} (${Math.round(st.progress)}%)`;
      if (["ready", "done", "error"].includes(st.status)) {
        clearInterval(VR.pollTimer);
        box.classList.toggle("hidden", st.status !== "error");
        if (st.status === "error") box.textContent = st.status_message;
        else if (VR.toastOnDone && st.status_message) showToast(st.status_message, "info", 12000);
        VR.toastOnDone = false;
        await loadReviewRows();
        refreshReviewButton(VR.projectId);
        refreshHiggsStatus();
      }
    } catch {
      clearInterval(VR.pollTimer);
    }
  }, 1000);
}

async function regenReview(indices, allFlagged = false, extra = {}) {
  const edits = {};
  for (const [k, v] of Object.entries(VR.edits)) {
    if (allFlagged || indices.includes(Number(k))) edits[k] = v;
  }
  const hasExtra = Object.values(extra).some(Boolean);
  if (!indices.length && !allFlagged && !hasExtra && !Object.keys(edits).length) return showToast("Chưa chọn câu nào.", "warn");
  try {
    const r = await api(`/api/projects/${VR.projectId}/voice-review/regen`, {
      method: "POST",
      body: JSON.stringify({ indices, all_flagged: allFlagged, edits, ...extra }),
    });
    for (const k of Object.keys(edits)) delete VR.edits[k];
    showToast(`Đang tạo lại ${r.count} câu (máy chủ Higgs tắt thì cần ~1 phút để bật)...`, "info", 5000);
    watchReviewJob();
    startStatusPolling(VR.projectId);
  } catch (e) {
    showToast(`Không tạo lại được: ${e.message}`);
  }
}

function bindHiggsEvents(bind) {
  bind("tr-fit-timing", "change", (p, el) => { p.config.translate.fit_timing = el.checked; });
  bind("voice-auto-fit", "change", (p, el) => { p.config.voice.auto_fit = el.checked; });
  bind("higgs-ref", "change", (p, el) => { p.config.voice.higgs_ref_audio = el.value.trim(); });
  bind("higgs-ref-text", "input", (p, el) => { p.config.voice.higgs_ref_text = el.value; });
  bind("higgs-clean", "change", (p, el) => { p.config.voice.higgs_clean_ref = el.checked; });
  bind("higgs-clean-strong", "change", (p, el) => { p.config.voice.higgs_clean_strong = el.checked; });
  bind("higgs-review", "change", (p, el) => { p.config.voice.higgs_review = el.checked; });
  bind("higgs-retry", "change", (p, el) => { p.config.voice.higgs_auto_retry = Number(el.value) || 0; });
  bind("higgs-pause", "change", (p, el) => { p.config.voice.higgs_pause_review = el.checked; });

  const folder = localStorage.getItem(HIGGS_FOLDER_KEY) || "D:\\Download\\Youtube\\Voice\\Mau";
  $("higgs-ref-folder").value = folder;
  loadHiggsRefFolder(folder).catch(() => {});
  $("btn-higgs-folder").addEventListener("click", () => loadHiggsRefFolder($("higgs-ref-folder").value.trim()).catch((e) => showToast(e.message)));
  $("higgs-ref-list").addEventListener("change", () => {
    const ref = (state.higgsRefs || []).find((x) => x.path === $("higgs-ref-list").value);
    const p = activeProject();
    if (!ref || !p) return;
    $("higgs-ref").value = ref.path;
    $("higgs-ref-text").value = ref.text || "";
    p.config.voice.higgs_ref_audio = ref.path;
    p.config.voice.higgs_ref_text = ref.text || "";
    $("higgs-ref-result").classList.add("hidden");
    scheduleAutoSave();
  });
  $("btn-higgs-pick").addEventListener("click", async () => {
    const r = await api("/api/system/pick", { method: "POST", body: JSON.stringify({ mode: "file", kind: "audio" }) });
    const path = (r.paths || [])[0];
    const p = activeProject();
    if (!path || !p) return;
    $("higgs-ref").value = path;
    p.config.voice.higgs_ref_audio = path;
    const ref = (state.higgsRefs || []).find((x) => x.path === path);
    if (ref && ref.text) {
      $("higgs-ref-text").value = ref.text;
      p.config.voice.higgs_ref_text = ref.text;
    }
    scheduleAutoSave();
  });
  $("btn-higgs-stop").addEventListener("click", async () => {
    try {
      await api("/api/higgs/stop", { method: "POST" });
      showToast("Đã tắt máy chủ Higgs (trả VRAM).", "info", 4000);
    } catch (e) {
      showToast(e.message);
    }
    refreshHiggsStatus();
  });
  $("btn-higgs-prepare").addEventListener("click", async () => {
    const p = activeProject();
    if (!p) return showToast("Hãy chọn một video trước.");
    if (!p.config.voice.higgs_ref_audio) return showToast("Chưa chọn file giọng mẫu.", "warn");
    const btn = $("btn-higgs-prepare");
    const box = $("higgs-ref-result");
    btn.disabled = true;
    btn.textContent = "⏳ Đang lọc + cắt giọng mẫu (lần đầu ~1 phút)...";
    try {
      await flushSave(p);
      const r = await api("/api/higgs/prepare-ref", { method: "POST", body: JSON.stringify({ config: p.config.voice }) });
      const d0 = r.dnsmos_before || {}, d1 = r.dnsmos_after || {};
      box.innerHTML = `
        <div><b>Gốc</b> <audio controls src="${mediaUrl(r.source)}" style="width:100%;height:30px;"></audio></div>
        ${r.full_clean ? `<div><b>Đã lọc (cả file)</b> <audio controls src="${mediaUrl(r.full_clean)}" style="width:100%;height:30px;"></audio></div>` : ""}
        <div><b>Đoạn dùng để clone (${r.seconds ?? "?"}s)</b> <audio controls src="${mediaUrl(r.wav)}" style="width:100%;height:30px;"></audio></div>
        <div style="margin-top:4px;">Chất lượng DNSMOS (1–5): tổng ${d0.OVRL ?? "?"} → <b>${d1.OVRL ?? "?"}</b> · giọng ${d0.SIG ?? "?"} → ${d1.SIG ?? "?"} · nền sạch ${d0.BAK ?? "?"} → ${d1.BAK ?? "?"}</div>
        <ul style="margin:4px 0 0;padding-left:16px;">${(r.steps || []).map((x) => `<li>${escHtml(x)}</li>`).join("")}
        ${(r.skipped || []).map((x) => `<li style="color:var(--text-2)">${escHtml(x)}</li>`).join("")}</ul>
        ${(r.warnings || []).length ? `<div style="color:var(--warn);margin-top:4px;">⚠ ${r.warnings.map(escHtml).join("<br/>⚠ ")}</div>` : ""}
        <div class="vr-time" style="margin-top:4px;">Lời đoạn clone: ${escHtml(r.text)}</div>`;
      box.classList.remove("hidden");
    } catch (e) {
      showToast(`Lọc giọng mẫu lỗi: ${e.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "🎧 Lọc & nghe thử giọng mẫu (trước / sau)";
    }
  });

  $("btn-fit-length").addEventListener("click", async () => {
    const p = activeProject();
    if (!p) return showToast("Hãy chọn một video trước.");
    try {
      await flushSave(p);
      await api(`/api/projects/${p.id}/fit-length`, { method: "POST" });
      startStatusPolling(p.id);
      updateStopButton && updateStopButton();
    } catch (e) {
      showToast(e.message);
    }
  });

  // review modal
  $("btn-voice-review").addEventListener("click", openReviewModal);
  $("btn-close-vr").addEventListener("click", () => {
    $("vr-modal").classList.add("hidden");
    $("vr-player").pause();
    clearInterval(VR.pollTimer);
  });
  $("vr-show").addEventListener("change", () => { VR.page = 0; loadReviewRows(); });
  $("vr-search").addEventListener("input", () => { VR.page = 0; renderReviewTable(); });
  $("btn-vr-prev").addEventListener("click", () => { VR.page = Math.max(0, VR.page - 1); renderReviewTable(); });
  $("btn-vr-next").addEventListener("click", () => { VR.page += 1; renderReviewTable(); });
  $("vr-check-all").addEventListener("change", (e) => {
    document.querySelectorAll(".vr-chk").forEach((x) => { x.checked = e.target.checked; });
  });
  $("vr-tbody").addEventListener("click", async (e) => {
    const t = e.target;
    if (t.classList.contains("vr-play")) {
      const pl = $("vr-player");
      document.querySelectorAll(".vr-table tr.vr-current").forEach((r) => r.classList.remove("vr-current"));
      t.closest("tr").classList.add("vr-current");
      pl.src = mediaUrl(t.dataset.wav);
      pl.play().catch(() => {});
    } else if (t.classList.contains("vr-regen")) {
      regenReview([Number(t.dataset.idx)]);
    }
  });
  $("vr-tbody").addEventListener("input", (e) => {
    const t = e.target;
    if (!t.classList.contains("vr-text")) return;
    const row = VR.rows.find((r) => r.index === Number(t.dataset.idx));
    if (row && t.value.trim() !== row.text.trim()) {
      VR.edits[row.index] = t.value;
      t.classList.add("edited");
    } else if (row) {
      delete VR.edits[row.index];
      t.classList.remove("edited");
    }
  });
  $("vr-tbody").addEventListener("change", async (e) => {
    const t = e.target;
    if (!t.classList.contains("vr-ver")) return;
    try {
      await api(`/api/projects/${VR.projectId}/voice-review/select`, {
        method: "POST",
        body: JSON.stringify({ index: Number(t.dataset.idx), t: Number(t.value) }),
      });
      await loadReviewRows();
    } catch (err) {
      showToast(err.message);
    }
  });
  $("btn-vr-regen-sel").addEventListener("click", () => regenReview(selectedReviewIdx()));
  $("btn-vr-regen-all").addEventListener("click", () => {
    const n = (VR.counts || {}).flagged || 0;
    if (!n && !Object.keys(VR.edits).length) return showToast("Không có câu nghi ngờ nào để tạo lại.", "info", 4000);
    regenReview([], true);
  });
  const markOk = async (ok) => {
    const idx = selectedReviewIdx();
    if (!idx.length) return showToast("Chưa chọn câu nào.", "warn");
    await api(`/api/projects/${VR.projectId}/voice-review/ok`, { method: "POST", body: JSON.stringify({ indices: idx, ok }) });
    await loadReviewRows();
    refreshReviewButton(VR.projectId);
  };
  $("btn-vr-ok-sel").addEventListener("click", () => markOk(true));
  $("btn-vr-unok-sel").addEventListener("click", () => markOk(false));
  $("btn-vr-pending").addEventListener("click", () => {
    VR.toastOnDone = true;
    regenReview([], false, { pending: true });
  });
  $("btn-vr-ai-bad").addEventListener("click", () => regenReview([], false, { ai_bad: true }));
  $("btn-vr-ai").addEventListener("click", async () => {
    const sel = selectedReviewIdx();
    try {
      await api(`/api/projects/${VR.projectId}/voice-review/ai-check`, {
        method: "POST",
        body: JSON.stringify({ indices: sel, recheck: sel.length > 0 }),
      });
      showToast(
        sel.length
          ? `AI đang soát ${sel.length} câu đã chọn...`
          : "AI đang soát tất cả câu nghi ngờ chưa soát (Antigravity ~1–4 phút)...",
        "info", 6000,
      );
      VR.toastOnDone = true;
      watchReviewJob();
      startStatusPolling(VR.projectId);
    } catch (e) {
      showToast(`Không gửi được cho AI: ${e.message}`);
    }
  });
  $("btn-voice-pending").addEventListener("click", () => {
    const p = activeProject();
    if (p) runPendingVoices(p.id);
  });
  $("btn-vr-done").addEventListener("click", async () => {
    if (Object.keys(VR.edits).length && !confirm("Có câu đã sửa chữ nhưng chưa bấm Tạo lại. Vẫn chạy tiếp (bỏ các chỗ sửa)?")) return;
    try {
      await api(`/api/projects/${VR.projectId}/voice-review/done`, { method: "POST" });
      $("vr-modal").classList.add("hidden");
      showToast("Đã ghi nhận kiểm tra xong — chạy tiếp ghép audio + xóa sub + xuất video.", "info", 6000);
      startBatchPolling();
    } catch (e) {
      showToast(e.message);
    }
  });
}

init();
