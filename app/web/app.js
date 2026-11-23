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

async function api(url, opts = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || `HTTP ${res.status}`);
  }
  return res.json();
}

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
}

async function loadHardware() {
  try {
    const hw = await api("/api/hardware");
    state.hardware = hw;
    $("hw-enc-pill").textContent = hw.recommended_encoder.toUpperCase();
    const dualTxt = hw.dual_gpu_capable ? "Dual-GPU" : `${hw.cpu_cores}C/${hw.cpu_threads}T`;
    $("hw-text").textContent = `${dualTxt} • ${hw.ram_gb}GB RAM`;
    $("hw-badge").title = `${hw.summary}\nGPUs: ${(hw.gpus || []).join(", ")}\nFFmpeg: ${hw.ffmpeg_version}`;
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
  $("clean-srt-model").value = "gemini-3.8-flash";
  $("dl-clean-model").innerHTML = modelOptions;
  $("dl-clean-model").value = "gemini-3.8-flash";

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
  const inner = $("ov-subtitle-inner");
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

  inner.textContent = getCurrentSubtitleText();
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

function syncAllOverlays() {
  syncSubtitleVisual();
  syncLogoVisual();
  syncGreenScreenVisual();
  syncSubRemovalVisual();
  syncAudioPreview();
}

function selectLayer(layerName) {
  state.activeLayer = layerName;
  document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c.dataset.layer === layerName));
  document.querySelectorAll(".ov-box").forEach((b) => b.classList.toggle("selected", b.dataset.layer === layerName));
}

// ---------------- DRAG & 8-HANDLE INDEPENDENT WxH RESIZE ----------------

function initOverlayDragResize() {
  let dragState = null;

  const getBoxRef = (layer) => {
    const proj = activeProject();
    if (!proj) return null;
    if (layer === "subtitle") return proj.config.subtitle.box;
    if (layer === "logo") return proj.config.logo.box;
    if (layer === "greenscreen") return proj.config.green_screen.box;
    if (layer === "subremoval") return proj.config.sub_removal.box;
    return null;
  };

  document.querySelectorAll(".ov-box").forEach((boxEl) => {
    boxEl.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const layer = boxEl.dataset.layer;
      selectLayer(layer);
      const boxRef = getBoxRef(layer);
      if (!boxRef) return;

      const wrap = $("video-canvas-wrap").getBoundingClientRect();
      const handle = e.target.closest(".handle");
      dragState = {
        layer,
        mode: handle ? "resize" : "move",
        dir: handle ? handle.dataset.dir : "",
        startX: e.clientX,
        startY: e.clientY,
        orig: { ...boxRef },
        wrapW: wrap.width,
        wrapH: wrap.height,
      };
    });
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragState) return;
    const boxRef = getBoxRef(dragState.layer);
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
  if (state.models.some((m) => m.id === c.translate.model)) {
    $("tr-model").value = c.translate.model;
  } else {
    $("tr-model").value = "gemini-4-neon";
    c.translate.model = "gemini-4-neon";
  }
  $("tr-thinking").value = c.translate.thinking_level === "max" ? "xhigh" : c.translate.thinking_level;
  $("tr-batch").value = c.translate.batch_size;
  $("tr-tol").value = c.translate.length_tolerance_pct;
  $("tr-concurrency").value = c.translate.concurrency || 4;
  $("tr-api-key").value = c.translate.api_key || "";

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
}

function scheduleAutoSave() {
  const proj = activeProject();
  if (!proj) return;
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => flushSave(proj).catch(console.error), 250);
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

  // Download Modal (YouTube / BiliBili)
  const openDlModal = () => {
    $("dl-modal").classList.remove("hidden");
    $("dl-status-box").classList.add("hidden");
  };
  const closeDlModal = () => $("dl-modal").classList.add("hidden");
  $("btn-open-dl-modal").addEventListener("click", openDlModal);
  $("btn-open-dl-hero").addEventListener("click", openDlModal);
  $("btn-close-dl-modal").addEventListener("click", closeDlModal);
  $("btn-cancel-dl").addEventListener("click", closeDlModal);
  $("dl-clean-ai").addEventListener("change", (e) => {
    $("dl-ai-opts").classList.toggle("hidden", !e.target.checked);
  });

  $("btn-start-dl").addEventListener("click", async () => {
    const url = $("dl-url").value.trim();
    if (!url) return alert("Vui lòng dán link YouTube hoặc BiliBili!");
    const stBox = $("dl-status-box");
    stBox.classList.remove("hidden");
    stBox.textContent = "Đang khởi tạo tải đa luồng...";
    $("btn-start-dl").disabled = true;

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
          api_key: $("tr-api-key").value || "",
          preset_id: $("preset-select").value || null,
        }),
      });

      clearInterval(state.dlPollTimer);
      state.dlPollTimer = setInterval(async () => {
        try {
          const st = await api(`/api/download/${job.job_id}/status`);
          const cleanMsg = cleanUiText(st.message);
          stBox.textContent = `[${Math.round(st.progress)}%] ${cleanMsg}`;
          if (st.status === "done") {
            clearInterval(state.dlPollTimer);
            $("btn-start-dl").disabled = false;
            if (st.project) {
              state.projects.push(st.project);
              renderVideoList();
              selectProject(st.project.id);
            }
            setTimeout(() => closeDlModal(), 900);
          } else if (st.status === "error") {
            clearInterval(state.dlPollTimer);
            $("btn-start-dl").disabled = false;
            stBox.textContent = `[${Math.round(st.progress)}%] ${cleanMsg}`;
          }
        } catch {
          clearInterval(state.dlPollTimer);
          $("btn-start-dl").disabled = false;
        }
      }, 700);
    } catch (e) {
      stBox.textContent = `Lỗi: ${cleanUiText(e.message)}`;
      $("btn-start-dl").disabled = false;
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
    if (!cfg.voice_id) return alert("Hãy chọn một giọng đọc trong danh sách trước!");
    const btn = $("btn-preview-voice");
    btn.textContent = "⏳ Đang tạo mẫu...";
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
      fn(proj, e.target);
      scheduleAutoSave();
    });
  };

  bind("sub-enabled", "change", (p, el) => { p.config.subtitle.enabled = el.checked; syncSubtitleVisual(); });
  bind("sub-font-select", "change", (p, el) => { p.config.subtitle.font_path = el.value; syncSubtitleVisual(); });
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
  bind("tr-provider", "change", (p, el) => { p.config.translate.provider = el.value; });
  bind("tr-model", "change", (p, el) => { p.config.translate.model = el.value; });
  bind("tr-thinking", "change", (p, el) => { p.config.translate.thinking_level = el.value; });
  bind("tr-batch", "input", (p, el) => { p.config.translate.batch_size = Number(el.value) || 45; });
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
  bind("voice-concurrency", "input", (p, el) => { p.config.voice.concurrency = Number(el.value); $("val-voice-conc").textContent = `${el.value} luồng`; });
  bind("voice-tiktok-sid", "input", (p, el) => { p.config.voice.tiktok_session_id = el.value; });
  bind("voice-fallback-edge", "change", (p, el) => { p.config.voice.fallback_edge_tts = el.checked; });

  // Render buttons
  $("btn-render-sample").addEventListener("click", () => triggerRender(8.0));
  $("btn-render-full").addEventListener("click", () => triggerRender(null));
}

init();
