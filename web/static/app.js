(() => {
  const $ = (s) => document.querySelector(s);
  const toastEl = $("#toast");
  const listEl = $("#imgList");
  const outListEl = $("#outList");
  const textListEl = $("#textList");
  const loadingEl = $("#loading");
  const metaBox = $("#metaBox");
  const healthPill = $("#healthPill");
  const currentNameEl = $("#currentName");
  const dirtyPill = $("#dirtyPill");
  const polyHint = $("#polyHint"); // optional
  const btnSave = $("#btnSave");
  const inputDirLabel = $("#inputDirLabel");
  const outputDirLabel = $("#outputDirLabel");
  const textsDirLabel = $("#textsDirLabel");
  const folderModal = $("#folderModal");
  const browseList = $("#browseList");
  const browsePathInput = $("#browsePathInput");
  const browseMeta = $("#browseMeta");
  const imgSearch = $("#imgSearch");

  let browseCwd = "";
  let cachedImages = [];
  let cachedOutputs = [];
  let cachedTexts = [];

  const baseCanvas = $("#baseCanvas");
  const maskCanvas = $("#maskCanvas");
  const uiCanvas = $("#uiCanvas");
  const baseCtx = baseCanvas.getContext("2d");
  const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
  const uiCtx = uiCanvas.getContext("2d");

  const CLOSE_DIST = 12; // image-pixel radius to click-close near first vertex

  const state = {
    items: [],
    stem: null,
    fileName: null,
    tool: "poly",
    brush: 5,
    alpha: 0.45,
    drawing: false,
    last: null,
    boxStart: null,
    dirty: false,
    history: [],
    imgW: 0,
    imgH: 0,
    fitScale: 1, // 适应窗口的基准缩放
    zoom: 1, // 相对 fitScale 的放大倍数
    viewScale: 1, // = fitScale * zoom
    panning: false,
    panOrigin: null,
    spaceDown: false,
    polyPoints: [], // [{x,y}, ...]
    polyHover: null,
    lastClickTs: 0,
  };

  function toast(msg, err = false) {
    toastEl.textContent = msg;
    toastEl.classList.toggle("err", !!err);
    toastEl.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => toastEl.classList.remove("show"), 2800);
  }

  function setLoading(on, text = "处理中…") {
    loadingEl.textContent = text;
    loadingEl.classList.toggle("show", !!on);
  }

  function setDirty(v) {
    state.dirty = !!v;
    if (dirtyPill) {
      dirtyPill.textContent = state.dirty ? "未保存" : "已保存";
      dirtyPill.classList.toggle("is-dirty", state.dirty);
    }
    if (btnSave) btnSave.classList.toggle("needs-save", state.dirty);
  }

  function updateCurrentBar(name) {
    if (currentNameEl) currentNameEl.textContent = name || "未选择";
  }

  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const j = await res.json();
        detail = j.detail || JSON.stringify(j);
      } catch (_) {}
      throw new Error(detail);
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) return res.json();
    return res;
  }

  function syncCanvasSize(w, h) {
    const wrap = $("#canvasWrap");
    const maxW = Math.max(320, wrap.clientWidth - 48);
    const maxH = Math.max(320, wrap.clientHeight - 48);
    state.fitScale = Math.min(maxW / w, maxH / h, 1.5);
    state.zoom = 1;
    // 只在换图时重置像素缓冲尺寸
    for (const c of [baseCanvas, maskCanvas, uiCanvas]) {
      c.width = w;
      c.height = h;
    }
    applyViewTransform();
  }

  function applyViewTransform(anchorClientX, anchorClientY) {
    if (!state.imgW || !state.imgH) return;
    const wrap = $("#canvasWrap");
    const prevScale = state.viewScale || state.fitScale || 1;
    state.zoom = Math.min(8, Math.max(0.25, state.zoom));
    state.viewScale = state.fitScale * state.zoom;
    const dw = Math.max(1, Math.round(state.imgW * state.viewScale));
    const dh = Math.max(1, Math.round(state.imgH * state.viewScale));

    // 相对锚点缩放，保持鼠标下的图像点不跳
    let ax = null;
    let ay = null;
    if (typeof anchorClientX === "number" && wrap) {
      const rect = wrap.getBoundingClientRect();
      ax = wrap.scrollLeft + (anchorClientX - rect.left);
      ay = wrap.scrollTop + (anchorClientY - rect.top);
    }

    for (const c of [baseCanvas, maskCanvas, uiCanvas]) {
      c.style.width = dw + "px";
      c.style.height = dh + "px";
    }
    const stage = $("#stage");
    stage.style.width = dw + "px";
    stage.style.height = dh + "px";

    const zoomLabel = $("#zoomLabel");
    if (zoomLabel) zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;

    if (ax != null && prevScale > 0) {
      const ratio = state.viewScale / prevScale;
      wrap.scrollLeft = ax * ratio - (anchorClientX - wrap.getBoundingClientRect().left);
      wrap.scrollTop = ay * ratio - (anchorClientY - wrap.getBoundingClientRect().top);
    }
  }

  function zoomBy(factor, clientX, clientY) {
    state.zoom *= factor;
    applyViewTransform(clientX, clientY);
  }

  function zoomToFit() {
    state.zoom = 1;
    applyViewTransform();
    const wrap = $("#canvasWrap");
    if (wrap) {
      wrap.scrollLeft = 0;
      wrap.scrollTop = 0;
    }
  }

  function pushHistory() {
    try {
      // 存 ImageData，避免 toDataURL 把透明背景压成不透明黑，
      // 再被导出逻辑误判成「整图掩码」。
      const snap = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
      state.history.push(snap);
      if (state.history.length > 30) state.history.shift();
    } catch (_) {}
  }

  function undo() {
    const prev = state.history.pop();
    if (!prev) return toast("没有可撤销的操作");
    try {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      maskCtx.putImageData(prev, 0, 0);
      setDirty(true);
      paintMaskTint();
    } catch (e) {
      toast("撤销失败", true);
    }
  }

  function paintMaskTint() {
    maskCanvas.style.opacity = String(state.alpha);
  }

  function loadMaskFromB64(b64) {
    return new Promise((resolve, reject) => {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      if (!b64) {
        paintMaskTint();
        resolve();
        return;
      }
      const img = new Image();
      img.onload = () => {
        try {
          const off = document.createElement("canvas");
          off.width = maskCanvas.width;
          off.height = maskCanvas.height;
          const octx = off.getContext("2d");
          octx.drawImage(img, 0, 0, off.width, off.height);
          const data = octx.getImageData(0, 0, off.width, off.height);
          const d = data.data;
          // 二值掩码：亮色=前景。不能用 alpha>200 判断——
          // 清空后的全黑不透明 PNG 会被误判成整图绿色掩码。
          for (let i = 0; i < d.length; i += 4) {
            const v = Math.max(d[i], d[i + 1], d[i + 2]);
            if (v > 127) {
              d[i] = 0;
              d[i + 1] = 220;
              d[i + 2] = 120;
              d[i + 3] = 255;
            } else {
              d[i] = d[i + 1] = d[i + 2] = 0;
              d[i + 3] = 0;
            }
          }
          octx.putImageData(data, 0, 0);
          maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
          maskCtx.drawImage(off, 0, 0);
          paintMaskTint();
          resolve();
        } catch (e) {
          reject(e);
        }
      };
      img.onerror = () => reject(new Error("掩码图像加载失败"));
      img.src = b64;
    });
  }

  async function exportMaskPngB64() {
    const w = maskCanvas.width;
    const h = maskCanvas.height;
    const src = maskCtx.getImageData(0, 0, w, h);
    const out = maskCtx.createImageData(w, h);
    // 画布前景是 rgba(0,220,120,a)。背景必须是透明。
    // 禁止「仅凭 alpha」判定——不透明黑底会被判成整图掩码。
    for (let i = 0; i < src.data.length; i += 4) {
      const r = src.data[i];
      const g = src.data[i + 1];
      const b = src.data[i + 2];
      const a = src.data[i + 3];
      const nearBlack = r < 40 && g < 40 && b < 40;
      const on = a > 20 && !nearBlack && (g > 100 || Math.max(r, g, b) > 127);
      const v = on ? 255 : 0;
      out.data[i] = out.data[i + 1] = out.data[i + 2] = v;
      out.data[i + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = w;
    tmp.height = h;
    tmp.getContext("2d").putImageData(out, 0, 0);
    return tmp.toDataURL("image/png");
  }

  function canvasPos(ev) {
    const rect = uiCanvas.getBoundingClientRect();
    return {
      x: ((ev.clientX - rect.left) / rect.width) * uiCanvas.width,
      y: ((ev.clientY - rect.top) / rect.height) * uiCanvas.height,
    };
  }

  function strokeAt(x, y, erase) {
    maskCtx.save();
    maskCtx.lineCap = "round";
    maskCtx.lineJoin = "round";
    maskCtx.lineWidth = state.brush;
    if (erase) {
      maskCtx.globalCompositeOperation = "destination-out";
      maskCtx.strokeStyle = "rgba(0,0,0,1)";
    } else {
      maskCtx.globalCompositeOperation = "source-over";
      maskCtx.strokeStyle = "rgba(0,220,120,1)";
    }
    maskCtx.beginPath();
    if (state.last) {
      maskCtx.moveTo(state.last.x, state.last.y);
      maskCtx.lineTo(x, y);
    } else {
      maskCtx.moveTo(x, y);
      maskCtx.lineTo(x + 0.01, y + 0.01);
    }
    maskCtx.stroke();
    maskCtx.restore();
    state.last = { x, y };
    setDirty(true);
  }

  // ---------- Polygon (Labelme-style) ----------
  function clearPoly() {
    state.polyPoints = [];
    state.polyHover = null;
    drawPolyPreview();
  }

  function drawPolyPreview() {
    uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
    const pts = state.polyPoints;
    if (!pts.length && !state.polyHover) return;

    uiCtx.save();
    uiCtx.lineWidth = 2;
    uiCtx.strokeStyle = "#ef4444";
    uiCtx.fillStyle = "rgba(239, 68, 68, 0.18)";

    if (pts.length) {
      uiCtx.beginPath();
      uiCtx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) uiCtx.lineTo(pts[i].x, pts[i].y);
      if (state.polyHover) uiCtx.lineTo(state.polyHover.x, state.polyHover.y);
      if (pts.length >= 3) {
        uiCtx.closePath();
        uiCtx.fill();
      }
      uiCtx.stroke();

      // vertices
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        uiCtx.beginPath();
        uiCtx.arc(p.x, p.y, i === 0 ? 5 : 3.5, 0, Math.PI * 2);
        uiCtx.fillStyle = i === 0 ? "#f59e0b" : "#ef4444";
        uiCtx.fill();
        uiCtx.strokeStyle = "#fff";
        uiCtx.lineWidth = 1.5;
        uiCtx.stroke();
      }

      // close hint near first point
      if (pts.length >= 3) {
        uiCtx.beginPath();
        uiCtx.setLineDash([4, 3]);
        uiCtx.strokeStyle = "#f59e0b";
        uiCtx.arc(pts[0].x, pts[0].y, CLOSE_DIST, 0, Math.PI * 2);
        uiCtx.stroke();
        uiCtx.setLineDash([]);
      }
    }
    uiCtx.restore();
  }

  function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function commitPolygon() {
    const pts = state.polyPoints;
    if (pts.length < 3) {
      toast("至少需要 3 个点才能闭合", true);
      return;
    }
    pushHistory();
    const mode = ($("#polyMode") && $("#polyMode").value) || "replace";

    // draw filled polygon onto an offscreen binary, then merge
    const off = document.createElement("canvas");
    off.width = maskCanvas.width;
    off.height = maskCanvas.height;
    const octx = off.getContext("2d");
    octx.clearRect(0, 0, off.width, off.height);
    octx.beginPath();
    octx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) octx.lineTo(pts[i].x, pts[i].y);
    octx.closePath();
    octx.fillStyle = "#fff";
    octx.fill();

    if (mode === "replace") {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      // convert white fill to green mask
      const data = octx.getImageData(0, 0, off.width, off.height);
      const d = data.data;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i] > 127) {
          d[i] = 0;
          d[i + 1] = 220;
          d[i + 2] = 120;
          d[i + 3] = 255;
        } else {
          d[i + 3] = 0;
        }
      }
      octx.putImageData(data, 0, 0);
      maskCtx.drawImage(off, 0, 0);
    } else if (mode === "add") {
      maskCtx.save();
      maskCtx.globalCompositeOperation = "source-over";
      maskCtx.beginPath();
      maskCtx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) maskCtx.lineTo(pts[i].x, pts[i].y);
      maskCtx.closePath();
      maskCtx.fillStyle = "rgba(0,220,120,1)";
      maskCtx.fill();
      maskCtx.restore();
    } else {
      // subtract / dig hole
      maskCtx.save();
      maskCtx.globalCompositeOperation = "destination-out";
      maskCtx.beginPath();
      maskCtx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) maskCtx.lineTo(pts[i].x, pts[i].y);
      maskCtx.closePath();
      maskCtx.fillStyle = "rgba(0,0,0,1)";
      maskCtx.fill();
      maskCtx.restore();
    }

    paintMaskTint();
    clearPoly();
    setDirty(true);
    toast("多边形已写入画布，请点「保存当前图片」落盘");
  }

  function setTool(name) {
    if (name === "pan") name = "poly"; // 平移已改为 Ctrl+滚轮缩放 / 中键拖拽
    if (state.tool === "poly" && name !== "poly") clearPoly();
    state.tool = name;
    document.querySelectorAll(".tool").forEach((b) => {
      b.classList.toggle("active", b.dataset.tool === name);
    });
    updateCursor();
    if (polyHint) polyHint.classList.toggle("hidden", name !== "poly");
  }

  function updateCursor() {
    const wrap = $("#canvasWrap");
    if (state.panning || state.spaceDown) {
      uiCanvas.style.cursor = "grab";
      if (wrap) wrap.classList.toggle("panning", state.panning);
      return;
    }
    uiCanvas.style.cursor = "crosshair";
    if (wrap) wrap.classList.remove("panning");
  }

  function wantPan(ev) {
    // 中键 / Space+左键：拖动画布；不切换标注工具
    return ev.button === 1 || (ev.button === 0 && (state.spaceDown || ev.altKey));
  }

  async function refreshOutputs() {
    const data = await api(`/api/outputs?t=${Date.now()}`);
    if (outputDirLabel) {
      outputDirLabel.textContent = shortPath(data.mask_dir || "outputs/masks");
      outputDirLabel.title = data.mask_dir || "";
    }
    if (textsDirLabel) {
      const refs = data.refs_txt || data.texts_dir || "outputs/refs.txt";
      textsDirLabel.textContent = shortPath(refs);
      textsDirLabel.title = refs;
    }

    cachedOutputs = data.items || [];
    cachedTexts = data.texts || [];
    renderOutputList();
    renderTextList();
  }

  function renderOutputList() {
    if (!outListEl) return;
    outListEl.innerHTML = "";
    const items = cachedOutputs;
    if (!items.length) {
      const li = document.createElement("li");
      li.style.cursor = "default";
      li.innerHTML = `<span class="name" style="color:#8a8276">暂无保存的掩码</span>`;
      outListEl.appendChild(li);
      return;
    }
    items.forEach((it) => {
      const li = document.createElement("li");
      li.className = "out-item";
      li.dataset.stem = it.stem;
      if (it.stem === state.stem) li.classList.add("active");
      li.innerHTML = `
        <img class="out-thumb" src="${it.mask_url}?t=${Date.now()}" alt="mask" title="二值掩码 PNG" />
        <div class="out-meta">
          <div style="display:flex;justify-content:space-between;gap:6px;align-items:center">
            <span class="name" title="${it.image_name}">${it.image_name}</span>
            <span class="badge ${it.status}">${it.status}</span>
          </div>
          <span class="txt-line">${it.mask_name || ""}</span>
        </div>`;
      // 点击：只预览掩码 PNG，不打开「原图+绿罩」编辑视图
      li.onclick = () => previewMaskPng(it);
      outListEl.appendChild(li);
    });
    syncSideSelection(state.stem);
  }

  function renderTextList() {
    if (!textListEl) return;
    textListEl.innerHTML = "";
    const texts = cachedTexts;
    if (!texts.length) {
      const li = document.createElement("li");
      li.style.cursor = "default";
      li.innerHTML = `<span class="name" style="color:#8a8276">暂无文本（保存后生成）</span>`;
      textListEl.appendChild(li);
      return;
    }
    texts.forEach((it) => {
      const li = document.createElement("li");
      li.dataset.stem = it.stem;
      if (it.stem === state.stem) li.classList.add("active");
      const imgName = it.image_name || `${it.stem}.jpg`;
      const desc = (it.prompt || "").trim() || extractPromptFromLine(it.text_line, imgName);
      li.innerHTML = `
        <span class="txt-img-name" title="${escapeHtml(imgName)}">${escapeHtml(imgName)}</span>
        <span class="txt-desc ${desc ? "" : "txt-empty"}">${escapeHtml(desc || "（无描述）")}</span>`;
      li.onclick = () => selectImage(it.stem);
      textListEl.appendChild(li);
    });
    syncSideSelection(state.stem);
  }

  function syncSideSelection(stem) {
    if (!stem) return;
    [listEl, outListEl, textListEl].forEach((ul) => {
      if (!ul) return;
      ul.querySelectorAll("li[data-stem]").forEach((li) => {
        const on = li.dataset.stem === stem;
        li.classList.toggle("active", on);
        if (on) {
          try {
            li.scrollIntoView({ block: "nearest", behavior: "smooth" });
          } catch (_) {
            li.scrollIntoView(false);
          }
        }
      });
    });
  }

  function extractPromptFromLine(line, imgName) {
    const s = (line || "").trim();
    if (!s) return "";
    if (imgName && s.startsWith(imgName)) return s.slice(imgName.length).trim();
    const parts = s.split(/\s+/);
    if (parts.length <= 1) return "";
    return parts.slice(1).join(" ");
  }

  function renderImageList() {
    const q = ((imgSearch && imgSearch.value) || "").trim().toLowerCase();
    listEl.innerHTML = "";
    const items = cachedImages.filter((it) => {
      if (!q) return true;
      return (it.name || "").toLowerCase().includes(q) || (it.stem || "").toLowerCase().includes(q);
    });
    if (!items.length) {
      const li = document.createElement("li");
      li.style.cursor = "default";
      li.innerHTML = `<span class="name" style="color:#8a8276">${q ? "无匹配图片" : "文件夹内无图片"}</span>`;
      listEl.appendChild(li);
      return;
    }
    items.forEach((it) => {
      const li = document.createElement("li");
      li.dataset.stem = it.stem;
      if (it.stem === state.stem) li.classList.add("active");
      const extra = it.has_text ? " ·txt" : "";
      li.innerHTML = `<span class="name" title="${it.name}">${it.name}</span><span class="badge ${it.status}">${it.status}${extra}</span>`;
      li.onclick = () => selectImage(it.stem);
      listEl.appendChild(li);
    });
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function shortPath(p) {
    if (!p) return "";
    const parts = String(p).split("/");
    return parts.slice(-2).join("/") + (String(p).endsWith("/") ? "/" : "");
  }

  async function refreshList() {
    const data = await api(`/api/images?t=${Date.now()}`);
    state.items = data.items || [];
    cachedImages = state.items;
    if (inputDirLabel) {
      inputDirLabel.textContent = shortPath(data.input_dir) || "raw_images/";
      inputDirLabel.title = data.input_dir || "";
    }
    if (outputDirLabel && data.mask_dir) {
      outputDirLabel.textContent = shortPath(data.mask_dir);
      outputDirLabel.title = data.mask_dir;
    }
    if (textsDirLabel && (data.refs_txt || data.texts_dir)) {
      const refs = data.refs_txt || data.texts_dir;
      textsDirLabel.textContent = shortPath(refs);
      textsDirLabel.title = refs;
    }
    renderImageList();
    await refreshOutputs();
    if (!state.stem && state.items.length) selectImage(state.items[0].stem);
  }

  async function openFolderBrowser(startPath) {
    if (!folderModal) return;
    folderModal.classList.remove("hidden");
    await loadBrowse(startPath || undefined);
  }

  function closeFolderBrowser() {
    if (folderModal) folderModal.classList.add("hidden");
  }

  async function loadBrowse(path) {
    const q = path ? `?path=${encodeURIComponent(path)}` : "";
    const data = await api(`/api/browse${q}`);
    browseCwd = data.cwd;
    if (browsePathInput) browsePathInput.value = data.cwd;
    if (browseMeta) {
      browseMeta.textContent = `子文件夹 ${data.dirs.length} 个 · 图片 ${data.image_count} 张`;
    }
    if (!browseList) return;
    browseList.innerHTML = "";
    if (data.parent) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="ico">⬆</span><span>..（上级目录）</span>`;
      li.onclick = () => loadBrowse(data.parent);
      browseList.appendChild(li);
    }
    data.dirs.forEach((d) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="ico">📁</span><span>${d.name}</span>`;
      li.onclick = () => loadBrowse(d.path);
      browseList.appendChild(li);
    });
    data.images.slice(0, 30).forEach((f) => {
      const li = document.createElement("li");
      li.style.opacity = "0.75";
      li.style.cursor = "default";
      li.innerHTML = `<span class="ico">🖼</span><span>${f.name}</span>`;
      browseList.appendChild(li);
    });
  }

  async function useCurrentBrowseFolder() {
    const path = (browsePathInput && browsePathInput.value.trim()) || browseCwd;
    if (!path) return toast("请选择文件夹", true);
    setLoading(true, "切换输入文件夹…");
    try {
      const res = await api("/api/set_dirs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input_dir: path }),
      });
      state.stem = null;
      closeFolderBrowser();
      await refreshList();
      toast(`已打开文件夹（${(res.items || []).length} 张图）`);
    } catch (e) {
      toast(String(e.message || e), true);
    } finally {
      setLoading(false);
    }
  }

  async function previewMaskPng(it) {
    /** 点击「输出掩码」：主画布只显示二值掩码 PNG，不叠原图。 */
    if (state.dirty && state.stem && state.stem !== it.stem) {
      if (!confirm("当前图片尚未保存，切换将丢失本地修改。继续？")) return;
    }
    setLoading(true, "加载掩码 PNG…");
    try {
      clearPoly();
      const item = await api(`/api/item/${encodeURIComponent(it.stem)}?t=${Date.now()}`);
      state.stem = it.stem;
      state.fileName = item.name;
      state.history = [];
      state.imgW = item.width;
      state.imgH = item.height;
      state.viewMode = "mask_only";
      syncCanvasSize(item.width, item.height);
      updateCurrentBar(`${item.name} · 掩码PNG`);
      setDirty(false);

      const img = new Image();
      img.crossOrigin = "anonymous";
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = (it.mask_url || `/api/mask/${it.stem}`) + "?t=" + Date.now();
      });
      baseCtx.fillStyle = "#000";
      baseCtx.fillRect(0, 0, baseCanvas.width, baseCanvas.height);
      baseCtx.drawImage(img, 0, 0, item.width, item.height);
      // 不叠绿色罩层
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
      paintMaskTint();

      if (item.meta?.prompt) $("#prompt").value = item.meta.prompt;
      metaBox.textContent = JSON.stringify(
        {
          view: "mask_png_only",
          stem: item.stem,
          mask: `${it.stem}.png`,
          size: `${item.width}x${item.height}`,
          prompt: item.meta?.prompt || "",
        },
        null,
        2
      );
      await refreshList();
      syncSideSelection(it.stem);
      toast(`掩码 PNG：${it.stem}.png（未叠原图）`);
    } catch (e) {
      toast(String(e.message || e), true);
    } finally {
      setLoading(false);
    }
  }

  async function selectImage(stem) {
    if (state.dirty && state.stem && state.stem !== stem) {
      if (!confirm("当前图片尚未保存，切换将丢失本地修改。继续？")) return;
    }
    setLoading(true, "加载图像…");
    try {
      clearPoly();
      const item = await api(`/api/item/${encodeURIComponent(stem)}?t=${Date.now()}`);
      state.stem = stem;
      state.fileName = item.name;
      state.history = [];
      state.imgW = item.width;
      state.imgH = item.height;
      state.viewMode = "edit";
      syncCanvasSize(item.width, item.height);
      updateCurrentBar(item.name);
      setDirty(false);

      const img = new Image();
      img.crossOrigin = "anonymous";
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = item.image_url + "?t=" + Date.now();
      });
      baseCtx.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
      baseCtx.drawImage(img, 0, 0, item.width, item.height);
      await loadMaskFromB64(item.mask_b64);
      uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);

      if (item.meta?.prompt) $("#prompt").value = item.meta.prompt;
      metaBox.textContent = JSON.stringify(
        {
          stem: item.stem,
          status: item.status,
          size: `${item.width}x${item.height}`,
          prompt: item.meta?.prompt || "",
          human_edited: item.meta?.human_edited,
          source: item.meta?.source,
          num_detections: item.meta?.num_detections,
        },
        null,
        2
      );
      await refreshList();
      syncSideSelection(stem);
      toast(`已打开 ${item.name}`);
    } catch (e) {
      toast(String(e.message || e), true);
    } finally {
      setLoading(false);
    }
  }

  async function reloadCurrentFromServer() {
    if (!state.stem) return;
    const item = await api(`/api/item/${encodeURIComponent(state.stem)}?t=${Date.now()}`);
    await loadMaskFromB64(item.mask_b64);
    if (item.meta?.prompt) $("#prompt").value = item.meta.prompt;
    updateCurrentBar(item.name);
    metaBox.textContent = JSON.stringify(
      {
        stem: item.stem,
        status: item.status,
        prompt: item.meta?.prompt || "",
        human_edited: item.meta?.human_edited,
        source: item.meta?.source,
        num_detections: item.meta?.num_detections,
      },
      null,
      2
    );
    return item;
  }

  async function saveMask() {
    if (!state.stem) return toast("请先选择图像", true);
    if (state.polyPoints.length) {
      if (!confirm("还有未闭合的多边形，是否丢弃并保存当前掩码？")) return;
      clearPoly();
    }
    setLoading(true, `正在保存 ${state.fileName || state.stem} …`);
    try {
      const b64 = await exportMaskPngB64();
      const res = await api("/api/save_mask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stem: state.stem,
          mask_png_b64: b64,
          human_edited: true,
          prompt: $("#prompt").value,
        }),
      });
      setDirty(false);
      await reloadCurrentFromServer();
      await refreshList();
      const textHint = res.text_path ? `；文本 → refs.txt` : "";
      toast(`已保存当前图片：${state.fileName}${textHint}`);
      metaBox.textContent = JSON.stringify(res, null, 2);
    } catch (e) {
      toast(String(e.message || e), true);
    } finally {
      setLoading(false);
    }
  }

  async function runAuto() {
    if (!state.stem) return toast("请先选择图像", true);
    setLoading(true, "自动标注中（首次会加载模型）…");
    try {
      clearPoly();
      const res = await api("/api/auto_annotate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stem: state.stem,
          prompt: $("#prompt").value,
          box_threshold: parseFloat($("#boxThr").value),
          text_threshold: parseFloat($("#textThr").value),
        }),
      });
      state.history = [];
      await loadMaskFromB64(res.mask_b64);
      setDirty(false);
      await refreshList();
      metaBox.textContent = JSON.stringify(res, null, 2);
      toast(`自动标注已写入（${res.num_detections || 0} 个）。若再手改，请点「保存当前图片」`);
    } catch (e) {
      toast(String(e.message || e), true);
    } finally {
      setLoading(false);
    }
  }

  async function clearServerMask() {
    if (!state.stem) return;
    if (!confirm("清空当前图片的掩码？")) return;
    setLoading(true, "清空中…");
    try {
      clearPoly();
      const res = await api(`/api/clear_mask?stem=${encodeURIComponent(state.stem)}`, { method: "POST" });
      state.history = [];
      // 强制清空画布层，再按服务器空掩码重载，避免残留绿色叠层
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
      await loadMaskFromB64(res.mask_b64);
      // 再刷一次原图，防止误解「绿色还在」
      const img = new Image();
      img.crossOrigin = "anonymous";
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = `/api/image/${encodeURIComponent(state.stem)}?t=${Date.now()}`;
      });
      baseCtx.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
      baseCtx.drawImage(img, 0, 0, baseCanvas.width, baseCanvas.height);
      setDirty(false);
      await refreshList();
      toast("掩码已清空（只去绿色叠层，原图本身不变）");
    } catch (e) {
      toast(String(e.message || e), true);
    } finally {
      setLoading(false);
    }
  }

  async function refineBox(xyxy) {
    setLoading(true, "SAM 框选重标…");
    try {
      const res = await api("/api/refine_box", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stem: state.stem,
          box_xyxy: xyxy,
          mode: $("#boxMode").value,
        }),
      });
      pushHistory();
      await loadMaskFromB64(res.mask_b64);
      setDirty(false);
      await refreshList();
      toast("框选 SAM 完成并已保存");
    } catch (e) {
      toast(String(e.message || e), true);
    } finally {
      setLoading(false);
    }
  }

  async function refinePoint(x, y) {
    setLoading(true, "SAM 点选重标…");
    try {
      const res = await api("/api/refine_point", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stem: state.stem,
          x,
          y,
          label: 1,
          mode: "add",
        }),
      });
      pushHistory();
      await loadMaskFromB64(res.mask_b64);
      setDirty(false);
      await refreshList();
      toast("点选 SAM 完成并已保存");
    } catch (e) {
      toast(String(e.message || e), true);
    } finally {
      setLoading(false);
    }
  }

  // Ctrl + 滚轮：缩放；中键/Space+拖：平移
  const canvasWrapEl = $("#canvasWrap");
  if (canvasWrapEl) {
    canvasWrapEl.addEventListener(
      "wheel",
      (ev) => {
        if (!state.stem) return;
        if (!(ev.ctrlKey || ev.metaKey)) return;
        ev.preventDefault();
        const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
        zoomBy(factor, ev.clientX, ev.clientY);
      },
      { passive: false }
    );
  }

  // pointer handlers
  uiCanvas.addEventListener("pointerdown", (ev) => {
    if (!state.stem) return;
    uiCanvas.setPointerCapture(ev.pointerId);
    const p = canvasPos(ev);

    if (wantPan(ev)) {
      ev.preventDefault();
      state.panning = true;
      state.panOrigin = {
        x: ev.clientX,
        y: ev.clientY,
        sl: $("#canvasWrap").scrollLeft,
        st: $("#canvasWrap").scrollTop,
      };
      updateCursor();
      return;
    }

    if (state.tool === "poly") {
      // right click = close
      if (ev.button === 2) {
        ev.preventDefault();
        commitPolygon();
        return;
      }
      const now = Date.now();
      const isDouble = now - state.lastClickTs < 320;
      state.lastClickTs = now;

      // click near first point to close
      if (
        state.polyPoints.length >= 3 &&
        dist(p, state.polyPoints[0]) <= CLOSE_DIST
      ) {
        commitPolygon();
        return;
      }

      if (isDouble && state.polyPoints.length >= 3) {
        // double-click closes (ignore the last accidental point if very close to previous)
        commitPolygon();
        return;
      }

      state.polyPoints.push(p);
      drawPolyPreview();
      return;
    }

    if (state.tool === "brush" || state.tool === "eraser") {
      pushHistory();
      state.drawing = true;
      state.last = null;
      strokeAt(p.x, p.y, state.tool === "eraser");
      return;
    }
    if (state.tool === "box") {
      state.boxStart = p;
      return;
    }
    if (state.tool === "point") {
      refinePoint(p.x, p.y);
    }
  });

  uiCanvas.addEventListener("pointermove", (ev) => {
    const p = canvasPos(ev);
    if (state.panning && state.panOrigin) {
      const wrap = $("#canvasWrap");
      wrap.scrollLeft = state.panOrigin.sl - (ev.clientX - state.panOrigin.x);
      wrap.scrollTop = state.panOrigin.st - (ev.clientY - state.panOrigin.y);
      return;
    }
    if (state.tool === "poly" && state.polyPoints.length) {
      state.polyHover = p;
      drawPolyPreview();
      return;
    }
    if (state.drawing && (state.tool === "brush" || state.tool === "eraser")) {
      strokeAt(p.x, p.y, state.tool === "eraser");
      return;
    }
    if (state.boxStart && state.tool === "box") {
      uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
      const x = Math.min(state.boxStart.x, p.x);
      const y = Math.min(state.boxStart.y, p.y);
      const w = Math.abs(p.x - state.boxStart.x);
      const h = Math.abs(p.y - state.boxStart.y);
      uiCtx.strokeStyle = "#ef4444";
      uiCtx.lineWidth = 2;
      uiCtx.setLineDash([6, 4]);
      uiCtx.strokeRect(x, y, w, h);
      uiCtx.setLineDash([]);
    }
  });

  uiCanvas.addEventListener("pointerup", (ev) => {
    const p = canvasPos(ev);
    if (state.panning) {
      state.panning = false;
      state.panOrigin = null;
      updateCursor();
      return;
    }
    if (state.drawing) {
      state.drawing = false;
      state.last = null;
      return;
    }
    if (state.boxStart && state.tool === "box") {
      const x1 = Math.min(state.boxStart.x, p.x);
      const y1 = Math.min(state.boxStart.y, p.y);
      const x2 = Math.max(state.boxStart.x, p.x);
      const y2 = Math.max(state.boxStart.y, p.y);
      state.boxStart = null;
      uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
      if (x2 - x1 < 4 || y2 - y1 < 4) return toast("框太小", true);
      refineBox([x1, y1, x2, y2]);
    }
  });

  uiCanvas.addEventListener("contextmenu", (ev) => {
    if (state.tool === "poly") ev.preventDefault();
  });
  // 避免中键自动滚动
  uiCanvas.addEventListener("auxclick", (ev) => {
    if (ev.button === 1) ev.preventDefault();
  });

  // UI bindings
  document.querySelectorAll(".tool").forEach((b) => {
    b.addEventListener("click", () => setTool(b.dataset.tool));
  });
  $("#brushSize").addEventListener("input", (e) => {
    state.brush = parseInt(e.target.value, 10);
    $("#brushVal").textContent = state.brush;
  });
  $("#maskAlpha").addEventListener("input", (e) => {
    state.alpha = parseInt(e.target.value, 10) / 100;
    paintMaskTint();
  });
  const aiChatLog = $("#aiChatLog");
  let lastAIText = "";

  function appendAIChat(role, text) {
    if (!aiChatLog) return;
    const div = document.createElement("div");
    div.className = `msg role-${role === "user" ? "user" : "ai"}`;
    div.textContent = (role === "user" ? "你： " : "AI： ") + text;
    aiChatLog.appendChild(div);
    aiChatLog.scrollTop = aiChatLog.scrollHeight;
  }

  async function saveTextOnly() {
    if (!state.stem) return toast("请先选择图像", true);
    const prompt = ($("#prompt") && $("#prompt").value) || "";
    setLoading(true, "保存文本中…");
    try {
      const res = await api("/api/save_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stem: state.stem, prompt }),
      });
      await refreshOutputs();
      toast(`文本已写入 refs.txt（未改动掩码）：${state.fileName || state.stem}`);
      return res;
    } catch (e) {
      toast(String(e.message || e), true);
      throw e;
    } finally {
      setLoading(false);
    }
  }

  async function runAICaption(alsoSaveTxt) {
    if (!state.stem) return toast("请先选择图像", true);
    const instruction = ($("#aiInstruction") && $("#aiInstruction").value) || "";
    const maskDirty = state.dirty;
    appendAIChat("user", instruction || "(默认：根据绿色掩码生成英文 caption)");
    setLoading(true, "Qwen3.8 生成描述中（首次加载约需几分钟）…");
    try {
      const maskB64 = await exportMaskPngB64();
      const res = await api("/api/ai_caption", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stem: state.stem,
          instruction,
          mask_png_b64: maskB64,
          apply_to_prompt: true,
          auto_save_text: !!alsoSaveTxt,
          gpu: "1,2",
        }),
      });
      lastAIText = res.text || "";
      appendAIChat("ai", lastAIText + (res.elapsed_sec ? `  (${res.elapsed_sec}s)` : ""));
      if (res.apply_to_prompt !== false) {
        $("#prompt").value = lastAIText;
      }
      // 只填入文本框，不写盘、不改掩码；保存请点「填入并保存文本」
      setDirty(true);
      toast("AI 描述已填入文本框，请点「填入并保存文本」写入 refs.txt");
      await refreshOutputs();
    } catch (e) {
      appendAIChat("ai", "失败：" + String(e.message || e));
      toast(String(e.message || e), true);
    } finally {
      setLoading(false);
    }
  }

  function openDeleteModal() {
    if (!state.stem) return toast("请先选择图像", true);
    const name = state.fileName || state.stem;
    const body = $("#deleteModalBody");
    if (body) {
      body.innerHTML = `
        <p>确定删除 <strong>${escapeHtml(name)}</strong> 吗？</p>
        <ul class="delete-warn-list">
          <li>输入图片文件</li>
          <li>对应掩码（masks / corrected）</li>
          <li>refs.txt 里该图片对应的那一行文本（一并删除）</li>
        </ul>
        <p class="delete-warn-note">此操作不可恢复。</p>`;
    }
    const modal = $("#deleteModal");
    if (modal) modal.classList.remove("hidden");
  }

  function closeDeleteModal() {
    const modal = $("#deleteModal");
    if (modal) modal.classList.add("hidden");
  }

  async function confirmDeleteItem() {
    if (!state.stem) {
      closeDeleteModal();
      return toast("请先选择图像", true);
    }
    const name = state.fileName || state.stem;
    closeDeleteModal();
    setLoading(true, "删除中…");
    try {
      const res = await api(`/api/delete_item?stem=${encodeURIComponent(state.stem)}`, { method: "POST" });
      state.stem = null;
      state.fileName = null;
      state.dirty = false;
      state.history = [];
      clearPoly();
      baseCtx.clearRect(0, 0, baseCanvas.width || 1, baseCanvas.height || 1);
      maskCtx.clearRect(0, 0, maskCanvas.width || 1, maskCanvas.height || 1);
      uiCtx.clearRect(0, 0, uiCanvas.width || 1, uiCanvas.height || 1);
      updateCurrentBar("未选择");
      setDirty(false);
      await refreshList();
      if (cachedImages.length) {
        await selectImage(cachedImages[0].stem);
      } else {
        metaBox.textContent = "列表为空";
      }
      toast(`已删除 ${name}（图+掩码+文本，${res.count || 0} 项）`);
    } catch (e) {
      toast(String(e.message || e), true);
    } finally {
      setLoading(false);
    }
  }

  $("#btnUndo").onclick = undo;
  $("#btnClear").onclick = clearServerMask;
  if ($("#btnDelete")) $("#btnDelete").onclick = openDeleteModal;
  if ($("#btnDeleteCancel")) $("#btnDeleteCancel").onclick = closeDeleteModal;
  if ($("#btnDeleteCancelX")) $("#btnDeleteCancelX").onclick = closeDeleteModal;
  if ($("#btnDeleteConfirm")) $("#btnDeleteConfirm").onclick = () => confirmDeleteItem();
  const deleteModalEl = $("#deleteModal");
  if (deleteModalEl) {
    deleteModalEl.addEventListener("click", (ev) => {
      if (ev.target === deleteModalEl) closeDeleteModal();
    });
  }
  $("#btnAuto").onclick = runAuto;
  $("#btnSave").onclick = saveMask;
  $("#btnReload").onclick = () => refreshList().catch((e) => toast(String(e), true));
  if ($("#btnPolyUndo")) {
    $("#btnPolyUndo").onclick = () => {
      if (!state.polyPoints.length) return toast("没有顶点可撤");
      state.polyPoints.pop();
      drawPolyPreview();
    };
  }
  if ($("#btnPolyClose")) $("#btnPolyClose").onclick = commitPolygon;
  if ($("#btnAIGen")) $("#btnAIGen").onclick = () => runAICaption(false);
  // 只保存文本框内容到 refs.txt，不再二次调用 AI
  if ($("#btnAIApplySave")) $("#btnAIApplySave").onclick = () => saveTextOnly().catch(() => {});
  if ($("#btnSaveTextOnly")) $("#btnSaveTextOnly").onclick = () => saveTextOnly().catch(() => {});

  if ($("#btnOpenFolder")) $("#btnOpenFolder").onclick = () => openFolderBrowser().catch((e) => toast(String(e), true));
  if ($("#btnCloseFolder")) $("#btnCloseFolder").onclick = closeFolderBrowser;
  if ($("#btnBrowseUp")) {
    $("#btnBrowseUp").onclick = async () => {
      const data = await api(`/api/browse?path=${encodeURIComponent(browseCwd || "")}`);
      if (data.parent) await loadBrowse(data.parent);
    };
  }
  if ($("#btnBrowseGo")) {
    $("#btnBrowseGo").onclick = () => loadBrowse(browsePathInput.value.trim()).catch((e) => toast(String(e), true));
  }
  if ($("#btnBrowseUse")) $("#btnBrowseUse").onclick = () => useCurrentBrowseFolder();
  if ($("#btnUseDefaultIn")) {
    $("#btnUseDefaultIn").onclick = async () => {
      try {
        await api("/api/set_dirs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ input_dir: "/home/zyserver/LLB/auto-annotate/raw_images" }),
        });
        state.stem = null;
        await refreshList();
        toast("已切回默认 raw_images");
      } catch (e) {
        toast(String(e.message || e), true);
      }
    };
  }
  if (imgSearch) imgSearch.addEventListener("input", renderImageList);
  if (folderModal) {
    folderModal.addEventListener("click", (ev) => {
      if (ev.target === folderModal) closeFolderBrowser();
    });
  }

  $("#prompt").addEventListener("input", () => {
    if (state.stem) setDirty(true);
  });

  window.addEventListener("keydown", (ev) => {
    const tag = (ev.target && ev.target.tagName) || "";
    const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";

    if (ev.code === "Space" && !ev.repeat) {
      if (!typing) {
        ev.preventDefault();
        state.spaceDown = true;
        updateCursor();
      }
      return;
    }

    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "s") {
      ev.preventDefault();
      saveMask();
      return;
    }
    if ((ev.ctrlKey || ev.metaKey) && (ev.key === "=" || ev.key === "+")) {
      ev.preventDefault();
      zoomBy(1.15);
      return;
    }
    if ((ev.ctrlKey || ev.metaKey) && ev.key === "-") {
      ev.preventDefault();
      zoomBy(1 / 1.15);
      return;
    }
    if ((ev.ctrlKey || ev.metaKey) && ev.key === "0") {
      ev.preventDefault();
      zoomToFit();
      return;
    }
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "z") {
      if (typing) return;
      ev.preventDefault();
      if (state.tool === "poly" && state.polyPoints.length) {
        state.polyPoints.pop();
        drawPolyPreview();
      } else {
        undo();
      }
      return;
    }
    if (typing) return;

    if (ev.key === "Enter" && state.tool === "poly") {
      ev.preventDefault();
      commitPolygon();
    }
    if (ev.key === "Escape" && state.tool === "poly") {
      clearPoly();
      toast("已取消当前多边形");
    }
    if (ev.key === "Backspace" && state.tool === "poly" && state.polyPoints.length) {
      ev.preventDefault();
      state.polyPoints.pop();
      drawPolyPreview();
    }
    if (ev.key.toLowerCase() === "p") setTool("poly");
    if (ev.key.toLowerCase() === "b") setTool("brush");
    if (ev.key.toLowerCase() === "e") setTool("eraser");
  });

  window.addEventListener("keyup", (ev) => {
    if (ev.code === "Space") {
      state.spaceDown = false;
      if (!state.panning) updateCursor();
    }
  });

  if ($("#btnZoomIn")) $("#btnZoomIn").onclick = () => zoomBy(1.2);
  if ($("#btnZoomOut")) $("#btnZoomOut").onclick = () => zoomBy(1 / 1.2);
  if ($("#btnZoomFit")) $("#btnZoomFit").onclick = () => zoomToFit();

  async function boot() {
    try {
      const h = await api("/api/health");
      healthPill.textContent = `OK · ${h.device}${h.models_loaded ? " · models" : ""}`;
      healthPill.classList.add("ok");
      if (inputDirLabel && h.input_dir) {
        inputDirLabel.textContent = shortPath(h.input_dir);
        inputDirLabel.title = h.input_dir;
      }
      if (outputDirLabel && h.mask_dir) {
        outputDirLabel.textContent = shortPath(h.mask_dir);
        outputDirLabel.title = h.mask_dir;
      }
      if (textsDirLabel && (h.refs_txt || h.texts_dir)) {
        const refs = h.refs_txt || h.texts_dir;
        textsDirLabel.textContent = shortPath(refs);
        textsDirLabel.title = refs;
      }
    } catch (e) {
      healthPill.textContent = "后端未连接";
    }
    paintMaskTint();
    setTool("poly");
    setDirty(false);
    await refreshList();
  }

  boot().catch((e) => toast(String(e), true));
})();
