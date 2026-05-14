#!/usr/bin/env python3
"""
Cat Pet DNA — Web UI (Flask)
Drag & drop image → run pipeline → show results
"""

import os
import sys
import json
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cat_pet_dna_pipeline import process_image

import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "output", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cat Pet DNA</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e; color: #eee; min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
  }
  h1 {
    margin: 30px 0 10px; font-size: 28px; font-weight: 600;
    background: linear-gradient(135deg, #f093fb, #f5576c);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .subtitle { color: #888; font-size: 14px; margin-bottom: 20px; }

  .drop-zone {
    width: 500px; max-width: 90vw; height: 260px;
    border: 3px dashed #555; border-radius: 20px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    cursor: pointer; transition: all 0.3s; background: #16213e; position: relative;
  }
  .drop-zone.dragover { border-color: #f5576c; background: #1a1a3e; transform: scale(1.02); }
  .drop-zone.has-image { border-color: #4ecdc4; height: auto; padding: 10px; }
  .drop-zone .icon { font-size: 50px; margin-bottom: 10px; opacity: 0.6; }
  .drop-zone .text { color: #888; font-size: 16px; }
  .drop-zone .hint { color: #666; font-size: 13px; margin-top: 8px; }
  .drop-zone img.preview { max-width: 100%; max-height: 300px; border-radius: 12px; }

  .status-bar {
    width: 500px; max-width: 90vw; margin-top: 15px;
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; border-radius: 10px; background: #16213e;
    font-size: 14px; min-height: 44px;
  }
  .spinner {
    width: 20px; height: 20px; border: 3px solid #555;
    border-top-color: #f5576c; border-radius: 50%;
    animation: spin 0.8s linear infinite; display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .results {
    width: 500px; max-width: 90vw; margin-top: 15px;
    display: none; flex-direction: column; gap: 12px;
  }
  .results.show { display: flex; }

  .result-img {
    width: 100%; border-radius: 12px; border: 2px solid #333;
    cursor: pointer; transition: transform 0.2s;
  }
  .result-img:hover { transform: scale(1.01); }

  .dna-panel {
    background: #16213e; border-radius: 12px; padding: 16px 20px;
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px 20px;
  }
  .dna-item {
    display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid #222; font-size: 14px;
  }
  .dna-item .label { color: #888; }
  .dna-item .value { color: #4ecdc4; font-weight: 500; }
  .dna-item .value.highlight { color: #f5576c; }
  .dna-item.full { grid-column: 1 / -1; }

  .color-bar {
    display: flex; height: 24px; border-radius: 6px; overflow: hidden;
    margin: 4px 0;
  }
  .color-seg {
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; color: #fff; text-shadow: 0 0 3px rgba(0,0,0,0.7);
  }

  .json-box {
    background: #0d1117; border-radius: 10px; padding: 14px;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px;
    line-height: 1.6; max-height: 300px; overflow-y: auto;
    white-space: pre-wrap; color: #8b949e;
  }

  .actions { display: flex; gap: 10px; margin-top: 5px; }
  .btn {
    padding: 8px 20px; border: none; border-radius: 8px;
    font-size: 14px; cursor: pointer; transition: all 0.2s;
  }
  .btn-primary { background: #f5576c; color: #fff; }
  .btn-primary:hover { background: #d9445a; }
  .btn-secondary { background: #333; color: #ccc; }
  .btn-secondary:hover { background: #444; }

  footer { margin: 30px 0 20px; color: #555; font-size: 12px; }
</style>
</head>
<body>
<h1>🐱 Cat Pet DNA</h1>
<p class="subtitle">拖拽猫图 → 自动分析 → 生成结构化 Pet DNA</p>

<div class="drop-zone" id="dropZone">
  <div class="icon" id="dropIcon">🐈</div>
  <div class="text" id="dropText">拖拽图片到这里</div>
  <div class="hint" id="dropHint">或点击选择文件 · JPG / PNG</div>
  <img class="preview" id="preview" style="display:none" />
  <input type="file" id="fileInput" accept="image/*" style="display:none" />
</div>

<div class="status-bar" id="statusBar">
  <div class="spinner" id="spinner"></div>
  <span id="statusText">等待图片...</span>
</div>

<div class="results" id="results">
  <img class="result-img" id="resultImg" />
  <div class="dna-panel" id="dnaPanel"></div>
  <div class="actions">
    <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">🔄 换一张图</button>
    <button class="btn btn-secondary" onclick="toggleJSON()">📄 查看 JSON</button>
  </div>
  <div class="json-box" id="jsonBox" style="display:none"></div>
</div>

<footer>Cat Pet DNA v1.0 · 纯 CPU 低算力管线 · 可商用</footer>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const dropIcon = document.getElementById('dropIcon');
const dropText = document.getElementById('dropText');
const dropHint = document.getElementById('dropHint');
const statusText = document.getElementById('statusText');
const spinner = document.getElementById('spinner');
const results = document.getElementById('results');
const resultImg = document.getElementById('resultImg');
const dnaPanel = document.getElementById('dnaPanel');
const jsonBox = document.getElementById('jsonBox');

// Drag events
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('dragover'); handleFile(e.dataTransfer.files[0]); });
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) { setStatus('请选择图片文件', 'error'); return; }
  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result; preview.style.display = 'block';
    dropIcon.style.display = 'none'; dropText.textContent = file.name;
    dropHint.textContent = `${(file.size/1024).toFixed(0)} KB`;
    dropZone.classList.add('has-image');
  };
  reader.readAsDataURL(file);
  uploadFile(file);
}

function uploadFile(file) {
  setStatus('正在分析...', 'loading');
  results.classList.remove('show');

  const formData = new FormData();
  formData.append('image', file);

  fetch('/analyze', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.error) { setStatus('❌ ' + data.error, 'error'); return; }
      setStatus(`✅ 完成 · ${data.time_ms}ms · 置信度 ${(data.dna.confidence.overall*100).toFixed(0)}%`, 'done');
      showResults(data);
    })
    .catch(err => { setStatus('❌ 请求失败: ' + err.message, 'error'); });
}

function setStatus(msg, type) {
  statusText.textContent = msg;
  spinner.style.display = type === 'loading' ? 'block' : 'none';
  statusBar.style.borderLeft = type === 'error' ? '4px solid #f5576c' : type === 'done' ? '4px solid #4ecdc4' : '4px solid transparent';
}

function showResults(data) {
  const dna = data.dna;
  resultImg.src = data.visualized_url + '?t=' + Date.now();
  results.classList.add('show');

  // DNA panel
  const a = dna.appearance;
  const palette = a.color_palette || [];
  const colorBar = palette.length ? '<div class="color-bar">' +
    palette.map(c => `<div class="color-seg" style="flex:${c.ratio};background:#${c.hex.replace('#','')}">${(c.ratio*100).toFixed(0)}%</div>`).join('') +
    '</div>' : '';

  dnaPanel.innerHTML = `
    <div class="dna-item"><span class="label">毛色</span><span class="value">${a.primary_colors.join(', ') || '?'}</span></div>
    <div class="dna-item"><span class="label">斑纹</span><span class="value">${a.color_pattern.type}</span></div>
    <div class="dna-item full">${colorBar}</div>
    <div class="dna-item"><span class="label">姿态</span><span class="value">${dna.pose.pose_type}</span></div>
    <div class="dna-item"><span class="label">动作倾向</span><span class="value">${dna.action_tendency.state}</span></div>
    <div class="dna-item"><span class="label">体型</span><span class="value">${a.body_size_estimate}</span></div>
    <div class="dna-item"><span class="label">头型</span><span class="value">${dna.face.head_shape.shape}</span></div>
    <div class="dna-item"><span class="label">耳型</span><span class="value">${dna.face.ears.ear_type}</span></div>
    <div class="dna-item"><span class="label">品种</span><span class="value highlight">${dna.breed.candidates[0]?.name || '?'}</span></div>
    <div class="dna-item"><span class="label">置信度</span><span class="value highlight">${(dna.confidence.overall*100).toFixed(1)}%</span></div>
  `;

  jsonBox.textContent = JSON.stringify(dna, null, 2);
}

function toggleJSON() {
  jsonBox.style.display = jsonBox.style.display === 'none' ? 'block' : 'none';
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    ext = Path(file.filename).suffix or ".jpg"
    uid = uuid.uuid4().hex[:8]
    save_name = f"{uid}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    file.save(save_path)

    # Read and process
    img_bgr = cv2.imread(save_path)
    if img_bgr is None:
        return jsonify({"error": "Cannot read image"}), 400
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        result = process_image(img_rgb)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    dna = result["dna"]

    # Save visualized image
    vis_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
    vis_name = f"{uid}_vis.jpg"
    vis_path = os.path.join(UPLOAD_DIR, vis_name)
    cv2.imwrite(vis_path, vis_bgr)

    return jsonify({
        "dna": dna,
        "time_ms": round(result["time_ms"], 1),
        "visualized_url": f"/output/uploads/{vis_name}",
    })


@app.route("/output/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    print(f"\n  🐱 Cat Pet DNA Web UI")
    print(f"  ─────────────────────")
    print(f"  Open: http://localhost:5000")
    print(f"  Quit: Ctrl+C\n")
    app.run(host="0.0.0.0", port=5000, debug=False)