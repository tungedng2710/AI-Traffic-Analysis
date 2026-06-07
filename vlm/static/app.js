const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const previewImage = document.getElementById("preview-image");
const runButton = document.getElementById("run-button");
const sampleSelect = document.getElementById("sample-select");
const resultOutput = document.getElementById("result-output");
const elapsedTime = document.getElementById("elapsed-time");
const statusPill = document.getElementById("status-pill");
const statusText = document.getElementById("status-text");

let selectedFile = null;
let previewUrl = null;

init();

async function init() {
  renderResultRows({ status: "No result" });
  await Promise.all([loadHealth(), loadSamples()]);
  bindEvents();
}

function bindEvents() {
  fileInput.addEventListener("change", () => {
    const [file] = fileInput.files;
    if (file) {
      setSelectedFile(file);
      sampleSelect.value = "";
    }
  });

  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragging");
  });

  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
    const [file] = event.dataTransfer.files;
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      sampleSelect.value = "";
    }
  });

  sampleSelect.addEventListener("change", async () => {
    if (!sampleSelect.value) {
      return;
    }

    const response = await fetch(sampleSelect.value);
    const blob = await response.blob();
    const name = sampleSelect.options[sampleSelect.selectedIndex].textContent;
    setSelectedFile(new File([blob], name, { type: blob.type || "image/jpeg" }));
  });

  runButton.addEventListener("click", runInference);
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    setStatus(data.ready ? "Ready" : "Unavailable", data.ready ? "ready" : "error");
  } catch (error) {
    setStatus("API offline", "error");
  }
}

async function loadSamples() {
  try {
    const response = await fetch("/api/samples");
    const samples = await response.json();
    for (const sample of samples) {
      const option = document.createElement("option");
      option.value = sample.url;
      option.textContent = sample.name;
      sampleSelect.appendChild(option);
    }
  } catch (error) {
    sampleSelect.disabled = true;
  }
}

function setSelectedFile(file) {
  selectedFile = file;
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;

  dropZone.classList.add("has-image");
  runButton.disabled = false;
  renderResultRows({ status: "Ready" });
  elapsedTime.textContent = "";
}

async function runInference() {
  if (!selectedFile) {
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  setBusy(true);
  renderResultRows({ status: "Processing" });
  elapsedTime.textContent = "";

  try {
    const response = await fetch("/api/read-plate", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Inference failed");
    }

    renderResultRows(data.prediction || pickPredictionFields(data));
    elapsedTime.textContent = `${data.elapsed_ms} ms`;
    setStatus("Ready", "ready");
  } catch (error) {
    renderResultRows({ error: error.message });
    setStatus("Error", "error");
  } finally {
    setBusy(false);
  }
}

function setStatus(text, state) {
  statusText.textContent = text;
  statusPill.classList.toggle("ready", state === "ready");
  statusPill.classList.toggle("error", state === "error");
}

function renderResultRows(values) {
  resultOutput.replaceChildren();

  for (const [key, value] of Object.entries(values)) {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const output = document.createElement("strong");

    row.className = "result-row";
    label.textContent = formatLabel(key);
    output.textContent = formatValue(value);
    if (key.toLowerCase().includes("plate")) {
      output.classList.add("mono-value");
    }

    row.append(label, output);
    resultOutput.append(row);
  }
}

function pickPredictionFields(data) {
  return {
    vehicle_type: data.vehicle_type || "unknown",
    vehicle_color: data.vehicle_color || "unknown",
    license_plate: data.license_plate || "unknown",
  };
}

function formatLabel(key) {
  return key.replace(/_/g, " ");
}

function formatValue(value) {
  if (value === null || typeof value === "undefined" || value === "") {
    return "unknown";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function setBusy(isBusy) {
  runButton.disabled = isBusy || !selectedFile;
  runButton.querySelector("span").textContent = isBusy
    ? "Processing"
    : "Read Plate";
}
