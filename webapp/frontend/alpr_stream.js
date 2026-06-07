document.addEventListener('DOMContentLoaded', () => {
  const streamInput = document.getElementById('stream-url');
  const cameraSelect = document.getElementById('camera-select');
  const modeSelect = document.getElementById('stream-mode');
  const readPlateToggle = document.getElementById('read-plate');
  const startBtn = document.getElementById('start-stream');
  const pauseBtn = document.getElementById('pause-stream');
  const stopBtn = document.getElementById('stop-stream');
  const vconf = document.getElementById('vconf');
  const pconf = document.getElementById('pconf');
  const vconfVal = document.getElementById('vconf-val');
  const pconfVal = document.getElementById('pconf-val');
  const streamImg = document.getElementById('alpr-stream');
  const webrtcVideo = document.getElementById('webrtc-stream');
  const transportSelect = document.getElementById('stream-transport');
  const vehicleModelSelect = document.getElementById('vehicle-model');
  const controlPanel = document.querySelector('.control-panel');
  const placeholder = document.getElementById('stream-placeholder');
  const placeholderText = placeholder ? placeholder.querySelector('#placeholder-message') : null;
  const loadingOverlay = document.getElementById('loading-overlay');
  const connectionBadge = document.getElementById('connection-badge');
  const fpsBadge = document.getElementById('fps-badge');
  const fpsText = document.getElementById('fps-text');
  const streamInfo = document.getElementById('stream-info');
  const refreshCamerasBtn = document.getElementById('refresh-cameras');

  // Upload / source toggle elements
  const sourceCameraBtn = document.getElementById('source-camera');
  const sourceUploadBtn = document.getElementById('source-upload');
  const cameraSection = document.getElementById('camera-section');
  const uploadSection = document.getElementById('upload-section');
  const videoFileInput = document.getElementById('video-file-input');
  const fileDropZone = document.getElementById('file-drop-zone');
  const fileNameDisplay = document.getElementById('file-name-display');
  const uploadProgressWrap = document.getElementById('upload-progress-wrap');
  const uploadProgressBar = document.getElementById('upload-progress-bar');
  const uploadProgressText = document.getElementById('upload-progress-text');
  
  const CUSTOM_OPTION_VALUE = '__custom';
  const TRANSPORT_MJPEG = 'mjpeg';
  const TRANSPORT_WEBRTC = 'webrtc';

  if (
    !streamInput ||
    !cameraSelect ||
    !startBtn ||
    !pauseBtn ||
    !stopBtn ||
    !streamImg ||
    !webrtcVideo ||
    !transportSelect ||
    !readPlateToggle ||
    !modeSelect ||
    !vehicleModelSelect
  ) {
    return;
  }

  let currentSrc = '';
  let paused = false;
  let currentTransport = TRANSPORT_MJPEG;
  let peerConnection = null;
  let currentStreamNonce = 0;
  let lastStartConfig = null;
  let currentVehicleModel = null;
  let fpsInterval = null;
  let lastFrameTime = 0;
  let frameCount = 0;
  let streamStartTime = null;
  let reconnectAttempts = 0;
  const MAX_RECONNECT_ATTEMPTS = 3;
  let currentSourceType = 'camera'; // 'camera' | 'upload'
  let pendingUploadXhr = null;

  function updateConnectionStatus(status, text) {
    if (!connectionBadge) return;
    connectionBadge.className = 'status-badge';
    if (status) {
      connectionBadge.classList.add(status);
    }
    const statusText = connectionBadge.querySelector('.status-text');
    if (statusText && text) {
      statusText.textContent = text;
    }
  }

  function showLoading(show = true) {
    if (loadingOverlay) {
      loadingOverlay.style.display = show ? 'flex' : 'none';
    }
  }

  function updateFPS(fps) {
    if (!fpsBadge || !fpsText) return;
    if (fps > 0) {
      fpsText.textContent = `${Math.round(fps)} FPS`;
      fpsBadge.style.display = 'flex';
    } else {
      fpsBadge.style.display = 'none';
    }
  }

  function startFPSCounter() {
    stopFPSCounter();
    frameCount = 0;
    lastFrameTime = Date.now();
    
    fpsInterval = setInterval(() => {
      const now = Date.now();
      const elapsed = (now - lastFrameTime) / 1000;
      if (elapsed > 0) {
        const fps = frameCount / elapsed;
        updateFPS(fps);
        frameCount = 0;
        lastFrameTime = now;
      }
    }, 1000);
  }

  function stopFPSCounter() {
    if (fpsInterval) {
      clearInterval(fpsInterval);
      fpsInterval = null;
    }
    updateFPS(0);
  }

  function updateStreamDuration() {
    if (!streamStartTime || !streamInfo) return;
    const duration = Math.floor((Date.now() - streamStartTime) / 1000);
    const minutes = Math.floor(duration / 60);
    const seconds = duration % 60;
    const durationEl = document.getElementById('stream-duration');
    if (durationEl) {
      durationEl.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }
  }

  function getSelectedTransport() {
    return transportSelect.value || TRANSPORT_MJPEG;
  }

  function getSelectedMode() {
    return modeSelect.value || 'alpr';
  }

  function applyModeState() {
    const preview = getSelectedMode() === 'preview';
    if (controlPanel) {
      controlPanel.classList.toggle('preview-mode', preview);
    }
    if (vconf) {
      vconf.disabled = preview;
    }
    if (pconf) {
      pconf.disabled = preview;
    }
    if (readPlateToggle) {
      readPlateToggle.disabled = preview;
    }
  }

  function hideAllStreams() {
    if (streamImg) {
      streamImg.style.display = 'none';
      streamImg.removeAttribute('src');
    }
    if (webrtcVideo) {
      webrtcVideo.pause();
      webrtcVideo.removeAttribute('src');
      webrtcVideo.srcObject = null;
      webrtcVideo.style.display = 'none';
    }
    if (streamInfo) {
      streamInfo.style.display = 'none';
    }
    stopFPSCounter();
  }

  function teardownWebRTC() {
    if (peerConnection) {
      try {
        peerConnection.ontrack = null;
        peerConnection.onconnectionstatechange = null;
        peerConnection.oniceconnectionstatechange = null;
        peerConnection.close();
      } catch (error) {
        // ignore teardown errors
      }
    }
    peerConnection = null;
    if (webrtcVideo) {
      webrtcVideo.pause();
      webrtcVideo.removeAttribute('src');
      webrtcVideo.srcObject = null;
      webrtcVideo.style.display = 'none';
    }
    stopFPSCounter();
  }

  function stopStream(message = 'No stream running') {
    teardownWebRTC();
    hideAllStreams();
    hideUploadProgress();
    currentSrc = '';
    currentTransport = getSelectedTransport();
    paused = false;
    pauseBtn.querySelector('span').textContent = 'Pause';
    lastStartConfig = null;
    showPlaceholder(message);
    updateConnectionStatus('', 'Disconnected');
    showLoading(false);
    streamStartTime = null;
    reconnectAttempts = 0;
  }

  function showPlaceholder(message) {
    if (!placeholder) {
      return;
    }
    if (placeholderText && message) {
      placeholderText.textContent = message;
    }
    placeholder.style.display = 'flex';
    hideAllStreams();
    showLoading(false);
  }

  function showMjpegStream(src) {
    currentTransport = TRANSPORT_MJPEG;
    currentSrc = src;
    streamImg.src = src;
    streamImg.classList.add('fade-in');
    streamImg.style.display = 'block';
    if (webrtcVideo) {
      webrtcVideo.style.display = 'none';
      webrtcVideo.srcObject = null;
    }
    if (placeholder) {
      placeholder.style.display = 'none';
    }
    if (streamInfo) {
      streamInfo.style.display = 'flex';
    }
    showLoading(false);
    updateConnectionStatus('connected', 'Connected');
    streamStartTime = Date.now();
    startFPSCounter();
    
    // Update stream resolution if possible
    streamImg.onload = () => {
      const resolutionEl = document.getElementById('stream-resolution');
      if (resolutionEl) {
        resolutionEl.textContent = `${streamImg.naturalWidth}x${streamImg.naturalHeight}`;
      }
    };
  }

  async function applyVehicleModel(weightId) {
    if (!weightId) {
      return;
    }
    const previousValue = currentVehicleModel;
    vehicleModelSelect.disabled = true;
    try {
      const response = await fetch('/api/vehicle_models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ weight: weightId }),
      });
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      const payload = await response.json();
      const selected = typeof payload?.selected === 'string' && payload.selected
        ? payload.selected
        : weightId;
      currentVehicleModel = selected;
      vehicleModelSelect.value = selected;
    } catch (error) {
      console.error('Failed to switch vehicle detector', error); // eslint-disable-line no-console
      if (previousValue) {
        vehicleModelSelect.value = previousValue;
      }
    } finally {
      vehicleModelSelect.disabled = false;
    }
  }

  async function loadVehicleModels() {
    vehicleModelSelect.disabled = true;
    vehicleModelSelect.innerHTML = '';

    const loadingOption = document.createElement('option');
    loadingOption.textContent = 'Loading...';
    loadingOption.disabled = true;
    loadingOption.selected = true;
    vehicleModelSelect.appendChild(loadingOption);

    try {
      const response = await fetch('/api/vehicle_models');
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
      const payload = await response.json();
      const models = Array.isArray(payload?.models) ? payload.models : [];
      const selected = typeof payload?.selected === 'string' ? payload.selected : null;

      vehicleModelSelect.innerHTML = '';
      if (!models.length) {
        const emptyOption = document.createElement('option');
        emptyOption.textContent = 'No models found';
        emptyOption.disabled = true;
        emptyOption.selected = true;
        vehicleModelSelect.appendChild(emptyOption);
        currentVehicleModel = null;
        return;
      }

      let foundSelected = false;
      let added = 0;
      models.forEach((model) => {
        if (!model) {
          return;
        }
        const option = document.createElement('option');
        const optionValue = typeof model.path === 'string' && model.path
          ? model.path
          : model.filename || '';
        if (!optionValue) {
          return;
        }
        option.value = optionValue;
        option.textContent = model.label || model.filename || optionValue;
        if (selected && optionValue === selected) {
          option.selected = true;
          foundSelected = true;
        }
        vehicleModelSelect.appendChild(option);
        added += 1;
      });

      if (added === 0) {
        const emptyOption = document.createElement('option');
        emptyOption.textContent = 'No models found';
        emptyOption.disabled = true;
        emptyOption.selected = true;
        vehicleModelSelect.appendChild(emptyOption);
        currentVehicleModel = null;
        return;
      }

      if (!foundSelected) {
        vehicleModelSelect.selectedIndex = 0;
        const fallbackValue = vehicleModelSelect.value;
        currentVehicleModel = fallbackValue || null;
        if (fallbackValue) {
          await applyVehicleModel(fallbackValue);
        }
        return;
      }

      const selectedValue = vehicleModelSelect.value;
      currentVehicleModel = selectedValue || null;
    } catch (error) {
      console.error('Failed to load vehicle models', error); // eslint-disable-line no-console
      vehicleModelSelect.innerHTML = '';
      const errorOption = document.createElement('option');
      errorOption.textContent = 'Unable to load models';
      errorOption.disabled = true;
      errorOption.selected = true;
      vehicleModelSelect.appendChild(errorOption);
      currentVehicleModel = null;
    } finally {
      vehicleModelSelect.disabled = false;
    }
  }

  function switchSourceType(type) {
    currentSourceType = type;
    if (type === 'upload') {
      if (cameraSection) cameraSection.style.display = 'none';
      if (uploadSection) uploadSection.style.display = 'block';
      if (sourceCameraBtn) sourceCameraBtn.classList.remove('active');
      if (sourceUploadBtn) sourceUploadBtn.classList.add('active');
    } else {
      if (cameraSection) cameraSection.style.display = 'block';
      if (uploadSection) uploadSection.style.display = 'none';
      if (sourceCameraBtn) sourceCameraBtn.classList.add('active');
      if (sourceUploadBtn) sourceUploadBtn.classList.remove('active');
    }
    stopStream('Configure your source and click Start.');
  }

  function setUploadProgress(ratio, label) {
    if (!uploadProgressWrap) return;
    uploadProgressWrap.style.display = ratio >= 0 ? 'block' : 'none';
    if (uploadProgressBar) {
      uploadProgressBar.style.width = `${Math.round(ratio * 100)}%`;
    }
    if (uploadProgressText) {
      uploadProgressText.textContent = label || `Uploading… ${Math.round(ratio * 100)}%`;
    }
  }

  function hideUploadProgress() {
    if (uploadProgressWrap) uploadProgressWrap.style.display = 'none';
    if (pendingUploadXhr) {
      pendingUploadXhr.abort();
      pendingUploadXhr = null;
    }
  }

  function uploadVideoFile(file, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      pendingUploadXhr = xhr;
      xhr.open('POST', '/api/upload_video');
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      });
      xhr.addEventListener('load', () => {
        pendingUploadXhr = null;
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            reject(new Error('Invalid server response'));
          }
        } else {
          let msg = `Upload failed (${xhr.status})`;
          try {
            const body = JSON.parse(xhr.responseText);
            if (body && body.detail) msg = body.detail;
          } catch { /* ignore */ }
          reject(new Error(msg));
        }
      });
      xhr.addEventListener('error', () => { pendingUploadXhr = null; reject(new Error('Network error during upload')); });
      xhr.addEventListener('abort', () => { pendingUploadXhr = null; reject(new Error('Upload cancelled')); });
      const formData = new FormData();
      formData.append('file', file);
      xhr.send(formData);
    });
  }

  function startUploadMjpegStream(videoId, config) {
    teardownWebRTC();
    const params = new URLSearchParams();
    if (typeof config.vconf !== 'undefined') params.set('vconf', String(config.vconf));
    if (typeof config.pconf !== 'undefined') params.set('pconf', String(config.pconf));
    if (typeof config.readPlate !== 'undefined') params.set('read_plate', String(config.readPlate));
    const query = params.toString();
    const src = `/api/alpr_stream/upload/${videoId}${query ? '?' + query : ''}`;
    paused = false;
    pauseBtn.querySelector('span').textContent = 'Pause';
    showMjpegStream(src);
  }

  async function startUploadStream(config) {
    const file = videoFileInput ? videoFileInput.files[0] : null;
    if (!file) {
      if (fileDropZone) fileDropZone.classList.add('drop-zone-error');
      showPlaceholder('Please select an MP4 video file first.');
      return;
    }
    if (fileDropZone) fileDropZone.classList.remove('drop-zone-error');

    const MAX_SIZE = 200 * 1024 * 1024;
    if (file.size > MAX_SIZE) {
      showPlaceholder('File exceeds the 200 MB limit. Please choose a smaller video.');
      return;
    }

    showLoading(true);
    updateConnectionStatus('connecting', 'Uploading…');
    setUploadProgress(0, 'Uploading… 0%');

    let payload;
    try {
      payload = await uploadVideoFile(file, (ratio) => {
        setUploadProgress(ratio, `Uploading… ${Math.round(ratio * 100)}%`);
      });
    } catch (err) {
      hideUploadProgress();
      showLoading(false);
      showPlaceholder(`Upload failed: ${err.message}`);
      updateConnectionStatus('error', 'Upload Failed');
      return;
    }

    setUploadProgress(1, 'Processing…');
    lastStartConfig = { ...config, sourceType: 'upload', videoId: payload.video_id };
    startUploadMjpegStream(payload.video_id, config);
    hideUploadProgress();
  }

  // Source toggle listeners
  if (sourceCameraBtn) {
    sourceCameraBtn.addEventListener('click', () => {
      if (currentSourceType !== 'camera') switchSourceType('camera');
    });
  }
  if (sourceUploadBtn) {
    sourceUploadBtn.addEventListener('click', () => {
      if (currentSourceType !== 'upload') switchSourceType('upload');
    });
  }

  // File input interactions
  if (videoFileInput) {
    videoFileInput.addEventListener('change', () => {
      const file = videoFileInput.files[0];
      if (fileDropZone) fileDropZone.classList.remove('drop-zone-error');
      if (fileNameDisplay) {
        fileNameDisplay.textContent = file ? file.name : 'No file selected';
      }
      hideUploadProgress();
    });
  }

  if (fileDropZone) {
    fileDropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      fileDropZone.classList.add('drag-over');
    });
    fileDropZone.addEventListener('dragleave', () => {
      fileDropZone.classList.remove('drag-over');
    });
    fileDropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      fileDropZone.classList.remove('drag-over');
      const file = e.dataTransfer && e.dataTransfer.files[0];
      if (file && videoFileInput) {
        const dt = new DataTransfer();
        dt.items.add(file);
        videoFileInput.files = dt.files;
        if (fileNameDisplay) fileNameDisplay.textContent = file.name;
        fileDropZone.classList.remove('drop-zone-error');
        hideUploadProgress();
      }
    });
  }

  function buildStreamConfig(url, mode) {
    const config = {
      url,
      mode,
      transport: getSelectedTransport(),
      vconf: vconf ? vconf.value : undefined,
      pconf: pconf ? pconf.value : undefined,
      readPlate: readPlateToggle ? readPlateToggle.checked : undefined,
    };
    return config;
  }

  function startMjpegStream(config) {
    teardownWebRTC();
    const params = new URLSearchParams({ url: config.url });
    const preview = config.mode === 'preview';
    if (!preview && typeof config.vconf !== 'undefined') {
      params.set('vconf', String(config.vconf));
    }
    if (!preview && typeof config.pconf !== 'undefined') {
      params.set('pconf', String(config.pconf));
    }
    if (!preview && typeof config.readPlate !== 'undefined') {
      params.set('read_plate', String(config.readPlate));
    }
    const endpoint = preview ? '/api/video' : '/api/alpr_stream';
    const src = `${endpoint}?${params.toString()}`;
    paused = false;
    pauseBtn.querySelector('span').textContent = 'Pause';
    showMjpegStream(src);
  }

  async function attemptWebRTCReconnect(config, nonce) {
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      showPlaceholder(`WebRTC connection failed after ${MAX_RECONNECT_ATTEMPTS} attempts. Click Start to retry.`);
      updateConnectionStatus('error', 'Connection Failed');
      return;
    }
    
    reconnectAttempts++;
    updateConnectionStatus('connecting', `Reconnecting (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`);
    
    await new Promise(resolve => setTimeout(resolve, 2000 * reconnectAttempts));
    
    if (nonce === currentStreamNonce) {
      await startWebRTCStream(config, nonce);
    }
  }

  async function startWebRTCStream(config, nonce) {
    teardownWebRTC();
    currentTransport = TRANSPORT_WEBRTC;
    hideAllStreams();
    showLoading(true);
    updateConnectionStatus('connecting', 'Connecting...');
    
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });
    peerConnection = pc;

    pc.addTransceiver('video', { direction: 'recvonly' });

    pc.ontrack = (event) => {
      if (nonce !== currentStreamNonce) {
        return;
      }
      const [stream] = event.streams || [];
      if (!stream || !webrtcVideo) {
        return;
      }
      webrtcVideo.srcObject = stream;
      webrtcVideo.classList.add('fade-in');
      webrtcVideo.style.display = 'block';
      if (placeholder) {
        placeholder.style.display = 'none';
      }
      if (streamInfo) {
        streamInfo.style.display = 'flex';
      }
      showLoading(false);
      updateConnectionStatus('connected', 'Connected (WebRTC)');
      reconnectAttempts = 0;
      streamStartTime = Date.now();
      
      const playPromise = webrtcVideo.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch((err) => {
          console.error('WebRTC playback error:', err);
        });
      }
      
      // Start FPS tracking
      startFPSCounter();
      
      // Track video metadata
      webrtcVideo.onloadedmetadata = () => {
        const resolutionEl = document.getElementById('stream-resolution');
        if (resolutionEl) {
          resolutionEl.textContent = `${webrtcVideo.videoWidth}x${webrtcVideo.videoHeight}`;
        }
      };
      
      // Count frames for FPS
      const trackFrames = () => {
        if (webrtcVideo.readyState >= 2) {
          frameCount++;
        }
        if (nonce === currentStreamNonce && !paused) {
          requestAnimationFrame(trackFrames);
        }
      };
      trackFrames();
    };

    pc.oniceconnectionstatechange = () => {
      console.log('ICE connection state:', pc.iceConnectionState);
      if (pc.iceConnectionState === 'connected') {
        updateConnectionStatus('connected', 'Connected (WebRTC)');
      } else if (pc.iceConnectionState === 'checking') {
        updateConnectionStatus('connecting', 'Establishing connection...');
      } else if (pc.iceConnectionState === 'disconnected') {
        updateConnectionStatus('error', 'Disconnected');
        if (nonce === currentStreamNonce && lastStartConfig) {
          attemptWebRTCReconnect(config, nonce);
        }
      } else if (pc.iceConnectionState === 'failed') {
        updateConnectionStatus('error', 'Connection Failed');
        if (nonce === currentStreamNonce && lastStartConfig) {
          attemptWebRTCReconnect(config, nonce);
        }
      }
    };

    pc.onconnectionstatechange = () => {
      console.log('Connection state:', pc.connectionState);
      if (pc.connectionState === 'failed') {
        showPlaceholder('WebRTC connection failed. Click Start to retry.');
        updateConnectionStatus('error', 'Connection Failed');
        teardownWebRTC();
      } else if (pc.connectionState === 'disconnected') {
        if (nonce === currentStreamNonce && !paused) {
          updateConnectionStatus('error', 'Connection lost');
        }
      } else if (pc.connectionState === 'connected') {
        updateConnectionStatus('connected', 'Connected (WebRTC)');
      }
    };

    const payload = {
      sdp: undefined,
      type: undefined,
      url: config.url,
      mode: config.mode,
    };

    const preview = config.mode === 'preview';
    if (!preview && typeof config.vconf !== 'undefined') {
      payload.vconf = config.vconf;
    }
    if (!preview && typeof config.pconf !== 'undefined') {
      payload.pconf = config.pconf;
    }
    if (!preview && typeof config.readPlate !== 'undefined') {
      payload.read_plate = config.readPlate;
    }

    try {
      const offer = await pc.createOffer();
      if (nonce !== currentStreamNonce) {
        return;
      }
      await pc.setLocalDescription(offer);
      payload.sdp = offer.sdp;
      payload.type = offer.type;

      const response = await fetch('/api/webrtc/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server responded with ${response.status}: ${errorText}`);
      }
      if (nonce !== currentStreamNonce) {
        return;
      }
      const answer = await response.json();
      await pc.setRemoteDescription(answer);
      showLoading(false);
      paused = false;
      pauseBtn.querySelector('span').textContent = 'Pause';
    } catch (error) {
      console.error('Failed to start WebRTC stream', error);
      showPlaceholder(`Unable to start WebRTC: ${error.message}`);
      updateConnectionStatus('error', 'Connection Failed');
      teardownWebRTC();
      showLoading(false);
    }
  }

  async function startStream(config) {
    lastStartConfig = config;
    paused = false;
    pauseBtn.querySelector('span').textContent = 'Pause';
    currentStreamNonce += 1;
    const nonce = currentStreamNonce;
    reconnectAttempts = 0;
    
    if (config.transport === TRANSPORT_WEBRTC) {
      await startWebRTCStream(config, nonce);
    } else {
      startMjpegStream(config);
    }
  }

  // Update stream duration periodically
  setInterval(() => {
    if (streamStartTime) {
      updateStreamDuration();
    }
  }, 1000);

  // Track MJPEG frames for FPS
  streamImg.addEventListener('load', () => {
    if (currentTransport === TRANSPORT_MJPEG && streamImg.style.display === 'block') {
      frameCount++;
    }
  });

  if (vconf) {
    vconf.addEventListener('input', () => {
      if (vconfVal) {
        vconfVal.textContent = Number(vconf.value).toFixed(2);
      }
    });
  }

  if (pconf) {
    pconf.addEventListener('input', () => {
      if (pconfVal) {
        pconfVal.textContent = Number(pconf.value).toFixed(2);
      }
    });
  }

  if (streamInput) {
    streamInput.addEventListener('input', () => {
      streamInput.classList.remove('input-error');
      if (cameraSelect.value !== CUSTOM_OPTION_VALUE) {
        cameraSelect.value = CUSTOM_OPTION_VALUE;
      }
    });
  }

  cameraSelect.addEventListener('change', () => {
    if (cameraSelect.value === CUSTOM_OPTION_VALUE) {
      return;
    }
    streamInput.value = cameraSelect.value;
    streamInput.classList.remove('input-error');
  });

  vehicleModelSelect.addEventListener('change', async () => {
    const value = vehicleModelSelect.value;
    if (!value || value === currentVehicleModel) {
      return;
    }
    await applyVehicleModel(value);
  });

  if (refreshCamerasBtn) {
    refreshCamerasBtn.addEventListener('click', () => {
      loadCameraPresets();
    });
  }

  startBtn.addEventListener('click', async () => {
    if (currentSourceType === 'upload') {
      startBtn.disabled = true;
      try {
        const mode = getSelectedMode();
        const config = {
          mode,
          vconf: vconf ? vconf.value : undefined,
          pconf: pconf ? pconf.value : undefined,
          readPlate: readPlateToggle ? readPlateToggle.checked : undefined,
        };
        await startUploadStream(config);
      } finally {
        startBtn.disabled = false;
      }
      return;
    }

    const url = streamInput.value.trim();
    if (!url) {
      streamInput.classList.add('input-error');
      streamInput.focus();
      showPlaceholder('Please provide a stream URL');
      return;
    }

    startBtn.disabled = true;
    try {
      const mode = getSelectedMode();
      const config = buildStreamConfig(url, mode);
      await startStream(config);
    } finally {
      startBtn.disabled = false;
    }
  });

  pauseBtn.addEventListener('click', async () => {
    if (!lastStartConfig && !currentSrc) {
      return;
    }

    if (currentTransport === TRANSPORT_MJPEG) {
      if (!currentSrc) {
        return;
      }
      if (!paused) {
        streamImg.removeAttribute('src');
        showPlaceholder('Stream paused');
        pauseBtn.querySelector('span').textContent = 'Resume';
        updateConnectionStatus('', 'Paused');
        paused = true;
        stopFPSCounter();
      } else {
        showMjpegStream(currentSrc);
        pauseBtn.querySelector('span').textContent = 'Pause';
        updateConnectionStatus('connected', 'Connected');
        paused = false;
      }
      return;
    }

    if (!paused) {
      teardownWebRTC();
      showPlaceholder('Stream paused');
      pauseBtn.querySelector('span').textContent = 'Resume';
      updateConnectionStatus('', 'Paused');
      paused = true;
    } else {
      if (!lastStartConfig) {
        return;
      }
      paused = false;
      pauseBtn.querySelector('span').textContent = 'Pause';
      await startStream(lastStartConfig);
    }
  });

  stopBtn.addEventListener('click', () => {
    const preview = getSelectedMode() === 'preview';
    const message = preview
      ? 'Camera preview stopped. Click Start to view the camera feed.'
      : 'Stream stopped. Configure your camera and click Start.';
    stopStream(message);
  });

  transportSelect.addEventListener('change', () => {
    const value = getSelectedTransport();
    const message = value === TRANSPORT_WEBRTC
      ? 'WebRTC selected. Low latency streaming with automatic reconnection.'
      : 'HTTP MJPEG selected. Click Start to view the camera feed.';
    stopStream(message);
  });

  streamImg.addEventListener('error', () => {
    if (!currentSrc) {
      return;
    }
    stopStream('Stream connection lost. Please check the URL and try again.');
    updateConnectionStatus('error', 'Stream Error');
  });

  if (modeSelect) {
    modeSelect.addEventListener('change', () => {
      applyModeState();
      const preview = getSelectedMode() === 'preview';
      const message = preview
        ? 'Preview mode: View camera feed without license plate detection.'
        : 'ALPR mode: Real-time license plate recognition enabled.';
      stopStream(message);
    });
  }

  applyModeState();
  showPlaceholder('Ready to start. Configure your camera and click Start.');
  updateConnectionStatus('', 'Disconnected');

  async function loadCameraPresets() {
    cameraSelect.disabled = true;
    try {
      const response = await fetch('/api/cameras');
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
      const payload = await response.json();
      const presets = Array.isArray(payload?.presets) ? payload.presets : [];
      if (!presets.length) {
        return;
      }

      const fragment = document.createDocumentFragment();
      presets.forEach((preset) => {
        if (!preset || typeof preset.url !== 'string' || typeof preset.label !== 'string') {
          return;
        }
        const option = document.createElement('option');
        option.value = preset.url;
        option.textContent = preset.label;
        fragment.appendChild(option);
      });

      cameraSelect.appendChild(fragment);
    } catch (error) {
      console.error('Failed to load camera presets', error); // eslint-disable-line no-console
    } finally {
      cameraSelect.disabled = false;
    }
  }

  loadCameraPresets();
  loadVehicleModels();
});
