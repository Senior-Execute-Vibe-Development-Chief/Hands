import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as THREE from 'three';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import './styles.css';

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

const FINGER_TIPS = {
  thumb: 4,
  index: 8,
  middle: 12,
  ring: 16,
  pinky: 20
};

const FINGER_PIPS = {
  index: 6,
  middle: 10,
  ring: 14,
  pinky: 18
};

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z || 0) - (b.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function makeDot(index) {
  const radius = index === 0 ? 0.025 : 0.016;
  const geometry = new THREE.SphereGeometry(radius, 18, 18);
  const material = new THREE.MeshStandardMaterial({ roughness: 0.36, metalness: 0.04 });
  return new THREE.Mesh(geometry, material);
}

function makeSegment() {
  const geometry = new THREE.CylinderGeometry(0.007, 0.007, 1, 14);
  const material = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.02 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.visible = false;
  return mesh;
}

function updateSegment(mesh, start, end) {
  const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
  const direction = new THREE.Vector3().subVectors(end, start);
  const length = direction.length();

  if (length < 0.0001) {
    mesh.visible = false;
    return;
  }

  mesh.visible = true;
  mesh.position.copy(midpoint);
  mesh.scale.set(1, length, 1);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
}

/** Image landmarks: x,y in frame — drives translation in the 3D view. */
function scenePointFromImageLm(point) {
  return new THREE.Vector3(
    (point.x - 0.5) * 2.65,
    -(point.y - 0.5) * 1.75,
    -point.z * 2.45
  );
}

/**
 * MediaPipe world landmarks are wrist-rooted (wrist ≈ origin): they encode pose,
 * not where the hand is in the room. Combine image wrist + world bone offsets
 * so the hand translates and articulates.
 */
function landmarksToScenePoints(imageLm, worldLm, useWorldModel) {
  if (!imageLm || imageLm.length < 21) return null;
  if (useWorldModel && worldLm && worldLm.length >= 21) {
    const wristScene = scenePointFromImageLm(imageLm[0]);
    const w0 = worldLm[0];
    return worldLm.map((w) => (
      wristScene.clone().add(new THREE.Vector3(
        (w.x - w0.x) * 5.35,
        -(w.y - w0.y) * 5.35,
        -(w.z - w0.z) * 5.35
      ))
    ));
  }
  return imageLm.map((p) => scenePointFromImageLm(p));
}

function createHandGroup(scene) {
  const root = new THREE.Group();
  const dots = Array.from({ length: 21 }, (_, index) => makeDot(index));
  const segments = HAND_CONNECTIONS.map(() => makeSegment());

  dots.forEach(dot => root.add(dot));
  segments.forEach(segment => root.add(segment));
  root.visible = false;
  scene.add(root);

  return { group: root, dots, segments };
}

const GRAB_RADIUS = 0.32;

function createPickables(scene) {
  const configs = [
    { geo: new THREE.BoxGeometry(0.13, 0.13, 0.13), pos: [-0.22, 0.08, -0.32], color: 0x6ae3ff },
    { geo: new THREE.SphereGeometry(0.1, 20, 20), pos: [0.2, 0.02, -0.34], color: 0xffb86a },
    { geo: new THREE.CylinderGeometry(0.07, 0.09, 0.16, 16), pos: [0.02, 0.14, -0.38], color: 0xc99dff },
    { geo: new THREE.TetrahedronGeometry(0.12, 0), pos: [-0.12, -0.12, -0.3], color: 0x7cffc4 },
    { geo: new THREE.OctahedronGeometry(0.11, 0), pos: [0.22, 0.12, -0.4], color: 0xff7cc8 }
  ];

  const meshes = [];
  for (const c of configs) {
    const mat = new THREE.MeshStandardMaterial({
      color: c.color,
      roughness: 0.38,
      metalness: 0.12
    });
    const mesh = new THREE.Mesh(c.geo, mat);
    mesh.position.set(c.pos[0], c.pos[1], c.pos[2]);
    scene.add(mesh);
    meshes.push(mesh);
  }

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(5, 4),
    new THREE.MeshStandardMaterial({
      color: 0x121a2e,
      roughness: 0.94,
      metalness: 0.02,
      transparent: true,
      opacity: 0.88
    })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.78;
  floor.name = 'play-floor';
  scene.add(floor);

  return meshes;
}

function meshHeldByOtherHand(mesh, myHandIndex, grabState) {
  for (let i = 0; i < grabState.length; i += 1) {
    if (i === myHandIndex) continue;
    const g = grabState[i];
    if (g && g.mesh === mesh) return true;
  }
  return false;
}

function applyGrabInteraction(grabFrames, pickables, grabState, scratch) {
  if (!pickables?.length) return;

  for (const { handIndex, points, pinch } of grabFrames) {
    const thumbTip = points[FINGER_TIPS.thumb];
    const indexTip = points[FINGER_TIPS.index];
    const pinchMid = scratch.pinchMid.copy(thumbTip).add(indexTip).multiplyScalar(0.5);

    const held = grabState[handIndex];

    if (held) {
      if (!pinch) {
        grabState[handIndex] = null;
      } else {
        held.mesh.position.copy(pinchMid).add(held.offset);
      }
      continue;
    }

    if (!pinch) continue;

    let best = null;
    let bestD = GRAB_RADIUS;
    for (const mesh of pickables) {
      if (meshHeldByOtherHand(mesh, handIndex, grabState)) continue;
      const d = pinchMid.distanceTo(mesh.position);
      if (d < bestD) {
        bestD = d;
        best = mesh;
      }
    }

    if (best) {
      const offset = best.position.clone().sub(pinchMid);
      grabState[handIndex] = { mesh: best, offset };
    }
  }
}

function classifyGestures(landmarks) {
  if (!landmarks || landmarks.length < 21) {
    return { name: 'No hand', pinch: false, openPalm: false, fist: false, point: false, metrics: {} };
  }

  const wrist = landmarks[0];
  const indexTip = landmarks[FINGER_TIPS.index];
  const thumbTip = landmarks[FINGER_TIPS.thumb];
  const middleTip = landmarks[FINGER_TIPS.middle];
  const ringTip = landmarks[FINGER_TIPS.ring];
  const pinkyTip = landmarks[FINGER_TIPS.pinky];

  const palmSize = Math.max(distance(wrist, landmarks[9]), 0.0001);
  const pinchDistance = distance(thumbTip, indexTip) / palmSize;

  const extended = {
    index: distance(wrist, indexTip) > distance(wrist, landmarks[FINGER_PIPS.index]) * 1.18,
    middle: distance(wrist, middleTip) > distance(wrist, landmarks[FINGER_PIPS.middle]) * 1.18,
    ring: distance(wrist, ringTip) > distance(wrist, landmarks[FINGER_PIPS.ring]) * 1.14,
    pinky: distance(wrist, pinkyTip) > distance(wrist, landmarks[FINGER_PIPS.pinky]) * 1.12
  };

  const extendedCount = Object.values(extended).filter(Boolean).length;
  const pinch = pinchDistance < 0.48;
  const openPalm = extendedCount >= 4 && !pinch;
  const fist = extendedCount <= 1 && !pinch;
  const point = extended.index && !extended.middle && !extended.ring && !extended.pinky;

  let name = 'Neutral';
  if (pinch) name = 'Pinch';
  else if (openPalm) name = 'Open palm';
  else if (fist) name = 'Fist / grab';
  else if (point) name = 'Point';

  return {
    name,
    pinch,
    openPalm,
    fist,
    point,
    metrics: {
      palmSize: palmSize.toFixed(3),
      pinchDistance: pinchDistance.toFixed(2),
      extendedCount
    }
  };
}

function App() {
  const videoRef = useRef(null);
  const mountRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const handGroupsRef = useRef([]);
  const landmarkerRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const fpsRef = useRef({ frames: 0, last: performance.now() });
  const historyRef = useRef({});
  const pickablesRef = useRef([]);
  const grabStateRef = useRef([null, null]);
  const scratchRef = useRef({ pinchMid: new THREE.Vector3() });

  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  const [hands, setHands] = useState([]);
  const [fps, setFps] = useState(0);
  const [mirrorCamera, setMirrorCamera] = useState(true);
  const [useWorldCoords, setUseWorldCoords] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [threshold, setThreshold] = useState(0.55);

  const isRunning = status === 'running';
  const isBusy = status === 'loading model' || status === 'requesting camera';

  const statusText = useMemo(() => {
    if (status === 'idle') return 'Idle';
    if (status === 'loading model') return 'Loading hand model';
    if (status === 'requesting camera') return 'Waiting for camera permission';
    if (status === 'running') return 'Tracking live';
    if (status === 'error') return 'Error';
    return status;
  }, [status]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1020);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(52, 1, 0.01, 100);
    camera.position.set(0, 0.08, 2.05);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    rendererRef.current = renderer;
    mount.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 0.75);
    scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
    keyLight.position.set(1.5, 2.2, 3.0);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
    fillLight.position.set(-2, -1, 2);
    scene.add(fillLight);

    const grid = new THREE.GridHelper(2.2, 10, 0x42506d, 0x232c3e);
    grid.name = 'tracking-grid';
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.55;
    scene.add(grid);

    handGroupsRef.current = [createHandGroup(scene), createHandGroup(scene)];
    pickablesRef.current = createPickables(scene);

    function resize() {
      const width = mount.clientWidth || 900;
      const height = mount.clientHeight || 560;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }

    function render() {
      const gridObject = scene.getObjectByName('tracking-grid');
      if (gridObject) gridObject.visible = showGrid;
      renderer.render(scene, camera);
      rafRef.current = requestAnimationFrame(render);
    }

    resize();
    window.addEventListener('resize', resize);
    rafRef.current = requestAnimationFrame(render);

    return () => {
      window.removeEventListener('resize', resize);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      stopCamera();
      scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose?.();
        if (object.material) object.material.dispose?.();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
    // showGrid is intentionally read dynamically in render loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadLandmarker() {
    if (landmarkerRef.current) return landmarkerRef.current;

    setStatus('loading model');
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: threshold,
      minHandPresenceConfidence: threshold,
      minTrackingConfidence: threshold
    });

    landmarkerRef.current = landmarker;
    return landmarker;
  }

  async function startCamera() {
    setError('');

    try {
      const landmarker = await loadLandmarker();
      setStatus('requesting camera');

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });

      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      await video.play();

      setStatus('running');
      lastVideoTimeRef.current = -1;
      fpsRef.current = { frames: 0, last: performance.now() };
      trackingLoop(landmarker);
    } catch (err) {
      setStatus('error');
      setError(err?.message || 'Unable to start camera or load tracking model.');
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    setHands([]);
    setFps(0);
    grabStateRef.current = [null, null];
    handGroupsRef.current.forEach(({ group }) => {
      group.visible = false;
    });
    if (status !== 'error') setStatus('idle');
  }

  function trackingLoop(landmarker) {
    const tick = () => {
      const video = videoRef.current;
      if (!video || !streamRef.current) return;

      if (video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
        const now = performance.now();
        const results = landmarker.detectForVideo(video, now);
        lastVideoTimeRef.current = video.currentTime;
        updateHands(results);
        updateFps(now);
      }

      requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
  }

  function updateFps(now) {
    const state = fpsRef.current;
    state.frames += 1;
    if (now - state.last >= 1000) {
      setFps(state.frames);
      state.frames = 0;
      state.last = now;
    }
  }

  function smoothPoints(handKey, points) {
    const alpha = 0.42;
    const previous = historyRef.current[handKey];
    if (!previous) {
      historyRef.current[handKey] = points;
      return points;
    }

    const smoothed = points.map((point, index) => {
      const old = previous[index] || point;
      return new THREE.Vector3(
        old.x * (1 - alpha) + point.x * alpha,
        old.y * (1 - alpha) + point.y * alpha,
        old.z * (1 - alpha) + point.z * alpha
      );
    });

    historyRef.current[handKey] = smoothed;
    return smoothed;
  }

  function updateHands(results) {
    const groups = handGroupsRef.current;
    const imageHands = results.landmarks || [];
    const worldHands = results.worldLandmarks || [];
    const handedness = results.handednesses || [];
    const sourceHands = useWorldCoords && worldHands.length ? worldHands : imageHands;

    groups.forEach((hg) => {
      hg.group.visible = false;
    });

    const handSummaries = [];
    const grabFrames = [];

    sourceHands.slice(0, 2).forEach((landmarks, handIndex) => {
      const hg = groups[handIndex];
      const label = handedness?.[handIndex]?.[0]?.categoryName || `Hand ${handIndex + 1}`;
      const score = handedness?.[handIndex]?.[0]?.score || 0;
      const imageLm = imageHands[handIndex];
      if (!imageLm?.length) return;

      const gestureSource = imageLm;
      const gesture = classifyGestures(gestureSource);

      const useWorldModel = Boolean(useWorldCoords && worldHands.length && worldHands[handIndex]);
      let points = landmarksToScenePoints(imageLm, worldHands[handIndex], useWorldModel);
      if (!points) return;

      points = smoothPoints(`${label}-${handIndex}`, points);

      hg.group.visible = true;
      points.forEach((point, index) => {
        hg.dots[index].visible = true;
        hg.dots[index].position.copy(point);
      });

      HAND_CONNECTIONS.forEach(([a, b], index) => {
        updateSegment(hg.segments[index], points[a], points[b]);
      });

      handSummaries.push({
        label,
        score,
        gesture: gesture.name,
        metrics: gesture.metrics
      });

      grabFrames.push({ handIndex, points, pinch: gesture.pinch });
    });

    groups.forEach((hg, slotIndex) => {
      if (!hg.group.visible) {
        grabStateRef.current[slotIndex] = null;
      }
    });

    applyGrabInteraction(
      grabFrames,
      pickablesRef.current,
      grabStateRef.current,
      scratchRef.current
    );

    setHands(handSummaries);
  }

  async function resetModelWithThreshold(value) {
    setThreshold(value);
    if (!landmarkerRef.current) return;
    await landmarkerRef.current.setOptions({
      minHandDetectionConfidence: value,
      minHandPresenceConfidence: value,
      minTrackingConfidence: value
    });
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <div className="eyebrow">Usable prototype</div>
          <h1>Webcam hand tracking → 3D hands & objects</h1>
          <p>
            Tracks up to two hands, shows a live 3D skeleton, and lets you pinch-grab colourful shapes in front of you. Open the pinch to release.
          </p>
        </div>
        <div className="controls-row">
          {!isRunning ? (
            <button className="primary" onClick={startCamera} disabled={isBusy}>
              {isBusy ? 'Starting...' : 'Start tracking'}
            </button>
          ) : (
            <button className="secondary" onClick={stopCamera}>Stop tracking</button>
          )}
        </div>
      </section>

      <section className="workspace">
        <div className="stage-card">
          <div className="stage-header">
            <div>
              <strong>3D play space</strong>
              <span>Pinch thumb and index near a shape to pick it up; open pinch to drop.</span>
            </div>
            <div className="pill-row">
              <span className="pill">{statusText}</span>
              <span className="pill">{fps} FPS</span>
              <span className="pill">{hands.length} hands</span>
            </div>
          </div>
          <div className="stage" ref={mountRef} />
        </div>

        <aside className="side-panel">
          <div className="panel-card">
            <div className="panel-title">Camera</div>
            <video ref={videoRef} className={mirrorCamera ? 'mirror' : ''} autoPlay playsInline muted />
            <div className="hint">Camera access requires localhost or HTTPS. Running through Vite localhost works.</div>
          </div>

          <div className="panel-card">
            <div className="panel-title">Detected gestures</div>
            {hands.length === 0 ? (
              <div className="empty">No hands detected yet.</div>
            ) : (
              hands.map((hand, index) => (
                <div className="hand-readout" key={`${hand.label}-${index}`}>
                  <div className="hand-title">
                    <span>{hand.label}</span>
                    <small>{Math.round(hand.score * 100)}%</small>
                  </div>
                  <div className="gesture">{hand.gesture}</div>
                  <div className="metric-grid">
                    <span>Pinch ratio</span><strong>{hand.metrics.pinchDistance ?? '—'}</strong>
                    <span>Extended fingers</span><strong>{hand.metrics.extendedCount ?? '—'}</strong>
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="panel-card">
            <div className="panel-title">Prototype settings</div>
            <label className="toggle">
              <input type="checkbox" checked={useWorldCoords} onChange={e => setUseWorldCoords(e.target.checked)} />
              <span>Use world bone offsets (recommended)</span>
            </label>
            <label className="toggle">
              <input type="checkbox" checked={mirrorCamera} onChange={e => setMirrorCamera(e.target.checked)} />
              <span>Mirror camera preview</span>
            </label>
            <label className="toggle">
              <input type="checkbox" checked={showGrid} onChange={e => setShowGrid(e.target.checked)} />
              <span>Show depth grid</span>
            </label>
            <div className="hint hint-block">
              Hand position follows your wrist in the camera frame; world offsets refine finger bends. Pinch when thumb and index tips straddle a shape.
            </div>
            <label className="slider">
              <span>Tracking confidence: {threshold.toFixed(2)}</span>
              <input
                type="range"
                min="0.3"
                max="0.8"
                step="0.05"
                value={threshold}
                onChange={e => resetModelWithThreshold(Number(e.target.value))}
              />
            </label>
          </div>

          {error && (
            <div className="panel-card error">
              <div className="panel-title">Error</div>
              <p>{error}</p>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
