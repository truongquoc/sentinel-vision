const $ = (s) => document.querySelector(s);
const SVGNS = 'http://www.w3.org/2000/svg';
const CLASSES = ['person', 'car', 'truck', 'motorcycle', 'bicycle', 'bus', 'dog', 'cat'];
const EVENT_NAMES = { zone_enter: 'entered', loitering: 'is loitering in', zone_exit: 'left' };
const COLORS = ['#4fd18b', '#f2b84b', '#6aa9ff', '#ef6f6c', '#c48af5', '#4dd0d4'];

const state = { camera: null, rules: [], drawing: null, editing: null, streaming: false, labels: new Set() };

const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const when = (iso) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' });

async function api(path, opts = {}) {
  const res = await fetch('/api' + path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (res.status === 204) return null;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(formatError(body) || res.statusText);
  return body;
}

function formatError(body) {
  if (typeof body.detail === 'string') return body.detail;
  if (Array.isArray(body.detail)) return body.detail.map((d) => `${d.loc.slice(-1)[0]}: ${d.msg}`).join('; ');
  return '';
}

// ---------------------------------------------------------------- health
async function refreshHealth() {
  let health;
  try { health = await api('/health'); } catch {
    $('#edge').textContent = 'UNREACHABLE';
    return;
  }
  const cam = health.cameras[0];
  state.camera = cam;
  $('#edge').textContent = health.status.toUpperCase();
  $('#cameraName').textContent = cam.name;
  $('#cameraState').textContent = cam.online ? 'ONLINE' : cam.running ? 'CONNECTING' : 'STOPPED';
  $('#cameraMeta').textContent = cam.online ? `${cam.frame_size?.join('×') ?? ''} · receiving frames` : (cam.error || 'Not started');
  $('#start').hidden = cam.running;
  $('#detectorState').textContent = health.detector.ready ? 'AI READY' : 'AI UNAVAILABLE';
  $('#detectorMeta').textContent = health.detector.detail;
  $('#pipelineState').textContent = cam.running ? `${cam.processing_fps ?? 0} fps` : '—';
  $('#pipelineMeta').textContent = cam.running ? `${cam.inference_ms ?? '–'} ms inference · ${cam.active_tracks ?? 0} tracked` : 'Idle';
  $('#eventCount').textContent = health.events_today;
  const n = health.notifier;
  $('#notifierMeta').textContent = n.enabled ? `Webhook: ${n.delivered} sent · ${n.failed} failed` : 'Webhook disabled';
  $('#subline').textContent = cam.pipeline_error ? `Pipeline error: ${cam.pipeline_error}` : 'Local AI processing · edge node';
  if (cam.frame_size) $('#video').style.aspectRatio = `${cam.frame_size[0]} / ${cam.frame_size[1]}`;
  updateFeed(cam);
  renderRuleOccupancy(cam.zone_occupancy || {});
}

function updateFeed(cam) {
  const live = cam.online;
  $('#feedState').textContent = live ? 'LIVE' : 'OFFLINE';
  $('#feedState').className = 'badge' + (live ? ' live' : '');
  $('#empty').hidden = live;
  $('#emptyDetail').textContent = cam.error || 'Waiting for camera connection';
  if (live && !state.streaming) {
    $('#feed').src = `/api/cameras/${cam.id}/mjpeg?t=${Date.now()}`;
    state.streaming = true;
  }
  if (!live && state.streaming) {
    $('#feed').removeAttribute('src');
    state.streaming = false;
  }
}
$('#feed').addEventListener('error', () => { state.streaming = false; });

$('#start').onclick = async () => {
  if (!state.camera) return;
  try { await api(`/cameras/${state.camera.id}/start`, { method: 'POST' }); } catch (e) { $('#subline').textContent = e.message; }
  setTimeout(refreshHealth, 600);
};

// ---------------------------------------------------------------- zones overlay
function color(i) { return COLORS[i % COLORS.length]; }

function renderOverlay() {
  const svg = $('#overlay');
  svg.replaceChildren();
  if ($('#showZones').checked || state.drawing) {
    state.rules.forEach((rule, i) => {
      const poly = document.createElementNS(SVGNS, 'polygon');
      poly.setAttribute('points', rule.config.polygon.map((p) => p.join(',')).join(' '));
      poly.setAttribute('class', 'zone' + (rule.enabled ? '' : ' paused'));
      poly.style.setProperty('--zone', color(i));
      poly.dataset.rule = rule.id;
      svg.append(poly);
    });
  }
  if (!state.drawing) return;
  const pts = state.drawing;
  const line = document.createElementNS(SVGNS, pts.length >= 3 ? 'polygon' : 'polyline');
  line.setAttribute('points', pts.map((p) => p.join(',')).join(' '));
  line.setAttribute('class', 'draft');
  svg.append(line);
  pts.forEach(([x, y]) => {
    const dot = document.createElementNS(SVGNS, 'ellipse');
    const rect = svg.getBoundingClientRect();
    dot.setAttribute('cx', x); dot.setAttribute('cy', y);
    dot.setAttribute('rx', 5 / rect.width); dot.setAttribute('ry', 5 / rect.height);
    dot.setAttribute('class', 'vertex');
    svg.append(dot);
  });
}
$('#showZones').onchange = renderOverlay;

function setDrawing(points) {
  state.drawing = points;
  $('#drawbar').hidden = points === null;
  $('#video').classList.toggle('drawing', points !== null);
  $('#drawZone').hidden = points !== null;
  if (points) {
    $('#finishDraw').textContent = `Finish (${points.length})`;
    $('#finishDraw').disabled = points.length < 3;
  }
  renderOverlay();
}

$('#drawZone').onclick = () => setDrawing([]);
$('#cancelDraw').onclick = () => setDrawing(null);
$('#undoPoint').onclick = () => setDrawing(state.drawing.slice(0, -1));
$('#overlay').addEventListener('click', (e) => {
  if (!state.drawing) return;
  const r = e.currentTarget.getBoundingClientRect();
  const clamp = (v) => Math.min(1, Math.max(0, +v.toFixed(4)));
  setDrawing([...state.drawing, [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)]]);
});
$('#finishDraw').onclick = () => openRuleDialog(null, state.drawing);

// ---------------------------------------------------------------- rule dialog
$('#classChips').innerHTML = CLASSES.map((c) =>
  `<label><input type="checkbox" name="cls" value="${c}" ${c === 'person' ? 'checked' : ''} data-testid="rule-class-${c}"> ${c}</label>`).join('');

function openRuleDialog(rule, polygon) {
  const f = $('#ruleForm');
  state.editing = { rule, polygon: polygon || rule.config.polygon };
  $('#ruleDialogTitle').textContent = rule ? 'Edit zone' : 'New zone';
  $('#ruleError').textContent = '';
  const cfg = rule?.config ?? { classes: ['person'], events: ['enter', 'loiter'], loiter_seconds: 30, severity: 'medium' };
  f.name.value = rule?.name ?? `Zone ${state.rules.length + 1}`;
  f.querySelectorAll('[name=cls]').forEach((el) => { el.checked = cfg.classes.includes(el.value); });
  f.querySelectorAll('[name=ev]').forEach((el) => { el.checked = cfg.events.includes(el.value); });
  f.loiter.value = cfg.loiter_seconds ?? 30;
  f.severity.value = cfg.severity;
  $('#ruleDialog').showModal();
}

$('#ruleForm').addEventListener('submit', async (e) => {
  if (e.submitter?.value !== 'save') { setDrawing(null); return; }
  e.preventDefault();
  const f = e.target;
  const events = [...f.querySelectorAll('[name=ev]:checked')].map((el) => el.value);
  const config = {
    polygon: state.editing.polygon,
    classes: [...f.querySelectorAll('[name=cls]:checked')].map((el) => el.value),
    events,
    loiter_seconds: events.includes('loiter') ? Number(f.loiter.value) : null,
    severity: f.severity.value,
  };
  const { rule } = state.editing;
  try {
    if (rule) await api(`/rules/${rule.id}`, { method: 'PUT', body: JSON.stringify({ name: f.name.value, config, enabled: rule.enabled }) });
    else await api('/rules', { method: 'POST', body: JSON.stringify({ camera_id: state.camera.id, name: f.name.value, config }) });
  } catch (err) {
    $('#ruleError').textContent = err.message;
    return;
  }
  $('#ruleDialog').close();
  setDrawing(null);
  refreshRules();
});

// ---------------------------------------------------------------- rule list
async function refreshRules() {
  state.rules = await api('/rules');
  const list = $('#ruleList');
  if (!state.rules.length) {
    list.innerHTML = '<p class="muted">No zones yet. Start the camera, then use <b>+ Draw zone</b> on the live feed.</p>';
    renderOverlay();
    return;
  }
  list.innerHTML = state.rules.map((r, i) => `
    <div class="rule" data-rule="${r.id}" data-testid="rule-item">
      <span class="dot" style="background:${color(i)}"></span>
      <div class="rule-main">
        <b>${esc(r.name)}</b>
        <small>${esc(r.config.classes.join(', ') || 'all objects')} · ${esc(r.config.events.join(' / '))}${r.config.loiter_seconds ? ` · loiter ${r.config.loiter_seconds}s` : ''}</small>
      </div>
      <span class="occ" data-occ="${r.id}" title="Objects inside now"></span>
      <button class="chip-btn" data-act="toggle" data-testid="rule-toggle">${r.enabled ? 'ARMED' : 'PAUSED'}</button>
      <button class="chip-btn" data-act="edit" data-testid="rule-edit">Edit</button>
      <button class="chip-btn danger" data-act="delete" data-testid="rule-delete">Delete</button>
    </div>`).join('');
  renderOverlay();
}

function renderRuleOccupancy(occupancy) {
  document.querySelectorAll('[data-occ]').forEach((el) => {
    const n = occupancy[el.dataset.occ] ?? 0;
    el.textContent = n ? `${n} inside` : '';
  });
}

$('#ruleList').addEventListener('click', async (e) => {
  const btn = e.target.closest('button[data-act]');
  if (!btn) return;
  const rule = state.rules.find((r) => r.id === btn.closest('[data-rule]').dataset.rule);
  if (btn.dataset.act === 'edit') return openRuleDialog(rule);
  if (btn.dataset.act === 'toggle') {
    await api(`/rules/${rule.id}`, { method: 'PUT', body: JSON.stringify({ name: rule.name, config: rule.config, enabled: !rule.enabled }) });
    return refreshRules();
  }
  if (btn.dataset.confirm !== '1') {          // two-step delete without a blocking confirm() dialog
    btn.dataset.confirm = '1';
    btn.textContent = 'Confirm?';
    setTimeout(() => { btn.dataset.confirm = ''; btn.textContent = 'Delete'; }, 3000);
    return;
  }
  await api(`/rules/${rule.id}`, { method: 'DELETE' });
  refreshRules();
});

$('#ruleList').addEventListener('mouseover', (e) => {
  const id = e.target.closest('[data-rule]')?.dataset.rule;
  document.querySelectorAll('#overlay .zone').forEach((p) => p.classList.toggle('hot', p.dataset.rule === id));
});

// ---------------------------------------------------------------- events
async function refreshEvents() {
  const q = new URLSearchParams({ limit: 100 });
  if ($('#filterType').value) q.set('event_type', $('#filterType').value);
  if ($('#filterLabel').value) q.set('label', $('#filterLabel').value);
  if ($('#filterUnacked').checked) q.set('unacknowledged', 'true');
  const events = await api('/events?' + q);
  events.forEach((e) => e.label && state.labels.add(e.label));
  syncLabelFilter();
  $('#eventList').innerHTML = events.length ? events.map(renderEvent).join('') : '<p class="muted">No events match.</p>';
}

function renderEvent(e) {
  const d = e.details || {};
  const dwell = d.dwell_seconds != null ? ` · ${d.dwell_seconds}s in zone` : '';
  const img = e.thumbnail_path || e.snapshot_path;
  const kind = e.thumbnail_path ? 'thumbnail' : 'snapshot';
  return `
    <div class="event ${e.acknowledged_at ? 'acked' : ''}" data-event="${e.id}" data-testid="event-item">
      <div class="severity ${esc(e.severity)}"></div>
      ${img ? `<img src="/api/events/${e.id}/${kind}" alt="" loading="lazy" data-act="open">` : '<div class="noimg"></div>'}
      <div class="event-main" data-act="open">
        <b>${esc(e.label)} #${e.track_id} ${EVENT_NAMES[e.event_type] ?? esc(e.event_type)} <em>${esc(d.rule_name)}</em></b>
        <small>${when(e.occurred_at)}${dwell}</small>
      </div>
      <button data-act="ack" data-testid="event-ack" ${e.acknowledged_at ? 'disabled' : ''}>${e.acknowledged_at ? 'ACKED' : 'Acknowledge'}</button>
    </div>`;
}

function syncLabelFilter() {
  const sel = $('#filterLabel');
  const have = new Set([...sel.options].map((o) => o.value));
  [...state.labels].filter((l) => !have.has(l)).forEach((l) => sel.add(new Option(l, l)));
}

$('#eventList').addEventListener('click', async (e) => {
  const act = e.target.closest('[data-act]')?.dataset.act;
  const id = e.target.closest('[data-event]')?.dataset.event;
  if (!act || !id) return;
  if (act === 'ack') { await api(`/events/${id}/ack`, { method: 'POST' }); return refreshEvents(); }
  const ev = await api(`/events/${id}`);
  $('#eventImage').src = `/api/events/${id}/snapshot`;
  $('#eventInfo').innerHTML = `<h2>${esc(ev.label)} #${ev.track_id} ${EVENT_NAMES[ev.event_type] ?? ''} ${esc(ev.details.rule_name)}</h2>
    <small>${when(ev.occurred_at)} · severity ${esc(ev.severity)}${ev.details.confidence ? ` · ${(ev.details.confidence * 100).toFixed(0)}% confidence` : ''}</small>`;
  $('#eventDialog').showModal();
});
['#filterType', '#filterLabel', '#filterUnacked'].forEach((s) => { $(s).onchange = refreshEvents; });

// ---------------------------------------------------------------- boot
async function boot() {
  await refreshHealth();
  await Promise.all([refreshRules(), refreshEvents()]);
  setInterval(refreshHealth, 2000);
  setInterval(refreshEvents, 4000);
}
window.addEventListener('resize', renderOverlay);
boot();
