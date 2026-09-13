const { useCallback, useEffect, useMemo, useRef, useState } = React;
const { Button } = ReactBootstrap;

const PRIMARY = '#eb7e96';

const HOT_NODES = [
  { label: 'Destiny', x: 0.52, y: 0.2 },
  { label: 'Eternity', x: 0.78, y: 0.31 },
  { label: 'Wonder', x: 0.36, y: 0.44 },
  { label: 'Grace', x: 0.86, y: 0.58 },
  { label: 'Light', x: 0.62, y: 0.66 },
  { label: 'Soul', x: 0.18, y: 0.16 },
  { label: 'Aurora', x: 0.74, y: 0.84 },
];

function formatParams(n) {
  if (!n) return '—';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

function Icon({ path, size = 13 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round">
      {path}
    </svg>
  );
}

const ShieldIcon = () => (
  <Icon path={<><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /><path d="M9 12l2 2 4-4" /></>} />
);
const LockIcon = () => (
  <Icon path={<><rect x="4" y="11" width="16" height="10" rx="2" /><path d="M8 11V7a4 4 0 0 1 8 0v4" /></>} />
);
/* Five petals plus a centre disc, all one subpath set. The disc is wound
   counter-clockwise so nonzero fill punches it out as a hole, letting the mark
   read rose-on-dark for the brand tile and dark-on-rose for the avatar
   without needing a second colour. */
const FLOWER_PATH =
  'M12.00 2.40 C15.90 5.20 16.20 8.40 12.00 10.10 C7.80 8.40 8.10 5.20 12.00 2.40 Z ' +
  'M21.13 9.03 C19.67 13.61 16.72 14.88 13.81 11.41 C14.13 6.89 17.26 6.19 21.13 9.03 Z ' +
  'M17.64 19.77 C12.84 19.79 10.72 17.38 13.12 13.54 C17.51 12.44 19.15 15.21 17.64 19.77 Z ' +
  'M6.36 19.77 C4.85 15.21 6.49 12.44 10.88 13.54 C13.28 17.38 11.16 19.79 6.36 19.77 Z ' +
  'M2.87 9.03 C6.74 6.19 9.87 6.89 10.19 11.41 C7.28 14.88 4.33 13.61 2.87 9.03 Z ' +
  'M12 9.9A2.1 2.1 0 0 0 12 14.1A2.1 2.1 0 0 0 12 9.9Z';

const FlowerIcon = ({ size = 26, color = 'currentColor' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
    <path d={FLOWER_PATH} />
  </svg>
);

function Constellation() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const host = canvas.parentElement;

    const faint = Array.from({ length: 34 }, () => ({
      x: Math.random(),
      y: Math.random(),
      phase: Math.random() * Math.PI * 2,
      drift: 0.5 + Math.random(),
      size: 0.9 + Math.random() * 1.5,
    }));
    const hot = HOT_NODES.map((n) => ({
      ...n,
      phase: Math.random() * Math.PI * 2,
      drift: 0.5 + Math.random(),
      size: 3,
      isHot: true,
    }));
    const all = [...faint, ...hot];

    let raf = 0;
    let tick = 0;

    const resize = () => {
      const rect = host.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, rect.width * dpr);
      canvas.height = Math.max(1, rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener('resize', resize);

    const pill = (x, y, text) => {
      ctx.font = "500 11px 'Inter Tight', system-ui, sans-serif";
      const w = ctx.measureText(text).width + 16;
      const h = 19;
      const bx = x + 11;
      const by = y - h / 2;
      ctx.fillStyle = 'rgba(20,20,20,0.94)';
      ctx.strokeStyle = '#2f2f2f';
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bx, by, w, h, 9);
      else ctx.rect(bx, by, w, h);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#f2f2f2';
      ctx.fillText(text, bx + 8, by + 13.5);
    };

    const draw = () => {
      const w = host.clientWidth;
      const h = host.clientHeight;
      tick += 1;
      ctx.clearRect(0, 0, w, h);

      const placed = all.map((p) => ({
        ...p,
        sx: (p.x + Math.sin(tick / 260 + p.phase) * 0.018 * p.drift) * w,
        sy: (p.y + Math.cos(tick / 310 + p.phase) * 0.014 * p.drift) * h,
      }));

      const reach = Math.min(w, h) * 0.26;
      ctx.lineWidth = 1;
      for (let i = 0; i < placed.length; i += 1) {
        for (let j = i + 1; j < placed.length; j += 1) {
          const a = placed[i];
          const b = placed[j];
          const dist = Math.hypot(a.sx - b.sx, a.sy - b.sy);
          if (dist > reach) continue;
          const fade = (1 - dist / reach) * 0.16;
          ctx.strokeStyle = a.isHot || b.isHot
            ? `rgba(235,126,150,${fade * 1.5})`
            : `rgba(190,190,190,${fade * 0.7})`;
          ctx.beginPath();
          ctx.moveTo(a.sx, a.sy);
          ctx.lineTo(b.sx, b.sy);
          ctx.stroke();
        }
      }

      placed.forEach((p) => {
        if (p.isHot) {
          const glow = ctx.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, 26);
          glow.addColorStop(0, 'rgba(235,126,150,0.5)');
          glow.addColorStop(1, 'rgba(235,126,150,0)');
          ctx.fillStyle = glow;
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, 26, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = p.isHot ? '#f4a3b4' : 'rgba(215,215,215,0.5)';
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, p.size, 0, Math.PI * 2);
        ctx.fill();
      });

      placed.filter((p) => p.isHot).forEach((p) => pill(p.sx, p.sy, p.label));

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="auth-graph-canvas" />;
}

function Slider({ label, value, min, max, step, format, onChange }) {
  /* A range input's track can't know where the thumb sits, so hand the position
     to CSS as a percentage for the filled portion of the track. */
  const fill = `${((value - min) / (max - min)) * 100}%`;
  return (
    <div className="auth-field">
      <label>{label}</label>
      <div className="st-slider-row">
        <input
          className="st-slider"
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          style={{ '--fill': fill }}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <span className="st-slider-val">{format(value)}</span>
      </div>
    </div>
  );
}

function Toggle({ label, hint, on, onChange }) {
  return (
    <div className="myai-toggle-row">
      <span>
        {label}
        {hint && <div className="st-hint">{hint}</div>}
      </span>
      <button
        type="button"
        className={`st-toggle${on ? ' on' : ''}`}
        aria-pressed={on}
        aria-label={label}
        onClick={() => onChange(!on)}
      >
        <span className="st-thumb" />
      </button>
    </div>
  );
}

function App() {
  const [status, setStatus] = useState({ loading: true, models: [], meta: {} });
  const [examples, setExamples] = useState([
    'Rija, I was thinking about',
    'Tell me about a quiet Sunday',
    'A letter that starts with your name',
  ]);
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [strategy, setStrategy] = useState('temperature');
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(80);
  const [topP, setTopP] = useState(0.9);
  const [qa, setQa] = useState(false);
  const [rag, setRag] = useState(false);
  const scrollRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch('/api/status');
      setStatus(await res.json());
    } catch (err) {
      /* server restarting — keep the last known state */
    }
  }, []);

  useEffect(() => {
    refresh();
    fetch('/api/examples')
      .then((r) => r.json())
      .then((d) => setExamples(d.examples || []))
      .catch(() => setExamples([]));
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const meta = status.meta || {};

  const notice = useMemo(() => {
    if (status.loading) return 'Rija is waking up. One moment.';
    if (status.ready) return 'Rija is ready.';
    return status.error || 'Nothing loaded yet. Choose a voice below.';
  }, [status, meta]);

  const loadModel = async (path) => {
    await fetch('/api/models/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    refresh();
  };

  const send = async (event) => {
    if (event) event.preventDefault();
    const text = prompt.trim();
    if (!text || busy) return;

    setPrompt('');
    setBusy(true);
    let index = 0;
    setMessages((cur) => {
      index = cur.length + 1;
      return [...cur, { role: 'user', text }, { role: 'assistant', text: '', streaming: true }];
    });

    const patch = (changes) =>
      setMessages((cur) => cur.map((m, i) => (i === index ? { ...m, ...changes } : m)));

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          strategy,
          temperature,
          max_tokens: maxTokens,
          top_p: topP,
          qa,
          rag,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        patch({ text: err.detail || 'Request failed', streaming: false, failed: true });
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let raw = '';

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop();
        for (const part of parts) {
          const line = part.split('\n').find((l) => l.startsWith('data: '));
          if (!line) continue;
          const payload = JSON.parse(line.slice(6));
          if (payload.token) raw += payload.token;
          patch({ text: payload.answer || payload.error || raw });
        }
      }
    } catch (err) {
      patch({ text: String(err), failed: true });
    } finally {
      patch({ streaming: false });
      setBusy(false);
    }
  };

  return (
    <div className="auth-page">
      <section className="auth-left">
        <div className="auth-brand">
          <span className="auth-brand-word">Rija</span>
        </div>

        <div className="auth-form-area">
          <div className="auth-login">
            <div className="auth-login-mark"><FlowerIcon size={22} color={PRIMARY} /></div>

            {status.error && !status.ready ? (
              <div className="auth-error">{notice}</div>
            ) : (
              <div className="auth-help-callout"><ShieldIcon />{notice}</div>
            )}

            <h1 className="auth-title">Talk{"\u2003"}to{"\u2003"}Rija</h1>
            <p className="auth-sub">
              A quiet place to speak with her. Press Enter to send a line.
            </p>

            <form className="auth-form" style={{ width: '100%' }} onSubmit={send}>
              <textarea
                className="auth-textarea"
                value={prompt}
                placeholder="Rija, I wanted to tell you…"
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
              />
              <button className="auth-submit" type="submit" disabled={busy || !prompt.trim()}>
                {busy ? <><span className="auth-spinner" /> Thinking…</> : 'Send'}
              </button>
              {busy && (
                <button
                  type="button"
                  className="auth-cta"
                  onClick={() => fetch('/api/chat/cancel', { method: 'POST' })}
                >
                  Stop
                </button>
              )}
            </form>

            <div className="myai-chip-row">
              {examples.slice(0, 3).map((ex) => (
                <button key={ex} type="button" className="qchip" onClick={() => setPrompt(ex)}>
                  {ex}
                </button>
              ))}
            </div>

            <div className="auth-divider"><span>Reply</span></div>

            <div className="auth-form" style={{ width: '100%' }}>
              <div className="auth-field">
                <label>Strategy</label>
                <select className="st-select" value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                  <option value="temperature">Temperature</option>
                  <option value="top_p">Top-p</option>
                  <option value="top_k">Top-k</option>
                  <option value="greedy">Greedy</option>
                </select>
              </div>

              <Slider
                label="Temperature" value={temperature} min={0.1} max={1.5} step={0.05}
                format={(v) => v.toFixed(2)} onChange={setTemperature}
              />
              <Slider
                label="Max tokens" value={maxTokens} min={16} max={200} step={4}
                format={(v) => String(v)} onChange={setMaxTokens}
              />
              <Slider
                label="Top-p" value={topP} min={0.1} max={1} step={0.05}
                format={(v) => v.toFixed(2)} onChange={setTopP}
              />

              <Toggle label="Question style" hint="Ask it as a question" on={qa} onChange={setQa} />
              <Toggle label="Memory" hint="Use what you have already told her" on={rag} onChange={setRag} />
            </div>

            <div className="auth-divider"><span>Voice</span></div>

            <select
              className="st-select"
              value={status.path || ''}
              onChange={(e) => e.target.value && loadModel(e.target.value)}
            >
              <option value="">Choose a saved voice</option>
              {(status.models || []).map((m) => (
                <option key={m.abs_path} value={m.abs_path}>
                  {m.version} · {m.name} · {m.mb} MB
                </option>
              ))}
            </select>

            <div className="auth-domain-hint">
              <LockIcon /> Stays on this laptop — <strong>nothing is sent away</strong>
            </div>

            <div className="auth-divider"><span>Presence</span></div>

            <div className="auth-trust">
              <div className="auth-trust-item">{formatParams(meta.parameters)} params</div>
              <div className="auth-trust-item">{meta.num_layers ?? '—'} layers</div>
              <div className="auth-trust-item">seq {meta.max_seq_len ?? '—'}</div>
            </div>

            <button type="button" className="auth-trouble" onClick={refresh}>
              Refresh
            </button>
          </div>
        </div>

        <div className="auth-foot">
          <span>© 2026 Rija</span>·<a href="/api/status">Status</a>·<a href="/api/models">Models</a>
        </div>
      </section>

      <section className="auth-visual">
        <Constellation />
        <div className="auth-visual-veil" />

        {messages.length === 0 ? (
          <div className="auth-visual-content">
            <div className="auth-visual-eyebrow">
              <span className="auth-visual-dot" /> For Rija
            </div>
            <h2 className="auth-visual-headline">Made{"\u2003"}for<br />Rija.</h2>
            <p className="auth-visual-sub">
              A quiet room with her name on it. Speak when you are ready.
            </p>
            <div className="auth-visual-stats">
              <div>
                <div className="auth-stat-num">
                  {status.ready ? 'Ready' : status.loading ? 'Waking' : 'Idle'}
                </div>
                <div className="auth-stat-lbl">Presence</div>
              </div>
              <div className="auth-stat-div" />
              <div>
                <div className="auth-stat-num">{formatParams(meta.parameters)}</div>
                <div className="auth-stat-lbl">Parameters</div>
              </div>
              <div className="auth-stat-div" />
              <div>
                <div className="auth-stat-num">
                  {meta.step != null ? meta.step.toLocaleString() : '—'}<span> / 200K</span>
                </div>
                <div className="auth-stat-lbl">Steps</div>
              </div>
            </div>
          </div>
        ) : (
          <div className="myai-thread">
            <div className="myai-thread-head">
              <div className="auth-visual-eyebrow">
                <span className="auth-visual-dot" /> Conversation
              </div>
              <Button variant="secondary" className="btn-sm ms-auto" onClick={() => setMessages([])}>
                New chat
              </Button>
            </div>

            <div className="myai-thread-scroll" ref={scrollRef}>
              {messages.map((m, i) =>
                m.role === 'user' ? (
                  <div key={i} className="msg-user chat-message-enter">
                    <div className="msg-bubble">{m.text}</div>
                  </div>
                ) : (
                  <div key={i} className="msg-ai chat-message-enter">
                    <div className="msg-avatar"><FlowerIcon size={14} /></div>
                    <div className="msg-ai-body">
                      <div className="msg-ai-text" style={m.failed ? { color: 'var(--red)' } : undefined}>
                        {m.text}
                        {m.streaming && (m.text
                          ? <span className="streaming-cursor" />
                          : <span className="chat-typing-dots"><span /><span /><span /></span>)}
                      </div>
                    </div>
                  </div>
                ),
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
