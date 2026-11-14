import React, { useState, useEffect } from 'react';
import { ArrowLeft, Download, RefreshCw, Mic, Clapperboard, AlertTriangle, CheckCircle2 } from 'lucide-react';
import type { ViralClip, RenderSettings, BatchRenderProgress, TranscriptLine } from '../types';

export type ClipDesign = 'podcast' | 'film';

const DESIGN_KEY = 'cheat_clip_simple_design';
const FILM_LABEL_KEY = 'cheat_clip_film_label';

interface ClipStudioSectionProps {
  videoUrl: string;
  videoId: string;
  allClips: ViralClip[];
  markedClips: ViralClip[];
  activeClip: ViralClip | null;
  onStartRender: (settings: RenderSettings) => void;
  isRendering: boolean;
  onToggleMarkClip?: (clip: ViralClip) => void;
  onToggleAllClips?: (forceSelect?: boolean) => void;
  batchProgress?: BatchRenderProgress | null;
  onDismissProgress?: () => void;
  onRetryClip?: (clipIndex?: number) => void;
  allowAppUpdates?: boolean;
  onOpenCookiesModal?: () => void;
  onBackToClips?: () => void;
  transcript?: TranscriptLine[];
  videoTitle?: string;
}

const fmt = (s: number) => {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
};

const clipKey = (c: ViralClip) => `${c.start_time}_${c.end_time}`;

const box: React.CSSProperties = { background: '#08080c', border: '1px solid #201e2c', padding: '1rem' };

export const ClipStudioSection: React.FC<ClipStudioSectionProps> = ({
  allClips,
  markedClips,
  onStartRender,
  isRendering,
  onToggleMarkClip,
  onToggleAllClips,
  batchProgress,
  onDismissProgress,
  onRetryClip,
  onBackToClips,
  videoTitle,
}) => {
  const [design, setDesign] = useState<ClipDesign>(() =>
    localStorage.getItem(DESIGN_KEY) === 'film' ? 'film' : 'podcast'
  );
  const [filmLabel, setFilmLabel] = useState<string>(() => localStorage.getItem(FILM_LABEL_KEY) ?? '');
  const [titles, setTitles] = useState<Record<string, string>>({});

  useEffect(() => { localStorage.setItem(DESIGN_KEY, design); }, [design]);
  useEffect(() => { localStorage.setItem(FILM_LABEL_KEY, filmLabel); }, [filmLabel]);

  // Auto-fill label from video title whenever a new film is loaded or design switches to film
  useEffect(() => {
    if (design === 'film' && videoTitle) {
      setFilmLabel(`🎬 ${videoTitle}`);
    }
  }, [design, videoTitle]);

  const markedKeys = new Set(markedClips.map(clipKey));
  const titleOf = (c: ViralClip) => titles[clipKey(c)] ?? (c.title_suggestion || c.title || '');
  const allSelected = allClips.length > 0 && allClips.every(c => markedKeys.has(clipKey(c)));
  const running = isRendering || batchProgress?.overall_status === 'running';

  const startRender = () => {
    if (markedClips.length === 0) return;
    const selectedClips = markedClips.map(c => ({ ...c, custom_title: titleOf(c) } as ViralClip));
    const settings: RenderSettings = {
      renderEngine: 'server',
      hardwareAccel: 'auto',
      aspectRatio: '16:9', // 16:9 video centered on black 9:16 canvas
      backgroundStyle: 'black',
      enableFaceTracking: false,
      streamerPreset: 'none',
      titlePosition: 'auto',
      titleDuration: 'entire',
      captionStyle: design === 'podcast' ? 'clean_minimal' : 'none',
      captionFont: 'Inter',
      titleFont: 'Inter',
      fontSize: 'medium',
      textCase: 'capitalize',
      titleTextCase: 'capitalize',
      subtitlePositionMode: 'bottom',
      subtitleYPercent: 26,
      renderMode: 'separate',
      design,
      filmLabel: design === 'film' ? filmLabel : undefined,
      selectedClips,
    };
    onStartRender(settings);
  };

  const designBtn = (id: ClipDesign, label: string, hint: string, Icon: typeof Mic) => (
    <button
      type="button"
      onClick={() => setDesign(id)}
      style={{
        flex: 1, padding: '1rem', cursor: 'pointer', textAlign: 'left',
        background: design === id ? 'rgba(153,0,0,0.25)' : '#0a0a0f',
        border: design === id ? '2px solid #ff1e27' : '1px solid #201e2c',
        color: '#fff', display: 'flex', flexDirection: 'column', gap: '0.35rem',
      }}
    >
      <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '1rem' }}>
        <Icon size={18} /> {label}
      </span>
      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{hint}</span>
    </button>
  );

  return (
    <section className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0, fontSize: '1.1rem' }}>AUTO CLIPPER</h2>
        {onBackToClips && (
          <button type="button" className="glowing-btn" onClick={onBackToClips} style={{ padding: '0.35rem 0.9rem', fontSize: '0.78rem', background: 'transparent' }}>
            <ArrowLeft size={13} /> 02 CLIPS
          </button>
        )}
      </div>

      {/* Design choice */}
      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
        {designBtn('podcast', 'PODCAST', 'Judul di atas video + subtitle di bawah', Mic)}
        {designBtn('film', 'FILM', 'Label film + judul di bawah video', Clapperboard)}
      </div>

      {design === 'film' && (
        <div style={box}>
          <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Label film (kotak putih)</label>
          <input
            className="form-input"
            value={filmLabel}
            onChange={e => setFilmLabel(e.target.value)}
            placeholder='🎬Film "Judul" (Studio)'
            style={{ width: '100%', marginTop: '0.4rem' }}
          />
        </div>
      )}

      {/* Clips to render */}
      <div style={box}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <strong style={{ fontSize: '0.85rem' }}>PILIH KLIP YANG DI-RENDER ({markedClips.length}/{allClips.length})</strong>
          {onToggleAllClips && (
            <button type="button" className="action-link-btn" onClick={() => onToggleAllClips(!allSelected)}>
              {allSelected ? 'Batal pilih semua' : 'Pilih semua'}
            </button>
          )}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '420px', overflowY: 'auto' }}>
          {allClips.map((c, i) => (
            <div key={clipKey(c)} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', padding: '0.4rem', background: markedKeys.has(clipKey(c)) ? 'rgba(255,30,39,0.08)' : 'transparent' }}>
              <input type="checkbox" checked={markedKeys.has(clipKey(c))} onChange={() => onToggleMarkClip?.(c)} style={{ width: 16, height: 16, accentColor: '#ff1e27' }} />
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', minWidth: 90 }}>
                #{i + 1} {fmt(c.start_time)}-{fmt(c.end_time)}
              </span>
              <input
                className="form-input"
                value={titleOf(c)}
                onChange={e => setTitles(prev => ({ ...prev, [clipKey(c)]: e.target.value }))}
                title="Judul di video. Bungkus kata dengan *kata* untuk highlight hijau."
                style={{ flex: 1, padding: '0.35rem 0.6rem', fontSize: '0.82rem' }}
              />
            </div>
          ))}
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.6rem' }}>
          Tips: bungkus kata dengan *tanda bintang* untuk highlight hijau. Tanpa tanda, kata pertama otomatis di-highlight.
        </div>
      </div>

      <button
        type="button"
        className="glowing-btn"
        onClick={startRender}
        disabled={running || markedClips.length === 0}
        style={{ padding: '0.8rem 1.5rem', fontWeight: 700, background: '#ff1e27', color: '#fff', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', opacity: running || markedClips.length === 0 ? 0.5 : 1 }}
      >
        {running ? <RefreshCw size={15} className="spinner-icon" /> : <Download size={15} />}
        {running ? 'RENDERING...' : `RENDER ${markedClips.length} KLIP (${design.toUpperCase()})`}
      </button>

      {/* Progress */}
      {batchProgress && (
        <div style={box}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.6rem' }}>
            <strong style={{ fontSize: '0.85rem' }}>
              {batchProgress.overall_status === 'completed' ? 'SELESAI' : batchProgress.overall_status === 'error' ? 'ERROR' : 'PROSES'}
            </strong>
            <span style={{ display: 'flex', gap: '0.6rem' }}>
              {batchProgress.zip_url && (
                <a href={batchProgress.zip_url} className="glowing-btn" style={{ padding: '0.25rem 0.7rem', fontSize: '0.75rem' }}>Download ZIP</a>
              )}
              {onDismissProgress && batchProgress.overall_status !== 'running' && (
                <button type="button" className="action-link-btn" onClick={onDismissProgress}>Tutup</button>
              )}
            </span>
          </div>
          {batchProgress.error_message && (
            <div style={{ color: '#fca5a5', fontSize: '0.78rem', marginBottom: '0.5rem' }}>
              <AlertTriangle size={13} /> {batchProgress.error_message}
            </div>
          )}
          {batchProgress.clips.map(c => (
            <div key={c.clip_index} style={{ marginBottom: '0.55rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', gap: '0.5rem' }}>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.title}</span>
                <span style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexShrink: 0 }}>
                  {c.status === 'completed' && c.download_url ? (
                    <a href={c.download_url} style={{ color: '#4ade80', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <CheckCircle2 size={13} /> Download
                    </a>
                  ) : c.status === 'error' ? (
                    <>
                      <span style={{ color: '#ff6b6b' }}>error</span>
                      {onRetryClip && <button type="button" className="action-link-btn" onClick={() => onRetryClip(c.clip_index)}>Retry</button>}
                    </>
                  ) : (
                    <span style={{ color: 'var(--text-muted)' }}>{c.status} {Math.round(c.progress_percent)}%</span>
                  )}
                </span>
              </div>
              {c.status === 'error' && (c.error_message || c.error) && (
                <div style={{ color: '#fca5a5', fontSize: '0.7rem', wordBreak: 'break-word' }}>{c.error_message || c.error}</div>
              )}
              <div style={{ height: 4, background: 'rgba(255,255,255,0.06)', marginTop: 3 }}>
                <div style={{ height: '100%', width: `${c.status === 'completed' ? 100 : c.progress_percent}%`, background: c.status === 'error' ? '#ff1e27' : '#00ff66', transition: 'width .3s' }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
};
