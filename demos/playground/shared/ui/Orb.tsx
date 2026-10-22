import type { CSSProperties } from 'react';
import orb from '../assets/figma-orb.png';

export function Orb({
  active = false,
  level = 0,
  small = false,
}: {
  active?: boolean;
  level?: number;
  small?: boolean;
}) {
  return (
    <div
      className={`orb-wrap ${active ? 'is-active' : ''} ${small ? 'small' : ''}`}
      aria-hidden="true"
      style={{ '--voice-scale': 1 + Math.min(level * 2, 0.13) } as CSSProperties}
    >
      <img className="orb" src={orb} alt="" draggable={false} />
    </div>
  );
}
