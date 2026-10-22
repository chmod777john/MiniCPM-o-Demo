import type { CSSProperties } from 'react';
import layout from '../assets/figma-layout.svg';
import document from '../assets/figma-document.svg';
import settings from '../assets/figma-settings.svg';
import palette from '../assets/figma-palette.svg';
import code from '../assets/figma-code.svg';

const assets = { layout, document, settings, palette, code };

/** Paths exported from the reference design, tinted by the surrounding component. */
export function DesignIcon({ name, size = 20 }: { name: keyof typeof assets; size?: number }) {
  if (name === 'palette')
    return <img src={palette} width={size} height={size} alt="" className="design-icon-color" />;
  return (
    <span
      aria-hidden="true"
      className="design-icon"
      style={
        {
          width: size,
          height: size,
          '--icon-src': `url("${assets[name]}")`,
        } as CSSProperties
      }
    />
  );
}
