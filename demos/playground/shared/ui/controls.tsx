import { useEffect, useId, useRef, type ButtonHTMLAttributes, type ReactNode } from 'react';
import { ChevronDown, X } from 'lucide-react';

export function IconButton({
  label,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button
      {...props}
      type="button"
      className={`icon-button ${props.className || ''}`}
      aria-label={label}
      title={label}
    >
      {children}
    </button>
  );
}
export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}
export function Toggle({
  label,
  checked,
  onChange,
  disabled = false,
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="toggle-row">
      <label htmlFor={id}>{label}</label>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`switch ${checked ? 'on' : ''}`}
      >
        <span />
      </button>
    </div>
  );
}
export function Section({
  title,
  badge,
  children,
  open = false,
}: {
  title: string;
  badge?: string;
  children: ReactNode;
  open?: boolean;
}) {
  return (
    <details className="settings-section" open={open || undefined}>
      <summary>
        <ChevronDown size={14} />
        <span>{title}</span>
        {badge && <small>{badge}</small>}
      </summary>
      <div className="section-content">{children}</div>
    </details>
  );
}
export function Dialog({
  title,
  children,
  onClose,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
}) {
  const id = useId();
  const element = useRef<HTMLElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const dialog = element.current;
    const focusable = dialog?.querySelector<HTMLElement>(
      '[autofocus], button, [href], input, textarea, select, [tabindex="0"]',
    );
    (focusable || dialog)?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  }, []);
  return (
    <div className="dialog-scrim" onClick={onClose}>
      <section
        ref={element}
        tabIndex={-1}
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={id}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            e.stopPropagation();
            onClose();
          }
          if (e.key !== 'Tab') return;
          const elements = Array.from(
            element.current?.querySelectorAll<HTMLElement>(
              'button:not(:disabled), [href], input:not(:disabled), textarea:not(:disabled), select:not(:disabled), [tabindex="0"]',
            ) || [],
          );
          const first = elements[0],
            last = elements.at(-1);
          if (!first) {
            e.preventDefault();
            return;
          }
          if (e.shiftKey && document.activeElement === first) {
            e.preventDefault();
            last?.focus();
          } else if (!e.shiftKey && document.activeElement === last) {
            e.preventDefault();
            first.focus();
          }
        }}
      >
        <header>
          <h2 id={id}>{title}</h2>
          <IconButton label="关闭" onClick={onClose}>
            <X size={18} />
          </IconButton>
        </header>
        {children}
      </section>
    </div>
  );
}
