import { useEffect, useRef, useState } from 'react';
import type { CaseDefinition, SessionSettings } from '../session/types';
import type { SessionController } from '../session/controller';

export function SceneHost({
  definition,
  controller,
  settings,
  visible,
  onReady,
}: {
  definition: CaseDefinition;
  controller: SessionController;
  settings: SessionSettings;
  visible: boolean;
  onReady: () => void;
}) {
  const stage = useRef<HTMLDivElement>(null);
  const [css, setCss] = useState('');
  useEffect(() => {
    let cancelled = false;
    definition
      .loadScene?.()
      .then(async (factory) => {
        if (cancelled || !stage.current) return;
        setCss(factory.css);
        await controller.mountScene(stage.current, factory, settings);
        if (!cancelled) onReady();
      })
      .catch((error) => {
        if (!cancelled) controller.setError(error);
      });
    return () => {
      cancelled = true;
      controller.unmountScene();
    };
    // Each controller is scoped to one tab activation. Config is applied again at connection time.
  }, [controller]);
  return (
    <div
      className={`scene-host ${visible ? 'visible' : ''}`}
      data-demo={definition.id}
      aria-hidden={!visible}
    >
      <style>{css}</style>
      <div ref={stage} className="scene-mount" />
    </div>
  );
}
