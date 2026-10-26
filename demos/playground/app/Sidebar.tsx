import { ArrowUpRight, ChevronLeft, PanelLeftOpen, Plus, Sparkles } from 'lucide-react';
import type { CaseDefinition, CaseId } from '../shared/session/types';
import brand from '../shared/assets/figma-brand.svg';
import { DesignIcon } from '../shared/ui/DesignIcon';
import { cases } from './registry';
import { IconButton } from '../shared/ui/controls';

export function Sidebar({
  selected,
  onSelect,
  collapsed,
  onCollapse,
  onNew,
  onAbout,
}: {
  selected: CaseId;
  onSelect: (id: CaseId) => void;
  collapsed: boolean;
  onCollapse: () => void;
  onNew: () => void;
  onAbout: () => void;
}) {
  function item(definition: CaseDefinition) {
    const Icon = definition.icon;
    const active = selected === definition.id;
    return (
      <button
        id={`tab-${definition.id}`}
        role="tab"
        aria-controls="experience-panel"
        aria-selected={active}
        tabIndex={active ? 0 : -1}
        key={definition.id}
        className={`case-tab ${active ? 'selected' : ''}`}
        onClick={() => onSelect(definition.id)}
        title={collapsed ? definition.label : undefined}
      >
        {definition.id === 'uicard' ? (
          <DesignIcon name="layout" />
        ) : (
          <Icon size={20} strokeWidth={1.5} />
        )}
        <span>{definition.label}</span>
        {active && <i className="tab-dot" />}
      </button>
    );
  }
  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-top">
        <button className="brand" onClick={onAbout} aria-label="关于 MiniCPM-o Playground">
          <img className="brand-symbol" src={brand} alt="" />
          <span className="brand-word">
            MiniCPM-o <span>5</span>
          </span>
        </button>
        <IconButton label={collapsed ? '展开导航' : '收起导航'} onClick={onCollapse}>
          {collapsed ? <PanelLeftOpen size={17} /> : <ChevronLeft size={17} />}
        </IconButton>
      </div>
      <button className="new-session" onClick={onNew} title="新建对话">
        <Plus size={18} />
        <span>新建对话</span>
      </button>
      <div
        className="nav-scroll"
        role="tablist"
        aria-label="体验类型"
        aria-orientation="vertical"
        onKeyDown={(e) => {
          const focusedId = (e.target as HTMLElement).closest('[role=tab]')?.id.replace('tab-', '');
          const index = cases.findIndex((c) => c.id === (focusedId || selected));
          let target = index;
          if (e.key === 'ArrowDown') target = (index + 1) % cases.length;
          else if (e.key === 'ArrowUp') target = (index + cases.length - 1) % cases.length;
          else if (e.key === 'Home') target = 0;
          else if (e.key === 'End') target = cases.length - 1;
          else return;
          e.preventDefault();
          document.getElementById(`tab-${cases[target].id}`)?.focus();
        }}
      >
        <div className="nav-group-label">对话体验</div>
        {cases.filter((c) => c.group === 'conversation').map(item)}
        <div className="nav-group-label agent-label">
          探索 Agent <span>5</span>
        </div>
        {cases.filter((c) => c.group === 'agent').map(item)}
      </div>
      <div className="sidebar-footer">
        <button className="about-card" onClick={onAbout}>
          <span className="about-icon">
            <Sparkles size={17} />
          </span>
          <span>
            听见，看见，然后行动<small>认识 MiniCPM-o</small>
          </span>
          <ArrowUpRight size={14} />
        </button>
        <a
          className="docs-link"
          href="https://github.com/OpenBMB/MiniCPM-o"
          target="_blank"
          rel="noreferrer"
        >
          <DesignIcon name="document" size={18} />
          <span>项目与文档</span>
          <ArrowUpRight size={12} />
        </a>
        <div className="sidebar-bottom">
          <span>OpenBMB × 面壁智能</span>
          <small>o5</small>
        </div>
      </div>
    </aside>
  );
}
