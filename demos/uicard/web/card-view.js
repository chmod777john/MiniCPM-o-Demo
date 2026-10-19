// UI Card 的 10 种卡片渲染（hint / emoji / animation / score / weather / stat / list / choice / custom / map）。
//
// 移植自 cjp/o5-fc-dev 分支 static/agent-tools/card-view.js，渲染逻辑未改。卡片对象来自模型的
// show_card 参数或工具拼好的天气 / 地图卡，一律当不可信输入：未知 type 退回 JSON，文本全部转义。
// 地图卡在 sandbox iframe 里画：配置了高德 JS key 用高德 JS API，否则 Leaflet + 高德栅格瓦片。
//
// 与 cjp 原实现的差异（2026-10-18）：底图从 OpenStreetMap 换成高德栅格瓦片，Leaflet 从 unpkg 换成本地 vendor。
// 原因：国内网络访问 tile.openstreetmap.org 不通（开发机实测连接失败），cjp 原样的地图卡只剩灰底 + 路线；
// 高德栅格瓦片（webrd0x.is.autonavi.com）不需要 key、国内可达。它是 GCJ-02 坐标，所以 Leaflet 这条路也要把
// 工具给的 WGS-84 坐标换成 GCJ-02 再画（与高德 JS API 那条路同一个换算）。

const TONES = new Set(['neutral', 'info', 'success', 'warning', 'danger', 'brand']);

// 本地 vendor（BSD-2，见 vendor/leaflet-1.9.4/LICENSE）。iframe 用 srcdoc，相对地址不可靠，按本模块地址拼绝对 URL。
const LEAFLET_CSS = new URL('./vendor/leaflet-1.9.4/leaflet.css', import.meta.url).href;
const LEAFLET_JS = new URL('./vendor/leaflet-1.9.4/leaflet.js', import.meta.url).href;
const AMAP_TILE_URL = 'https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}';
const MAP_BRAND = '#3949ab';
const MAP_PADDING = 24;
const MAP_MARKER_LIMIT = 20;
const MAP_ROUTE_LIMIT = 3;
const MAP_PATH_LIMIT = 300;
const ROUTE_MODES = { driving: '驾车', walking: '步行', transit: '公交', cycling: '骑行' };

const MAP_PIN_CSS = `.pin{width:25px;height:25px;border-radius:13px 13px 13px 3px;background:${MAP_BRAND};color:#fff;`
  + 'transform:rotate(-45deg);display:flex;align-items:center;justify-content:center;border:2px solid #fff;box-shadow:0 2px 5px #0005}'
  + '.pin span{transform:rotate(45deg);font:700 11px sans-serif}';

/**
 * WGS-84 → GCJ-02, the inverse of the shift coords.py applies on the server.
 * AMap's JS tiles are GCJ-02, so plotting the tool's WGS-84 output unshifted
 * puts every pin a few hundred metres off.
 */
const WGS84_TO_GCJ02_JS = `function outOfChina(lat,lng){return lng<72.004||lng>137.8347||lat<0.8293||lat>55.8271}
function delta(lat,lng){const x=lng-105,y=lat-35,pi=Math.PI;
let dLat=-100+2*x+3*y+.2*y*y+.1*x*y+.2*Math.sqrt(Math.abs(x));
dLat+=(20*Math.sin(6*x*pi)+20*Math.sin(2*x*pi))*2/3;dLat+=(20*Math.sin(y*pi)+40*Math.sin(y*pi/3))*2/3;
dLat+=(160*Math.sin(y*pi/12)+320*Math.sin(y*pi/30))*2/3;
let dLng=300+x+2*y+.1*x*x+.1*x*y+.1*Math.sqrt(Math.abs(x));
dLng+=(20*Math.sin(6*x*pi)+20*Math.sin(2*x*pi))*2/3;dLng+=(20*Math.sin(x*pi)+40*Math.sin(x*pi/3))*2/3;
dLng+=(150*Math.sin(x*pi/12)+300*Math.sin(x*pi/30))*2/3;
const rad=lat/180*pi,magic=Math.sin(rad),ee=.00669342162296594323,a=6378245,m=1-ee*magic*magic;
return [dLat*180/((a*(1-ee))/(m*Math.sqrt(m))*pi),dLng*180/(a/Math.sqrt(m)*Math.cos(rad)*pi)]}
function gcj(p){if(outOfChina(p.latitude,p.longitude))return[p.longitude,p.latitude];
const d=delta(p.latitude,p.longitude);return[p.longitude+d[1],p.latitude+d[0]]}`;

/** AMap JS credentials from the defaults endpoint; null until the page loads them. */
let mapDisplayConfig = null;

/**
 * Hand the page's map credentials to the renderer.
 *
 * A module-level store rather than a renderCard argument: both demo pages call
 * renderCard from several places, and only the map card ever needs this.
 */
export function setMapDisplayConfig(config) {
  mapDisplayConfig = config && typeof config === 'object' ? config : null;
}

const CONDITION_ICONS = {
  sunny: '☀️',
  clear: '🌙',
  cloudy: '⛅',
  overcast: '☁️',
  rain: '🌧️',
  showers: '🌦️',
  thunder: '⛈️',
  snow: '❄️',
  fog: '🌫️',
  wind: '💨',
};

export function renderCard(card) {
  if (!card || typeof card !== 'object') return '';
  const type = String(card.type || '').toLowerCase();
  const tone = TONES.has(card.tone) ? card.tone : 'neutral';
  const body = renderBody(type, card);
  return `<article class="card card-${escapeAttr(type || 'unknown')} tone-${tone}">
    <div class="card-type">${escapeHtml(type || 'unknown')}</div>
    ${body}
  </article>`;
}

function renderBody(type, card) {
  switch (type) {
    case 'hint': return renderHint(card);
    case 'emoji': return renderEmoji(card);
    case 'animation': return renderAnimation(card);
    case 'score': return renderScore(card);
    case 'weather': return renderWeather(card);
    case 'stat': return renderStat(card);
    case 'list': return renderList(card);
    case 'choice': return renderChoice(card);
    case 'custom': return renderCustom(card);
    case 'map': return renderMap(card);
    default: return `<pre class="card-raw">${escapeHtml(stringify(card))}</pre>`;
  }
}

function renderHint(card) {
  return `${title(card.title)}
    <p class="card-body">${escapeHtml(card.body)}</p>`;
}

function renderEmoji(card) {
  return `<div class="card-emoji">${escapeHtml(card.emoji)}</div>
    <p class="card-caption">${escapeHtml(card.caption)}</p>`;
}

function renderAnimation(card) {
  const name = String(card.name || 'pulse');
  return `<div class="card-animation anim-${escapeAttr(name)}"><span></span><span></span><span></span></div>
    <p class="card-caption">${escapeHtml(card.caption)}</p>`;
}

function renderScore(card) {
  const side = (team) => `<div class="score-side">
      <span class="score-name">${escapeHtml(team?.name)}</span>
      <span class="score-value">${escapeHtml(team?.score)}</span>
    </div>`;
  return `${title(card.title)}
    ${card.status ? `<div class="card-status">${escapeHtml(card.status)}</div>` : ''}
    <div class="card-score">${side(card.home)}<span class="score-sep">:</span>${side(card.away)}</div>
    ${card.note ? `<p class="card-note">${escapeHtml(card.note)}</p>` : ''}`;
}

function renderWeather(card) {
  const icon = CONDITION_ICONS[String(card.condition || '').toLowerCase()] || '🌡️';
  const range = [card.high, card.low].filter((value) => value !== undefined && value !== null);
  const meta = [
    range.length === 2 ? `${escapeHtml(range[0])}° / ${escapeHtml(range[1])}°` : '',
    card.humidity ? `湿度 ${escapeHtml(card.humidity)}` : '',
    card.wind ? escapeHtml(card.wind) : '',
  ].filter(Boolean);
  return `<div class="card-weather">
      <div class="weather-icon">${icon}</div>
      <div class="weather-main">
        <div class="weather-city">${escapeHtml(card.city)}</div>
        <div class="weather-temp">${escapeHtml(card.temp)}<span>${escapeHtml(card.unit || '℃')}</span></div>
        <div class="weather-desc">${escapeHtml(card.desc)}</div>
      </div>
    </div>
    ${meta.length ? `<div class="card-meta">${meta.join(' · ')}</div>` : ''}`;
}

function renderStat(card) {
  const items = asArray(card.items).map((item) => `<div class="stat-item">
      <span class="stat-label">${escapeHtml(item?.label)}</span>
      <strong class="stat-value">${escapeHtml(item?.value)}</strong>
    </div>`).join('');
  return `${title(card.title)}<div class="card-stat">${items}</div>`;
}

function renderList(card) {
  const rows = asArray(card.rows).map((row) => `<li>
      <div class="row-main">
        <span class="row-title">${escapeHtml(row?.title)}</span>
        ${row?.subtitle ? `<span class="row-subtitle">${escapeHtml(row.subtitle)}</span>` : ''}
      </div>
      ${row?.trailing ? `<span class="row-trailing">${escapeHtml(row.trailing)}</span>` : ''}
    </li>`).join('');
  return `${title(card.title)}<ul class="card-list">${rows}</ul>`;
}

function renderChoice(card) {
  const options = asArray(card.options).map((option) => `<li class="${option?.selected ? 'selected' : ''}">
      <span class="option-label">${escapeHtml(option?.label)}</span>
      ${option?.hint ? `<span class="option-hint">${escapeHtml(option.hint)}</span>` : ''}
    </li>`).join('');
  return `${card.prompt ? `<p class="card-prompt">${escapeHtml(card.prompt)}</p>` : ''}
    <ul class="card-choice">${options}</ul>`;
}

function renderCustom(card) {
  const lines = asArray(card.lines).map((line) => `<li>${escapeHtml(line)}</li>`).join('');
  return `<div class="card-custom-head">
      ${card.emoji ? `<span class="custom-emoji">${escapeHtml(card.emoji)}</span>` : ''}
      ${title(card.title)}
    </div>
    <ul class="card-custom">${lines}</ul>`;
}

/**
 * Map cards draw a real tile map inside a sandboxed iframe, with the marker
 * list kept underneath so the raw map_search data stays inspectable.
 *
 * Two backends: AMap JS when the deployment configured a JS-API key (correct
 * labels and coverage in China), Leaflet + OpenStreetMap otherwise so the card
 * still works with no configuration at all. The tool returns WGS-84, which
 * Leaflet takes as-is and AMap needs shifted to GCJ-02 — that conversion runs
 * inside the frame, mirroring coords.py on the server.
 */
function renderMap(card) {
  const data = cleanMapData(card);
  const hasGeometry = data.markers.length > 0 || data.routes.length > 0;
  const amap = amapDisplayConfig();
  const useAmap = String(card.provider || '').toLowerCase() === 'amap' && Boolean(amap);
  const frame = hasGeometry
    ? `<iframe class="card-map-frame" title="${escapeHtml(card.title || '地图')}"
        referrerpolicy="no-referrer" loading="lazy"
        srcdoc="${escapeHtml(useAmap ? amapFrameHtml(data, amap) : leafletFrameHtml(data))}"></iframe>`
    : '<div class="card-map-empty">暂无有效坐标</div>';
  const markers = data.markers.map((marker, index) => `<li>
      <span class="marker-index">${index + 1}</span>
      <div class="marker-main">
        <span class="marker-name">${escapeHtml(marker.name)}</span>
        ${marker.address ? `<span class="marker-address">${escapeHtml(marker.address)}</span>` : ''}
        <span class="marker-coord">${formatCoord(marker.latitude)}, ${formatCoord(marker.longitude)}</span>
      </div>
      ${Number.isFinite(marker.distanceMeters) ? `<span class="marker-distance">${formatDistance(marker.distanceMeters)}</span>` : ''}
    </li>`).join('');
  const routes = asArray(card.routes).map((route) => `<li>
      <span class="route-mode">${escapeHtml(ROUTE_MODES[route?.mode] || route?.mode)}</span>
      <span class="route-metric">${formatDistance(route?.distanceMeters)} · ${formatDuration(route?.durationSeconds)}</span>
    </li>`).join('');
  const source = useAmap ? '高德地图' : card.provider === 'amap' ? '高德数据 · 高德底图' : '高德底图';
  return `${title(card.title)}
    <div class="card-meta">${escapeHtml(source)}</div>
    ${frame}
    ${routes ? `<ul class="card-routes">${routes}</ul>` : ''}
    ${markers ? `<ul class="card-map">${markers}</ul>` : ''}`;
}

/** Drop unusable points and cap the geometry an untrusted card can push. */
function cleanMapData(card) {
  const markers = asArray(card.markers).filter(isCoordinate).slice(0, MAP_MARKER_LIMIT);
  const routes = asArray(card.routes)
    .map((route) => ({
      mode: String(route?.mode || ''),
      path: asArray(route?.path).filter(isCoordinate).slice(0, MAP_PATH_LIMIT),
    }))
    .filter((route) => route.path.length >= 2)
    .slice(0, MAP_ROUTE_LIMIT);
  return { markers, routes };
}

function isCoordinate(point) {
  return Number.isFinite(point?.latitude) && Number.isFinite(point?.longitude)
    && point.latitude >= -90 && point.latitude <= 90
    && point.longitude >= -180 && point.longitude <= 180;
}

/**
 * Normalise the AMap JS credentials the defaults endpoint sent, or null.
 *
 * A bare key is rejected: AMap JS 2.0 refuses requests that carry neither the
 * security code nor a proxy host, and failing over to the Leaflet path beats a blank frame.
 */
function amapDisplayConfig() {
  const apiKey = String(mapDisplayConfig?.amap_js_api_key || '').trim();
  if (!apiKey) return null;
  const securityCode = String(mapDisplayConfig?.amap_js_security_code || '').trim();
  const serviceHost = String(mapDisplayConfig?.amap_js_service_host || '').trim();
  if (!securityCode && !serviceHost) return null;
  return { apiKey, securityCode, serviceHost };
}

/** Inline JSON for a <script> body: only `<` can break out of it. */
function inlineJson(value) {
  return JSON.stringify(value).replace(/</g, '\\u003c');
}

function leafletFrameHtml(data) {
  return `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="${LEAFLET_CSS}"><style>
html,body,#map{width:100%;height:100%;margin:0;background:#ececea}
.leaflet-control-attribution{font:9px sans-serif!important}
${MAP_PIN_CSS}</style></head><body><div id="map"></div>
<script src="${LEAFLET_JS}"></script><script>
const data=${inlineJson(data)};
${WGS84_TO_GCJ02_JS}
const ll=p=>{const g=gcj(p);return [g[1],g[0]];};
const all=[...data.markers,...data.routes.flatMap(r=>r.path)];
const first=all[0]||{latitude:39.9042,longitude:116.4074};
const map=L.map('map').setView(ll(first),13);
L.tileLayer(${inlineJson(AMAP_TILE_URL)},{subdomains:'1234',maxZoom:18,attribution:'&copy; 高德地图'}).addTo(map);
data.routes.forEach((r,i)=>L.polyline(r.path.map(ll),{color:i?'#9aa0a6':'${MAP_BRAND}',weight:i?4:6,opacity:i?.55:.9}).addTo(map));
data.markers.forEach((m,i)=>{const icon=L.divIcon({className:'',html:'<div class="pin"><span>'+(i+1)+'</span></div>',iconSize:[29,29],iconAnchor:[8,27]});
L.marker(ll(m),{icon}).addTo(map).bindPopup('<b>'+esc(m.name||'地点')+'</b>'+(m.address?'<br>'+esc(m.address):''));});
if(all.length>1)map.fitBounds(L.latLngBounds(all.map(ll)),{padding:[${MAP_PADDING},${MAP_PADDING}],maxZoom:16});
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
</script></body></html>`;
}

function amapFrameHtml(data, config) {
  const security = config.serviceHost
    ? { serviceHost: config.serviceHost }
    : { securityJsCode: config.securityCode };
  return `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body,#map{width:100%;height:100%;margin:0;background:#ececea}
${MAP_PIN_CSS}</style>
<script>window._AMapSecurityConfig=${inlineJson(security)};</script>
<script src="https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(config.apiKey)}"></script>
</head><body><div id="map"></div><script>
const data=${inlineJson(data)};
${WGS84_TO_GCJ02_JS}
const all=[...data.markers,...data.routes.flatMap(r=>r.path)];
const first=all[0]||{latitude:39.9042,longitude:116.4074};
const map=new AMap.Map('map',{center:gcj(first),zoom:13});
const overlays=[];
data.routes.forEach((r,i)=>{const line=new AMap.Polyline({path:r.path.map(gcj),strokeColor:i?'#9aa0a6':'${MAP_BRAND}',strokeWeight:i?4:6,strokeOpacity:i?.55:.9,lineJoin:'round',lineCap:'round'});map.add(line);overlays.push(line);});
data.markers.forEach((m,i)=>{const marker=new AMap.Marker({position:gcj(m),content:'<div class="pin"><span>'+(i+1)+'</span></div>',offset:new AMap.Pixel(-8,-27),title:m.name||'地点'});map.add(marker);overlays.push(marker);
const info=new AMap.InfoWindow({content:'<b>'+esc(m.name||'地点')+'</b>'+(m.address?'<br>'+esc(m.address):''),offset:new AMap.Pixel(6,-28)});
marker.on('click',()=>info.open(map,marker.getPosition()));});
if(overlays.length>1)map.setFitView(overlays,false,[${MAP_PADDING},${MAP_PADDING},${MAP_PADDING},${MAP_PADDING}],16);
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
</script></body></html>`;
}


function title(value) {
  return value ? `<h3 class="card-title">${escapeHtml(value)}</h3>` : '';
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function formatCoord(value) {
  return Number.isFinite(value) ? value.toFixed(5) : '—';
}

function formatDistance(meters) {
  if (!Number.isFinite(meters)) return '—';
  return meters >= 1000 ? `${(meters / 1000).toFixed(1)} km` : `${Math.round(meters)} m`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} 分钟`;
  return `${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分`;
}

function stringify(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (err) {
    return String(value);
  }
}

export function escapeHtml(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/\s+/g, '-');
}
