const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('static/turnbased.html', 'utf8');
function section(start, end) { return html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start))); }
class Socket {
  static OPEN = 1;
  static instances = [];
  constructor() { this.readyState = 1; this.sent = []; Socket.instances.push(this); }
  send(text) { this.sent.push(JSON.parse(text)); }
  close() { this.readyState = 3; this.onclose?.(); }
  event(message) { this.onmessage?.({data: JSON.stringify(message)}); }
}
const ctx = { state: {activeChat: null}, WebSocket: Socket,
  location: {protocol: 'https:'}, window: {location: {host: 'test'}} };
vm.createContext(ctx);
vm.runInContext(section('function _openRealtimeChatWS(', '// ============ Chat 模式'), ctx);
let oldCalls = [], newCalls = [];
const handlers = list => ({onDelta: x => list.push(['delta',x]), onDone: x => list.push(['done',x]), onCancel: () => list.push(['cancel']), onClosed: () => list.push(['closed']), onError: x => list.push(['error',x])});
const first = ctx._openRealtimeChatWS({messages:['old']}, handlers(oldCalls));
const old = Socket.instances.at(-1);
old.event({type:'session.queue_done'});old.event({type:'session.created'});
old.event({type:'response.output.delta',kind:'text',text:'partial'});
first.cancel();first.cancel();
assert.equal(old.sent.filter(x=>x.type==='session.close').length,1);
assert.equal(old.sent.at(-1).reason,'client_cancel');
const second = ctx._openRealtimeChatWS({messages:['new']}, handlers(newCalls));
const current = Socket.instances.at(-1);
old.event({type:'response.output.delta',text:'stale'});
old.event({type:'response.done',text:'stale done'});
old.event({type:'session.closed',reason:'client_cancel'});
assert.deepEqual(oldCalls.map(x=>x[0]),['delta','cancel']);
assert.equal(ctx.state.activeChat,second);
assert.deepEqual(newCalls,[]);
current.event({type:'session.queue_done'});current.event({type:'session.created'});
current.event({type:'response.done',text:'new answer'});
current.event({type:'session.closed',reason:'turn_done'});
assert.deepEqual(newCalls.map(x=>x[0]),['done']);
assert.equal(current.sent.at(-1).type,'session.close');
assert.equal(ctx.state.activeChat,null);

// A queued socket has no backend session yet; cancel its transport only.
let queuedCalls = [];
const queuedHandle = ctx._openRealtimeChatWS({}, handlers(queuedCalls));
const queued = Socket.instances.at(-1);
queuedHandle.cancel();
assert.equal(queued.readyState,3);
assert.deepEqual(queued.sent,[]);
assert.deepEqual(queuedCalls.map(x=>x[0]),['cancel']);

// Execute the actual streaming and cancellation callbacks with a minimal page.
let displayed='', meta=[], paused=0, marked=0;
ctx.state={isGenerating:false,messages:[],requestId:'req'};
ctx.setGenerating=v=>ctx.state.isGenerating=v;
ctx.updateCancelStopButton=()=>{};ctx.addMessageUI=()=>{};
ctx.updateLastAssistantMessage=t=>displayed=t;ctx.updateLastAssistantMeta=x=>meta.push(x);
ctx.queueCountdownStop=()=>{};ctx._buildChatWSBody=()=>({body:{messages:[]},enableTts:true});
ctx.appendStreamingAudioControl=()=>{};
const last={dataset:{},remove(){this.removed=true},querySelector(){return {textContent:displayed,classList:{contains:()=>false}}}};
ctx.document={querySelectorAll:()=>[last]};
ctx.StreamingAudioPlayer=class {pushChunk(){} play(){} pause(){paused++} markStreamFinished(){marked++}};
vm.runInContext(section('function _cleanupLastAssistantMessage(', 'function setGenerating('),ctx);
vm.runInContext(section('function cancelGeneration()', 'function _cleanupLastAssistantMessage('),ctx);
vm.runInContext(section('async function sendStreamingMessage(', '// ============ 启动'),ctx);
ctx.sendStreamingMessage(256);
const stream=Socket.instances.at(-1);
stream.event({type:'session.queue_done'});stream.event({type:'session.created'});
stream.event({type:'response.output.delta',kind:'text',text:'收到一半'});
stream.event({type:'response.output.delta',kind:'audio',audio:'PCM'});
ctx.cancelGeneration();
assert.equal(paused,1);assert.equal(marked,1);
assert.equal(ctx.state.messages.length,1);assert.equal(ctx.state.messages[0].content,'收到一半');
assert.equal(last.dataset.msgIndex,0);assert.equal(ctx.state.isGenerating,false);
stream.event({type:'response.done',text:'不能覆盖'});
assert.equal(ctx.state.messages.length,1);assert.equal(displayed,'收到一半');
console.log('desktop simplex cancellation: session.close, stale replies, partial history, audio stop PASS');
