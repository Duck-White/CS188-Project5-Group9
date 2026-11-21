'use strict';
const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (num, digits = 3) => Number(num).toFixed(digits);
let backend = false;

async function api(path, data = undefined) {
  let response;
  try {
    response = await fetch(`api/${path}`, {
      method: data === undefined ? 'GET' : 'POST',
      headers: data === undefined ? {} : { 'Content-Type': 'application/json' },
      body: data === undefined ? undefined : JSON.stringify(data)
    });
  } catch {
    throw new Error('Không kết nối được Python backend. Chạy: python -m uvicorn webapp.server:app --port 8000');
  }
  let json;
  try { json = await response.json(); } catch {
    throw new Error('Website đang ở chế độ tĩnh. Cần chạy FastAPI để thực nghiệm với PyTorch.');
  }
  if (!response.ok) {
    let detail = json.detail;
    if (typeof detail !== 'string') detail = 'Dữ liệu không hợp lệ hoặc backend không thể xử lý.';
    throw new Error(detail);
  }
  return json;
}

function result(id, text, kind = 'success') {
  const box = $(id);
  box.textContent = text;
  box.className = `result ${kind}`;
}
async function run(buttonId, callback, onError) {
  const button = $(buttonId);
  if (!button) return;
  button.disabled = true;
  try { await callback(); }
  catch (e) { onError(e.message || String(e)); }
  finally { button.disabled = false; }
}
async function loadStatus() {
  try {
    const status = await api('status');
    backend = Boolean(status.backend);
    $('connection-dot').classList.add('online');
    $('connection-text').textContent = 'PyTorch backend đã kết nối';
    for (const question of ['q3', 'q4', 'q5']) {
      $(question + '-status').textContent = status[question] ? 'MODEL SẴN SÀNG' : 'CẦN CHECKPOINT';
      $(question + '-status').classList.toggle('unavailable', !status[question]);
    }
  } catch {
    $('connection-text').textContent = 'Chế độ xem tĩnh · chưa có API';
    for (const question of ['q3', 'q4', 'q5']) $(question + '-status').classList.add('unavailable');
  }
}

function plotFrame(canvasId, transform, gridY = true) {
  const canvas = $(canvasId), ctx = canvas.getContext('2d');
  const { width: W, height: H } = canvas;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#fbfcff'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#e9edf6'; ctx.lineWidth=1;
  for(let i=0;i<=10;i++) { const x=i*W/10;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke(); }
  if (gridY) for(let i=0;i<=6;i++) { const y=i*H/6;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke(); }
  ctx.strokeStyle='#aebad0';ctx.lineWidth=1.5;
  ctx.beginPath();ctx.moveTo(0,transform.y(0));ctx.lineTo(W,transform.y(0));ctx.moveTo(transform.x(0),0);ctx.lineTo(transform.x(0),H);ctx.stroke();
  return {ctx,W,H};
}
function perceptronPlot(data) {
  const canvas=$('q1-plot');
  const padding=25,W=canvas.width,H=canvas.height;
  const T={x:x=>padding+(x+3.3)/6.6*(W-2*padding),y:y=>H-padding-(y+3.3)/6.6*(H-2*padding)};
  const {ctx}=plotFrame('q1-plot',T);
  ctx.lineWidth=3;
  function drawBoundary(weights,color,dash) {
    if(!weights || Math.abs(weights[1])<1e-5) return;
    ctx.beginPath();ctx.strokeStyle=color;ctx.setLineDash(dash);
    for(let j=0;j<=90;j++){
      const x=-3.3+6.6*j/90, y=-(weights[0]*x+weights[2])/weights[1];
      if(j===0)ctx.moveTo(T.x(x),T.y(y));else ctx.lineTo(T.x(x),T.y(y));
    }ctx.stroke();ctx.setLineDash([]);
  }
  drawBoundary(data.initial_weights,'#f1a46e',[8,5]);
  drawBoundary(data.trained_weights,'#6458de',[]);
  for(const p of data.samples) {
    ctx.beginPath();ctx.arc(T.x(p.x),T.y(p.y),7,0,Math.PI*2);
    ctx.fillStyle=p.label===1?'#6a62e6':'#1eb9b1';ctx.fill();
    ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
  }
  ctx.font='bold 12px DM Sans, sans-serif';ctx.fillStyle='#5f6f87';
  ctx.fillText('Ban đầu  - - - ',20,22);ctx.fillText('Sau train ━━━ ',160,22);
}
function emptyPerceptron() {
  perceptronPlot({initial_weights:[1,1,1],trained_weights:[1,1,1],samples:[{x:-2,y:-1,label:-1},{x:2,y:1,label:1},{x:-1,y:2,label:1},{x:1,y:-2,label:-1}]});
}
$('run-q1').addEventListener('click',()=>run('run-q1',async()=>{
  $('q1-state').textContent='Đang huấn luyện với PyTorch…';
  const d=await api('q1',{});
  perceptronPlot(d);
  $('q1-state').textContent='Đã huấn luyện mô hình thật';
  result('q1-result',`Accuracy demo: ${(d.accuracy*100).toFixed(0)}% · w=[${d.trained_weights.map(v=>fmt(v,2)).join(', ')}] · ${d.training}`);
},err=>{$('q1-state').textContent='Không thể huấn luyện';result('q1-result',err,'error');}));

function sinPlot(data=null){
 const canvas=$('q2-plot'),W=canvas.width,H=canvas.height,pad=30;
 const T={x:x=>pad+(x+2*Math.PI)/(4*Math.PI)*(W-2*pad),y:y=>H/2-y*(H/2-32)};
 const {ctx}=plotFrame('q2-plot',T);
 function stroke(points,color,thickness){ctx.beginPath();ctx.lineWidth=thickness;ctx.strokeStyle=color;ctx.setLineDash([]);points.forEach((v,i)=>{let x=T.x(-2*Math.PI+i/(points.length-1)*4*Math.PI),y=T.y(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}
 stroke(Array.from({length:121},(_,i)=>Math.sin(-2*Math.PI+i/120*4*Math.PI)),'#2bb8ad',3);
 if(data)stroke(data.prediction,'#6255de',3);
 ctx.fillStyle='#788aa2';ctx.font='11px monospace';ctx.fillText('-2π',17,H-12);ctx.fillText('2π',W-38,H-12);
}
$('run-q2').addEventListener('click',()=>run('run-q2',async()=>{
 $('q2-state').textContent='Đang train MLP với train_regression…';
 const d=await api('q2/train',{});
 const curve=await api('q2/curve');
 sinPlot(curve);
 $('q2-state').textContent='Đã huấn luyện trên 200 điểm mẫu';
 result('q2-result',`MSE demo = ${fmt(d.mse,6)}. ${d.note}`);
},err=>{$('q2-state').textContent='Chưa chạy được demo';result('q2-result',err,'error');}));

function initDigitPad(q) {
 const canvas=$('digit-'+q),ctx=canvas.getContext('2d');
 ctx.lineCap='round';ctx.lineJoin='round';
 function clear(){ctx.fillStyle='#08172c';ctx.fillRect(0,0,canvas.width,canvas.height)}
 clear();let drawing=false,last=null;
 const xy=e=>{let r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)/r.width*canvas.width,y:(e.clientY-r.top)/r.height*canvas.height}};
 const paint=(from,to)=>{ctx.strokeStyle='#fff';ctx.lineWidth=26;ctx.beginPath();ctx.moveTo(from.x,from.y);ctx.lineTo(to.x,to.y);ctx.stroke()};
 canvas.addEventListener('pointerdown',e=>{drawing=true;last=xy(e);canvas.setPointerCapture(e.pointerId);paint(last,last)});
 canvas.addEventListener('pointermove',e=>{if(!drawing)return;const now=xy(e);paint(last,now);last=now});
 for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,()=>{drawing=false;last=null});
 document.querySelector(`[data-clear="${q}"]`).addEventListener('click',clear);
 function pixels(){const off=document.createElement('canvas');off.width=28;off.height=28;const c=off.getContext('2d');c.drawImage(canvas,0,0,28,28);const d=c.getImageData(0,0,28,28).data;const out=[];for(let i=0;i<784;i++){let v=(d[i*4]+d[i*4+1]+d[i*4+2])/3;out.push(clamp((v-11)/244,0,1))}return out}
 return pixels;
}
const pixelsQ3=initDigitPad('q3'),pixelsQ5=initDigitPad('q5');
function renderProbabilities(containerId, labels, probabilities){
 const container=$(containerId);container.replaceChildren();
 const indices=labels.map((_,i)=>i).sort((a,b)=>probabilities[b]-probabilities[a]);
 for(const i of indices){const item=document.createElement('div');item.className='prob-item';const name=document.createElement('span');name.textContent=labels[i];const track=document.createElement('div');track.className='prob-track';const fill=document.createElement('div');fill.className='prob-fill';fill.style.width=`${clamp(probabilities[i]*100,0,100).toFixed(1)}%`;track.append(fill);const pct=document.createElement('span');pct.textContent=`${(probabilities[i]*100).toFixed(1)}%`;item.append(name,track,pct);container.append(item)}
}
$('run-q3').addEventListener('click',()=>run('run-q3',async()=>{
 const d=await api('q3',{pixels:pixelsQ3()});result('q3-result',`Dự đoán Q3: ${d.label} · Xác suất cao nhất ${(Math.max(...d.probabilities)*100).toFixed(1)}%. ${d.source}`);
},err=>result('q3-result',err,'error')));
$('run-q4').addEventListener('click',()=>run('run-q4',async()=>{
 const word=$('q4-word').value.trim();if(!word)throw new Error('Hãy nhập một từ.');
 const d=await api('q4',{word});renderProbabilities('q4-bars',d.languages,d.probabilities);result('q4-result',`Dự đoán Q4: ${d.language}. ${d.source}`);
},err=>result('q4-result',err,'error')));
$('q4-word').addEventListener('keydown',e=>{if(e.key==='Enter'){$('run-q4').click()}});

function parseMatrix(text){const rows=text.trim().split(/\n+/).map(row=>row.trim().split(/[\s,;]+/).filter(Boolean).map(Number));if(!rows.length||!rows[0].length||rows.some(row=>row.length!==rows[0].length||row.some(v=>!Number.isFinite(v))))throw new Error('Ma trận cần có cùng số cột trên mọi dòng; chỉ nhập số.');return rows;}
$('run-conv').addEventListener('click',()=>run('run-conv',async()=>{
 const image=parseMatrix($('conv-image').value),kernel=parseMatrix($('conv-kernel').value);
 const d=await api('q5/convolve',{image,kernel});$('conv-result').textContent=`Output: ${d.shape.join(' × ')}\n${d.output.map(row=>row.map(v=>String(fmt(v,2)).padStart(8,' ')).join(' ')).join('\n')}\n${d.source}`;
},err=>{$('conv-result').textContent=err;}));
$('run-q5').addEventListener('click',()=>run('run-q5',async()=>{
 const d=await api('q5/predict',{pixels:pixelsQ5()});result('q5-result',`Dự đoán CNN Q5: ${d.label} · Xác suất cao nhất ${(Math.max(...d.probabilities)*100).toFixed(1)}%. ${d.source}`);
},err=>result('q5-result',err,'error')));

function attentionGrid(matrix){const grid=$('attention-grid');grid.style.gridTemplateColumns=`repeat(${matrix.length}, minmax(0, 1fr))`;grid.replaceChildren();for(let i=0;i<matrix.length;i++){for(let j=0;j<matrix.length;j++){const w=matrix[i][j],box=document.createElement('div');box.className='attn-cell'+(j>i?' off':'');box.style.background=j>i?'#ebeff6':`rgba(98, 80, 219, ${clamp(.13+w*.85,0,1).toFixed(2)})`;box.style.color=w>.56?'#fff':'#34466f';box.textContent=j>i?'×':fmt(w,2);box.setAttribute('title',`token ${i+1} nhìn token ${j+1}: ${fmt(w,4)}`);grid.append(box)}}}
$('run-q6').addEventListener('click',()=>run('run-q6',async()=>{
 const tokens=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
 const d=await api('q6',{tokens});attentionGrid(d.attention);
 const maxFuture=Math.max(...d.attention.flatMap((row,i)=>row.slice(i+1)));
 result('q6-result',`Attention.forward trả về ${d.output.length} × ${d.output[0].length} phần tử. Trọng số nhìn tương lai tối đa: ${fmt(maxFuture,6)} (mục tiêu = 0). ${d.source}`);
},err=>result('q6-result',err,'error')));

function setNav(){let items=document.querySelectorAll('[data-nav]');const observer=new IntersectionObserver(entries=>{for(let e of entries){if(e.isIntersecting){items.forEach(a=>a.classList.toggle('selected',a.dataset.nav===e.target.id))}}},{rootMargin:'-15% 0px -72% 0px'});['overview','q1','q2','q3','q4','q5','q6'].forEach(id=>observer.observe($(id)))}
emptyPerceptron();sinPlot();setNav();loadStatus();