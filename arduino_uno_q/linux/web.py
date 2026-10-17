import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>QML ECG monitor</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#f7f7f5;color:#222}
.row{display:flex;gap:12px;flex-wrap:wrap}.card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px;flex:1;min-width:200px}
.big{font-size:22px;font-weight:600}.NORMAL{color:#1d7f4e}.ABNORMAL{color:#b42318}.POOR_SIGNAL{color:#b54708}.WARMUP{color:#666}
canvas{width:100%;background:#fff;border:1px solid #ddd;border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px}td{border-bottom:1px solid #eee;padding:3px 6px}
</style></head><body>
<div class="row">
<div class="card">Status<div id="status" class="big">-</div><div id="reasons"></div></div>
<div class="card">Abnormal probability<div id="p" class="big">-</div><div id="thr"></div></div>
<div class="card">Heart rate<div id="hr" class="big">-</div><div id="beats"></div></div>
<div class="card">Pipeline<div id="lat"></div><div id="link"></div></div>
</div>
<h3>ECG, last 10 s</h3><canvas id="ecg" width="1000" height="180"></canvas>
<h3>Probability history</h3><canvas id="hist" width="1000" height="140"></canvas>
<div class="row">
<div class="card"><h3>Patient</h3>
<label>Age <input id="age" type="number" min="1" max="120"></label>
<label>Sex <select id="sex"><option value="0">Male</option><option value="1">Female</option></select></label>
<button onclick="save()">Save</button> <span id="msg"></span></div>
<div class="card"><h3>Model inputs</h3><table id="feats"></table></div>
</div>
<script>
let loaded=false;
function line(c,ys,lo,hi,color){const g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);if(ys.length<2)return;
g.strokeStyle=color;g.lineWidth=1.5;g.beginPath();const dx=c.width/(ys.length-1),r=(hi-lo)||1;
ys.forEach((y,i)=>{const px=i*dx,py=c.height-(y-lo)/r*c.height;i?g.lineTo(px,py):g.moveTo(px,py)});g.stroke();}
async function tick(){try{const s=await (await fetch('/api/state')).json();
const st=document.getElementById('status');st.textContent=s.status;st.className='big '+s.status;
document.getElementById('reasons').textContent=s.reasons.join(', ');
document.getElementById('p').textContent=s.p==null?'-':s.p.toFixed(3);
document.getElementById('thr').textContent='threshold '+s.threshold.toFixed(3);
document.getElementById('hr').textContent=s.hr==null?'-':s.hr.toFixed(0)+' bpm';
document.getElementById('beats').textContent=s.n_beats+' beats';
document.getElementById('lat').textContent='latency '+(s.latency_ms==null?'-':s.latency_ms.toFixed(1)+' ms');
document.getElementById('link').textContent='chunks '+s.received+', dropped '+s.dropped;
const w=s.waveform;if(w.length)line(document.getElementById('ecg'),w,Math.min(...w),Math.max(...w),'#1f5fbf');
line(document.getElementById('hist'),s.p_history,0,1,'#b42318');
if(!loaded){document.getElementById('age').value=s.age;document.getElementById('sex').value=s.sex;loaded=true;}
document.getElementById('feats').innerHTML=Object.entries(s.features).map(([k,v])=>'<tr><td>'+k+'</td><td>'+Number(v).toFixed(3)+'</td></tr>').join('');
}catch(e){}}
async function save(){const age=parseInt(document.getElementById('age').value),sex=parseInt(document.getElementById('sex').value);
const m=document.getElementById('msg');if(!(age>=1&&age<=120)){m.textContent='Enter an age between 1 and 120';return;}
const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({age,sex})});
m.textContent=r.ok?'Saved':'Error';}
setInterval(tick,1000);tick();
</script></body></html>"""

def start_dashboard(state, port: int, on_settings):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, json.dumps(state.snapshot()).encode("utf-8"), "application/json")
            else:
                self._send(404, b"not found", "text/plain")
        
        def do_POST(self):
            if self.path != "/api/settings":
                self._send(404, b"not found", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                age, sex = int(body["age"]), int(body["sex"])
                if not (1 <= age <= 120 and sex in (0, 1)):
                    raise ValueError
                on_settings(age, sex)
                self._send(200, b'{"ok":true}', "application/json")
            except Exception:
                self._send(400, b'{"ok":false}', "application/json")
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server