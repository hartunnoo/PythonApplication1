"""
Web dashboard — live security event monitor.

Serves a browser-based UI on http://localhost:5000 (configurable).
Reads directly from logs/events.jsonl — no database needed.

Routes:
  GET /                      — dashboard HTML
  GET /api/events            — last N events as JSON
  GET /api/stats             — per-person counts, today summary
  GET /api/status            — system uptime + camera count
  GET /screenshots/<file>    — serve screenshot images
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, date
from typing import Optional

from src.config import DashboardConfig
from src.logger_setup import get_logger

log = get_logger(__name__)

# ── Embedded HTML (no template files needed) ─────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>His Majesty Office &mdash; Security Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0d0d1a;color:#d0d0e8;min-height:100vh}
header{background:#13132a;border-bottom:2px solid #1e1e40;padding:12px 28px;display:flex;align-items:flex-start;justify-content:space-between;gap:16px}
.header-titles h1{font-size:17px;font-weight:700;color:#ccccff;letter-spacing:.4px}
.header-sub{font-size:13px;color:#8888cc;margin-top:4px;display:flex;align-items:center;gap:7px}
.header-author{font-size:10.5px;color:#3a3a5a;margin-top:5px;font-style:italic;line-height:1.5}
#status-dot{width:9px;height:9px;border-radius:50%;background:#22dd66;display:inline-block;animation:pulse 2s infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.uptime{font-size:12px;color:#5555aa;white-space:nowrap;padding-top:4px;flex-shrink:0}
/* Stats bar */
.stats-bar{display:flex;flex-wrap:wrap;gap:10px;padding:13px 28px;background:#10101f;border-bottom:1px solid #1a1a30}
.stat-card{background:#16162a;border:1px solid #1e1e3a;border-radius:8px;padding:10px 18px;min-width:120px;text-align:center}
.stat-card .val{font-size:26px;font-weight:700;line-height:1}
.stat-card .lbl{font-size:10.5px;color:#6666aa;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}
.val.red{color:#ff3355}.val.green{color:#22dd66}.val.gray{color:#8888cc}.val.orange{color:#ffaa33}.val.white{color:#d0d0e8}
/* Main */
main{padding:18px 28px}
.toolbar{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.view-btn{background:#16162a;border:1px solid #252545;color:#6666aa;padding:6px 16px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;letter-spacing:.3px;transition:all .15s}
.view-btn:hover{border-color:#4444aa;color:#aaaaee}
.view-btn.active{background:#22224a;border-color:#5555bb;color:#ccccff}
.refresh-info{font-size:11px;color:#333366;margin-left:auto}
/* Table */
table{width:100%;border-collapse:collapse;font-size:12.5px}
thead tr{background:#13132a;color:#5566aa;text-transform:uppercase;font-size:10.5px;letter-spacing:.4px}
th{padding:8px 10px;text-align:left;font-weight:600;white-space:nowrap}
tbody tr{border-bottom:1px solid #181828;transition:background .12s}
tbody tr:hover{background:#14142a}
td{padding:8px 10px;vertical-align:middle}
.thumb{width:52px;height:38px;object-fit:cover;border-radius:4px;border:1px solid #222240;cursor:pointer;transition:transform .15s}
.thumb:hover{transform:scale(1.08)}
.conf-bar-wrap{width:68px;background:#1a1a30;border-radius:3px;height:4px;display:inline-block;vertical-align:middle;margin-left:5px}
.conf-bar{height:4px;border-radius:3px}
/* Badges */
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10.5px;font-weight:700;letter-spacing:.4px;text-transform:uppercase}
.badge.blacklist{background:#4a0010;color:#ff3355;border:1px solid #7a0020}
.badge.whitelist{background:#0a3a1a;color:#22dd66;border:1px solid #0f5a28}
.badge.unknown{background:#222240;color:#8888cc;border:1px solid #333366}
.badge.ic_scan{background:#332200;color:#ffaa33;border:1px solid #554400}
.conf-bar.red,.conf-bar.blacklist{background:#ff3355}
.conf-bar.green,.conf-bar.whitelist{background:#22dd66}
.conf-bar.gray{background:#5555aa}.conf-bar.orange{background:#ffaa33}
/* Gallery */
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}
.card{background:#16162a;border:1px solid #1e1e3a;border-radius:12px;overflow:hidden;transition:transform .18s,box-shadow .18s}
.card:hover{transform:translateY(-3px);box-shadow:0 8px 28px #000b}
.card.blacklist{border-top:3px solid #ff3355}
.card.whitelist{border-top:3px solid #22dd66}
.card.unknown{border-top:3px solid #5555aa}
.card.ic_scan{border-top:3px solid #ffaa33}
.card-photo{width:100%;height:160px;object-fit:cover;display:block;cursor:pointer;transition:opacity .15s}
.card-photo:hover{opacity:.88}
.card-no-photo{width:100%;height:160px;background:#111122;display:flex;align-items:center;justify-content:center;font-size:46px;color:#1e1e3a;border-bottom:1px solid #1a1a30}
.card-body{padding:11px 13px 13px}
.card-name{font-size:13px;font-weight:700;color:#e0e0ff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:6px}
.card-badge-row{margin-bottom:7px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.card-conf-label{font-size:10.5px;color:#8888bb;margin-top:7px;margin-bottom:2px}
.card-conf-track{width:100%;height:4px;background:#1a1a30;border-radius:3px}
.card-conf-fill{height:4px;border-radius:3px;transition:width .4s}
.card-info{margin-top:7px;display:flex;flex-direction:column;gap:3px}
.card-info-row{font-size:10.5px;color:#4a5588;display:flex;align-items:center;gap:5px}
.card-info-row span:last-child{color:#9090bb}
/* IC card specific */
.card-ic-section{margin-top:8px;padding-top:7px;border-top:1px solid #1c1c2e;display:flex;flex-direction:column;gap:3px}
.card-ic-row{display:flex;align-items:baseline;gap:7px}
.ic-lbl{color:#5566aa;font-size:10px;text-transform:uppercase;letter-spacing:.3px;min-width:38px;flex-shrink:0}
.ic-val{color:#ccccee;font-size:11.5px;font-weight:600}
.ic-ok{color:#22aa55;font-size:9.5px;margin-left:3px}
/* Table scroll on narrow screens */
#view-table{overflow-x:auto}
/* Empty */
.no-data{text-align:center;padding:52px;color:#333355;font-size:14px}
/* Modal */
#modal{display:none;position:fixed;inset:0;background:#000d;z-index:200;align-items:center;justify-content:center}
#modal.open{display:flex}
#modal img{max-width:92vw;max-height:92vh;border:2px solid #2a2a50;border-radius:8px;box-shadow:0 6px 40px #000c}
#modal-close{position:fixed;top:18px;right:24px;font-size:26px;cursor:pointer;color:#fff;background:#0008;border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center}
</style>
</head>
<body>
<header>
  <div class="header-titles">
    <h1>His Majesty Office, Istana Nurul Iman</h1>
    <div class="header-sub"><span id="status-dot"></span>Security Dashboard &mdash; Face Recognition System</div>
    <div class="header-author">
      <em>Web app by: Ahmmad Hartunnoo Suharddy Bin Mohd Soud &nbsp;&middot;&nbsp; Pegawai Kerja &nbsp;&middot;&nbsp; Ketua Server and Security System</em>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:16px">
    <a href="/person" style="color:#8888cc;text-decoration:none;font-size:12px;font-weight:600;border:1px solid #333366;padding:5px 14px;border-radius:6px;white-space:nowrap;transition:all .15s" onmouseover="this.style.borderColor='#5555bb';this.style.color='#ccccff'" onmouseout="this.style.borderColor='#333366';this.style.color='#8888cc'">Person Audit &rarr;</a>
    <span class="uptime" id="uptime-label">Loading&hellip;</span>
  </div>
</header>

<div class="stats-bar">
  <div class="stat-card"><div class="val red"    id="s-blacklist">0</div><div class="lbl">Blacklist Today</div></div>
  <div class="stat-card"><div class="val green"  id="s-whitelist">0</div><div class="lbl">Whitelist Today</div></div>
  <div class="stat-card"><div class="val gray"   id="s-unknown">0</div><div class="lbl">Unknown Today</div></div>
  <div class="stat-card"><div class="val orange" id="s-icscan">0</div><div class="lbl">IC Scans Today</div></div>
  <div class="stat-card"><div class="val white"  id="s-cameras">&mdash;</div><div class="lbl">Cameras</div></div>
</div>

<main>
  <div class="toolbar">
    <button class="view-btn active" id="btn-table"   onclick="setView('table')">&#9776;&nbsp; Table</button>
    <button class="view-btn"        id="btn-gallery" onclick="setView('gallery')">&#9638;&nbsp; Gallery</button>
    <span class="refresh-info">Auto-refresh 3 s &mdash; <span id="last-refresh">&mdash;</span></span>
  </div>

  <div id="view-table">
    <table>
      <thead><tr>
        <th>Time</th><th>Camera</th><th>Name</th><th>Status</th>
        <th>Confidence</th><th>ID / Info</th><th>DOB &middot; Sex</th><th>Screenshot</th>
      </tr></thead>
      <tbody id="events-body">
        <tr><td colspan="8" class="no-data">Loading events&hellip;</td></tr>
      </tbody>
    </table>
  </div>

  <div id="view-gallery" style="display:none">
    <div class="card-grid" id="gallery-grid">
      <div class="no-data" style="grid-column:1/-1">Loading events&hellip;</div>
    </div>
  </div>
</main>

<div id="modal" onclick="closeModal()">
  <div id="modal-close" onclick="closeModal()">&#x2715;</div>
  <img id="modal-img" src="" alt="">
</div>

<script>
var _view=(new URLSearchParams(location.search).get('view'))||'table';
var _events=[];

function closeModal(){document.getElementById('modal').classList.remove('open')}
function openModal(src){document.getElementById('modal-img').src=src;document.getElementById('modal').classList.add('open')}

function setView(v){
  _view=v;
  history.replaceState(null,'',v==='gallery'?'?view=gallery':'?');
  document.getElementById('view-table').style.display=v==='table'?'':'none';
  document.getElementById('view-gallery').style.display=v==='gallery'?'':'none';
  document.getElementById('btn-table').classList.toggle('active',v==='table');
  document.getElementById('btn-gallery').classList.toggle('active',v==='gallery');
  renderCurrent();
}

function pad(n){return String(n).padStart(2,'0')}
function fmtTime(iso){var d=new Date(iso);return pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds())}
function fmtDate(iso){var d=new Date(iso);return d.toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'})}
function ssToUrl(ss){return ss?('/'+ss.replace(/\\\\/g,'/')):'';}
function barCls(lt){return lt==='blacklist'?'red':lt==='whitelist'?'green':lt==='ic_scan'?'orange':'gray'}
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function sexIcon(s){return s==='M'?'&#9794;':s==='F'?'&#9792;':''}

function renderTable(events){
  var tbody=document.getElementById('events-body');
  if(!events||!events.length){tbody.innerHTML='<tr><td colspan="8" class="no-data">No events recorded yet.</td></tr>';return;}
  tbody.innerHTML=events.map(function(e){
    var lt=e.list_type||'unknown';
    var isIC=lt==='ic_scan';
    var conf=parseFloat(e.confidence||0);
    var bc=barCls(lt);
    var barW=Math.round(conf*68);
    var ssUrl=ssToUrl(e.screenshot||'');
    var thumb=ssUrl?('<img class="thumb" loading="lazy" src="'+ssUrl+'" onclick="openModal(this.src)">'):'&mdash;';
    var idInfo,dobSex;
    if(isIC){
      var doc=e.ic_doc_number||'';
      var nat=e.ic_nationality||e.ic_country||'';
      idInfo=(doc?'<b>'+esc(doc)+'</b>':'N/A')+(nat?' <small style="color:#556688;font-size:10px">'+esc(nat)+'</small>':'');
      dobSex=(e.ic_dob?esc(e.ic_dob):'&mdash;')+(e.ic_sex?' '+sexIcon(e.ic_sex):'');
    } else {
      idInfo=esc(e.emotion||'')||'&mdash;';
      dobSex=e.age!=null?'~'+e.age+' yrs':'&mdash;';
    }
    return '<tr>'
      +'<td title="'+esc(fmtDate(e.timestamp))+'">'+fmtTime(e.timestamp)+'</td>'
      +'<td>'+esc(e.camera||'')+'</td>'
      +'<td><b>'+esc(e.name||'&mdash;')+'</b></td>'
      +'<td><span class="badge '+lt+'">'+lt.replace('_',' ')+'</span></td>'
      +'<td>'+(conf*100).toFixed(0)+'%<span class="conf-bar-wrap"><span class="conf-bar '+bc+'" style="width:'+barW+'px"></span></span></td>'
      +'<td>'+idInfo+'</td>'
      +'<td>'+dobSex+'</td>'
      +'<td>'+thumb+'</td>'
      +'</tr>';
  }).join('');
}

function renderGallery(events){
  var grid=document.getElementById('gallery-grid');
  if(!events||!events.length){grid.innerHTML='<div class="no-data" style="grid-column:1/-1">No events recorded yet.</div>';return;}
  grid.innerHTML=events.map(function(e){
    var lt=e.list_type||'unknown';
    var isIC=lt==='ic_scan';
    var conf=parseFloat(e.confidence||0);
    var bc=barCls(lt);
    var pct=Math.round(conf*100);
    var ssUrl=ssToUrl(e.screenshot||'');
    var photoHtml=ssUrl
      ?('<img class="card-photo" loading="lazy" src="'+ssUrl+'" onclick="openModal(this.src)" alt="'+esc(e.name||'')+'">')
      :'<div class="card-no-photo">'+(isIC?'&#128196;':'&#128100;')+'</div>';

    var bodyHtml;
    if(isIC){
      var nat=e.ic_nationality||e.ic_country||'';
      var mrz=e.ic_mrz_format?'MRZ '+e.ic_mrz_format:'';
      var okBadge=e.ic_checks_ok?'<span class="ic-ok">&#10003;</span>':'';
      bodyHtml=''
        +'<div class="card-name" title="'+esc(e.name||'')+'">'+esc(e.name||'Unknown')+'</div>'
        +'<div class="card-badge-row">'
          +'<span class="badge ic_scan">IC Scan</span>'
          +(mrz?'<small style="color:#446688;font-size:10px">'+esc(mrz)+'</small>':'')
        +'</div>'
        +'<div class="card-ic-section">'
          +(e.ic_doc_number?'<div class="card-ic-row"><span class="ic-lbl">Doc</span><span class="ic-val">'+esc(e.ic_doc_number)+'</span>'+okBadge+'</div>':'')
          +(e.ic_dob?'<div class="card-ic-row"><span class="ic-lbl">DOB</span><span class="ic-val">'+esc(e.ic_dob)+'</span></div>':'')
          +(e.ic_sex||nat?'<div class="card-ic-row"><span class="ic-lbl">Sex/Nat</span><span class="ic-val">'
            +(e.ic_sex==='M'?'&#9794; Male':e.ic_sex==='F'?'&#9792; Female':e.ic_sex||'')
            +(e.ic_sex&&nat?' &middot; ':'')+(nat?esc(nat):'')+'</span></div>':'')
          +(e.ic_expiry?'<div class="card-ic-row"><span class="ic-lbl">Expiry</span><span class="ic-val">'+esc(e.ic_expiry)+'</span></div>':'')
          +(e.ic_pob?'<div class="card-ic-row"><span class="ic-lbl">POB</span><span class="ic-val">'+esc(e.ic_pob)+'</span></div>':'')
        +'</div>'
        +'<div class="card-info">'
          +'<div class="card-info-row"><span>&#128247;</span><span>'+esc(e.camera||'&mdash;')+'</span></div>'
          +'<div class="card-info-row"><span>&#128336;</span><span>'+fmtTime(e.timestamp)+' &middot; '+fmtDate(e.timestamp)+'</span></div>'
        +'</div>';
    } else {
      var emotionRow=(e.emotion||e.age!=null)
        ?('<div class="card-info-row"><span>&#128512;</span><span>'+esc(e.emotion||'')+(e.age!=null?' &middot; ~'+e.age+' yrs':'')+'</span></div>'):'';
      bodyHtml=''
        +'<div class="card-name" title="'+esc(e.name||'')+'">'+esc(e.name||'Unknown')+'</div>'
        +'<div class="card-badge-row"><span class="badge '+lt+'">'+lt+'</span></div>'
        +'<div class="card-conf-label">'+pct+'% confidence</div>'
        +'<div class="card-conf-track"><div class="card-conf-fill '+bc+'" style="width:'+pct+'%"></div></div>'
        +'<div class="card-info">'
          +'<div class="card-info-row"><span>&#128247;</span><span>'+esc(e.camera||'&mdash;')+'</span></div>'
          +'<div class="card-info-row"><span>&#128336;</span><span>'+fmtTime(e.timestamp)+' &middot; '+fmtDate(e.timestamp)+'</span></div>'
          +emotionRow
        +'</div>';
    }
    return '<div class="card '+lt+'">'+photoHtml+'<div class="card-body">'+bodyHtml+'</div></div>';
  }).join('');
}

function renderCurrent(){if(_view==='table')renderTable(_events);else renderGallery(_events);}

async function fetchJSON(url){
  var r=await fetch(url);
  if(!r.ok)throw new Error('HTTP '+r.status);
  var ct=r.headers.get('content-type')||'';
  if(!ct.includes('json'))throw new Error('Non-JSON');
  return r.json();
}

async function refresh(){
  var results=await Promise.allSettled([
    fetchJSON('/api/events?limit=60'),
    fetchJSON('/api/stats'),
    fetchJSON('/api/status'),
  ]);
  var events=results[0].status==='fulfilled'?results[0].value:null;
  var stats =results[1].status==='fulfilled'?results[1].value:null;
  var sys   =results[2].status==='fulfilled'?results[2].value:null;
  if(sys){
    document.getElementById('s-cameras').textContent=sys.cameras||'0';
    document.getElementById('uptime-label').textContent='Uptime: '+sys.uptime;
  }
  if(stats){
    document.getElementById('s-blacklist').textContent=stats.today_blacklist||0;
    document.getElementById('s-whitelist').textContent=stats.today_whitelist||0;
    document.getElementById('s-unknown').textContent  =stats.today_unknown||0;
    document.getElementById('s-icscan').textContent   =stats.today_ic_scan||0;
  }
  if(events!==null){_events=events;renderCurrent();}
  document.getElementById('last-refresh').textContent=new Date().toLocaleTimeString();
}

setView(_view);
refresh();
setInterval(refresh,3000);
</script>
</body>
</html>
"""


_PERSON_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>His Majesty Office &mdash; Person Audit</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0d0d1a;color:#d0d0e8;min-height:100vh;display:flex;flex-direction:column}
header{background:#13132a;border-bottom:2px solid #1e1e40;padding:12px 28px;display:flex;align-items:flex-start;justify-content:space-between;gap:16px}
.header-titles h1{font-size:17px;font-weight:700;color:#ccccff;letter-spacing:.4px}
.header-sub{font-size:13px;color:#8888cc;margin-top:4px;display:flex;align-items:center;gap:7px}
.header-author{font-size:10.5px;color:#3a3a5a;margin-top:5px;font-style:italic;line-height:1.5}
.nav-links{display:flex;align-items:center;gap:10px;flex-shrink:0;padding-top:4px}
.nav-link{color:#8888cc;text-decoration:none;font-size:12px;font-weight:600;border:1px solid #333366;padding:5px 14px;border-radius:6px;white-space:nowrap;transition:all .15s}
.nav-link:hover{border-color:#5555bb;color:#ccccff}
.nav-link.active{background:#22224a;border-color:#5555bb;color:#ccccff}
/* Layout */
.layout{display:flex;flex:1;overflow:hidden}
/* Sidebar */
.sidebar{width:280px;min-width:280px;background:#10101f;border-right:1px solid #1a1a30;display:flex;flex-direction:column;overflow:hidden}
.sidebar-header{padding:14px 16px 10px;border-bottom:1px solid #1a1a30}
.sidebar-header h2{font-size:13px;font-weight:700;color:#8888cc;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
.search-box{width:100%;background:#16162a;border:1px solid #252545;color:#d0d0e8;padding:7px 11px;border-radius:6px;font-size:12px;outline:none}
.search-box:focus{border-color:#5555bb}
.search-box::placeholder{color:#444466}
.person-list{flex:1;overflow-y:auto;padding:6px 0}
.person-item{padding:10px 16px;cursor:pointer;border-bottom:1px solid #141428;transition:background .12s;display:flex;align-items:center;gap:10px}
.person-item:hover{background:#14142a}
.person-item.selected{background:#1a1a3a;border-left:3px solid #5555bb}
.person-avatar{width:36px;height:36px;border-radius:50%;background:#1e1e3a;display:flex;align-items:center;justify-content:center;font-size:15px;color:#333366;flex-shrink:0;overflow:hidden}
.person-avatar img{width:100%;height:100%;object-fit:cover}
.person-info{flex:1;min-width:0}
.person-name{font-size:12.5px;font-weight:600;color:#d0d0e8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.person-meta{font-size:10.5px;color:#555588;margin-top:2px;display:flex;align-items:center;gap:6px}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:9.5px;font-weight:700;letter-spacing:.3px;text-transform:uppercase}
.badge.blacklist{background:#4a0010;color:#ff3355;border:1px solid #7a0020}
.badge.whitelist{background:#0a3a1a;color:#22dd66;border:1px solid #0f5a28}
/* Main content */
.main-content{flex:1;overflow-y:auto;padding:24px 28px}
.empty-state{display:flex;align-items:center;justify-content:center;height:100%;color:#333355;font-size:14px;flex-direction:column;gap:10px}
.empty-state .icon{font-size:48px;color:#1e1e3a}
/* Profile section */
.profile{margin-bottom:24px}
.profile-header{display:flex;align-items:center;gap:18px;margin-bottom:16px}
.profile-name{font-size:22px;font-weight:700;color:#e0e0ff}
.profile-stats{display:flex;gap:16px;margin-top:6px}
.profile-stat{font-size:11px;color:#6666aa}
.profile-stat b{color:#9999cc;font-weight:600}
/* Training photos strip */
.training-photos{display:flex;gap:8px;margin-bottom:20px;overflow-x:auto;padding-bottom:6px}
.training-photo{width:80px;height:80px;object-fit:cover;border-radius:8px;border:2px solid #222240;cursor:pointer;transition:transform .15s;flex-shrink:0}
.training-photo:hover{transform:scale(1.08);border-color:#5555bb}
/* Section headers */
.section-title{font-size:13px;font-weight:700;color:#8888cc;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #1a1a30;display:flex;align-items:center;gap:8px}
.section-title .count{background:#22224a;color:#7777bb;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600}
/* Timeline table */
.timeline-table{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:28px}
.timeline-table thead tr{background:#13132a;color:#5566aa;text-transform:uppercase;font-size:10px;letter-spacing:.4px}
.timeline-table th{padding:7px 10px;text-align:left;font-weight:600;white-space:nowrap}
.timeline-table tbody tr{border-bottom:1px solid #181828;transition:background .12s}
.timeline-table tbody tr:hover{background:#14142a}
.timeline-table td{padding:7px 10px;vertical-align:middle}
.thumb{width:48px;height:36px;object-fit:cover;border-radius:4px;border:1px solid #222240;cursor:pointer;transition:transform .15s}
.thumb:hover{transform:scale(1.08)}
.conf-bar-wrap{width:60px;background:#1a1a30;border-radius:3px;height:4px;display:inline-block;vertical-align:middle;margin-left:5px}
.conf-bar{height:4px;border-radius:3px}
.conf-bar.red{background:#ff3355}.conf-bar.green{background:#22dd66}.conf-bar.gray{background:#5555aa}.conf-bar.orange{background:#ffaa33}
/* Similarity grid */
.sim-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px;margin-bottom:28px}
.sim-card{background:#16162a;border:1px solid #1e1e3a;border-radius:10px;overflow:hidden;transition:transform .18s,box-shadow .18s}
.sim-card:hover{transform:translateY(-2px);box-shadow:0 6px 24px #000b}
.sim-photo{width:100%;height:140px;object-fit:cover;display:block;cursor:pointer;transition:opacity .15s}
.sim-photo:hover{opacity:.88}
.sim-body{padding:10px 12px 12px}
.sim-pct{font-size:18px;font-weight:700;margin-bottom:4px}
.sim-pct.high{color:#22dd66}
.sim-pct.medium{color:#ffaa33}
.sim-pct.low{color:#ff3355}
.sim-dist{font-size:10px;color:#555588;margin-bottom:6px}
.sim-badge{display:inline-block;padding:2px 7px;border-radius:3px;font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:.3px}
.sim-badge.match{background:#0a3a1a;color:#22dd66;border:1px solid #0f5a28}
.sim-badge.no-match{background:#1a1a30;color:#555588;border:1px solid #252545}
.sim-file{font-size:9.5px;color:#444466;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* Add training photo button on similarity cards */
.sim-add-btn{display:block;width:100%;margin-top:8px;padding:5px 0;background:#16162a;border:1px solid #333366;color:#8888cc;border-radius:5px;font-size:10.5px;font-weight:600;cursor:pointer;transition:all .15s;letter-spacing:.3px}
.sim-add-btn:hover{border-color:#5555bb;color:#ccccff;background:#1e1e3a}
.sim-add-btn:disabled{opacity:.5;cursor:default}
.sim-add-btn.added{background:#0a3a1a;border-color:#0f5a28;color:#22dd66;cursor:default}
/* Training photo wrapper with remove button */
.training-photo-wrap{position:relative;flex-shrink:0}
.training-remove-btn{position:absolute;top:-4px;right:-4px;width:20px;height:20px;border-radius:50%;background:#4a0010;border:1px solid #7a0020;color:#ff3355;font-size:13px;line-height:18px;text-align:center;cursor:pointer;display:none;transition:all .15s;z-index:2;padding:0}
.training-photo-wrap:hover .training-remove-btn{display:block}
.training-remove-btn:hover{background:#7a0020;color:#fff}
/* No data */
.no-data{text-align:center;padding:40px;color:#333355;font-size:13px}
/* Loading spinner */
.loading{text-align:center;padding:40px;color:#444466;font-size:13px}
.loading::after{content:'';display:inline-block;width:14px;height:14px;border:2px solid #333366;border-top-color:#7777bb;border-radius:50%;animation:spin .6s linear infinite;margin-left:8px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
/* Modal */
#modal{display:none;position:fixed;inset:0;background:#000d;z-index:200;align-items:center;justify-content:center}
#modal.open{display:flex}
#modal img{max-width:92vw;max-height:92vh;border:2px solid #2a2a50;border-radius:8px;box-shadow:0 6px 40px #000c}
#modal-close{position:fixed;top:18px;right:24px;font-size:26px;cursor:pointer;color:#fff;background:#0008;border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center}
/* Scrollbar */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#0d0d1a}
::-webkit-scrollbar-thumb{background:#252545;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#333366}
</style>
</head>
<body>
<header>
  <div class="header-titles">
    <h1>His Majesty Office, Istana Nurul Iman</h1>
    <div class="header-sub">Person Audit &mdash; Face Recognition System</div>
    <div class="header-author">
      <em>Web app by: Ahmmad Hartunnoo Suharddy Bin Mohd Soud &nbsp;&middot;&nbsp; Pegawai Kerja &nbsp;&middot;&nbsp; Ketua Server and Security System</em>
    </div>
  </div>
  <div class="nav-links">
    <a href="/" class="nav-link">&larr; Live Dashboard</a>
    <a href="/person" class="nav-link active">Person Audit</a>
  </div>
</header>

<div class="layout">
  <div class="sidebar">
    <div class="sidebar-header">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
        <h2>Known Persons</h2>
        <button id="reload-btn" onclick="reloadDB()" style="background:#16162a;border:1px solid #333366;color:#8888cc;padding:4px 10px;border-radius:5px;font-size:11px;cursor:pointer;font-weight:600;transition:all .15s" onmouseover="this.style.borderColor='#5555bb';this.style.color='#ccccff'" onmouseout="this.style.borderColor='#333366';this.style.color='#8888cc'">&#x21BB; Reload</button>
      </div>
      <input type="text" class="search-box" id="person-search" placeholder="Search name..." oninput="filterPersons()">
    </div>
    <div class="person-list" id="person-list">
      <div class="loading">Loading persons</div>
    </div>
  </div>

  <div class="main-content" id="main-content">
    <div class="empty-state">
      <div class="icon">&#128100;</div>
      <div>Select a person from the sidebar to view their audit details.</div>
    </div>
  </div>
</div>

<div id="modal" onclick="closeModal()">
  <div id="modal-close" onclick="closeModal()">&#x2715;</div>
  <img id="modal-img" src="" alt="">
</div>

<script>
var _persons=[];
var _selectedPerson=null;

function closeModal(){document.getElementById('modal').classList.remove('open')}
function openModal(src){document.getElementById('modal-img').src=src;document.getElementById('modal').classList.add('open')}
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function pad(n){return String(n).padStart(2,'0')}
function fmtTime(iso){var d=new Date(iso);return pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds())}
function fmtDate(iso){var d=new Date(iso);return d.toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'})}
function fmtDateTime(iso){return fmtDate(iso)+' '+fmtTime(iso)}
function ssToUrl(ss){return ss?('/'+ss.replace(/\\\\/g,'/')):'';}
function barCls(lt){return lt==='blacklist'?'red':lt==='whitelist'?'green':lt==='ic_scan'?'orange':'gray'}

async function fetchJSON(url){
  var r=await fetch(url);
  if(!r.ok)throw new Error('HTTP '+r.status);
  return r.json();
}

/* ── Sidebar ────────────────────────────────────────────────────── */

async function loadPersons(){
  try{
    _persons=await fetchJSON('/api/person/list');
    renderPersonList();
  }catch(e){
    document.getElementById('person-list').innerHTML='<div class="no-data">Failed to load persons.</div>';
  }
}

function filterPersons(){
  renderPersonList();
}

async function reloadDB(){
  var btn=document.getElementById('reload-btn');
  btn.textContent='Reloading...';
  btn.disabled=true;
  try{
    await fetchJSON('/api/person/reload');
    await loadPersons();
    btn.textContent='\u21BB Reload';
  }catch(e){
    btn.textContent='Failed';
    setTimeout(function(){btn.textContent='\u21BB Reload'},2000);
  }
  btn.disabled=false;
}

function renderPersonList(){
  var q=(document.getElementById('person-search').value||'').toLowerCase();
  var list=document.getElementById('person-list');
  var filtered=_persons.filter(function(p){return p.name.toLowerCase().indexOf(q)>=0});
  if(!filtered.length){list.innerHTML='<div class="no-data">No persons found.</div>';return;}
  list.innerHTML=filtered.map(function(p){
    var sel=_selectedPerson===p.name?' selected':'';
    var lt=p.list_type||'whitelist';
    var avatarUrl=p.photo_url||'';
    var avatarHtml=avatarUrl
      ?'<img src="'+esc(avatarUrl)+'" alt="">'
      :'&#128100;';
    return '<div class="person-item'+sel+'" data-person="'+esc(p.name)+'" onclick="selectPerson(this.dataset.person)">'
      +'<div class="person-avatar">'+avatarHtml+'</div>'
      +'<div class="person-info">'
        +'<div class="person-name">'+esc(p.name)+'</div>'
        +'<div class="person-meta">'
          +'<span class="badge '+lt+'">'+lt+'</span>'
          +'<span>'+p.photo_count+' photo'+(p.photo_count!==1?'s':'')+'</span>'
          +'<span>'+p.detection_count+' det.</span>'
        +'</div>'
      +'</div>'
    +'</div>';
  }).join('');
}

/* ── Person detail ──────────────────────────────────────────────── */

async function selectPerson(name){
  _selectedPerson=name;
  renderPersonList();
  var main=document.getElementById('main-content');
  main.innerHTML='<div class="loading">Loading audit data for '+esc(name)+'</div>';

  try{
    var results=await Promise.allSettled([
      fetchJSON('/api/person/'+encodeURIComponent(name)+'/events'),
      fetchJSON('/api/person/'+encodeURIComponent(name)+'/similarity'),
    ]);
    var eventsData=results[0].status==='fulfilled'?results[0].value:null;
    var simData   =results[1].status==='fulfilled'?results[1].value:null;

    var person=_persons.find(function(p){return p.name===name});
    renderPersonDetail(person,eventsData,simData);
  }catch(e){
    main.innerHTML='<div class="no-data">Failed to load data for '+esc(name)+'.</div>';
  }
}

function renderPersonDetail(person,eventsData,simData){
  var main=document.getElementById('main-content');
  if(!person){main.innerHTML='<div class="no-data">Person not found.</div>';return;}

  var events=eventsData?eventsData.events:[];
  var simResults=simData?simData.results:[];
  var photos=person.photos||[];
  var lt=person.list_type||'whitelist';

  /* Profile */
  var photoCount=photos.length;
  var photosHtml=photos.map(function(url){
    var parts=url.split('/');
    var fname=parts[parts.length-1];
    var removeBtn=(photoCount>1)
      ?'<button class="training-remove-btn" title="Remove photo" data-person="'+esc(person.name)+'" data-file="'+esc(fname)+'" onclick="event.stopPropagation();removeTrainingPhoto(this.dataset.person,this.dataset.file)">&#x2715;</button>'
      :'';
    return '<div class="training-photo-wrap">'
      +removeBtn
      +'<img class="training-photo" src="'+esc(url)+'" onclick="openModal(this.src)" loading="lazy">'
      +'</div>';
  }).join('')||'<div style="color:#333355;font-size:12px;padding:8px 0">No training photos.</div>';

  var firstSeen=events.length?events[events.length-1].timestamp:'N/A';
  var lastSeen=events.length?events[0].timestamp:'N/A';

  var html=''
    +'<div class="profile">'
      +'<div class="profile-header">'
        +'<div>'
          +'<div class="profile-name">'+esc(person.name)+' <span class="badge '+lt+'" style="vertical-align:middle;font-size:11px">'+lt+'</span></div>'
          +'<div class="profile-stats">'
            +'<div class="profile-stat">Total detections: <b>'+person.detection_count+'</b></div>'
            +'<div class="profile-stat">First seen: <b>'+(firstSeen!=='N/A'?fmtDateTime(firstSeen):firstSeen)+'</b></div>'
            +'<div class="profile-stat">Last seen: <b>'+(lastSeen!=='N/A'?fmtDateTime(lastSeen):lastSeen)+'</b></div>'
          +'</div>'
        +'</div>'
      +'</div>'
      +'<div class="training-photos">'+photosHtml+'</div>'
    +'</div>';

  /* Detection timeline */
  html+='<div class="section-title">Detection Timeline <span class="count">'+events.length+'</span></div>';
  if(events.length){
    html+='<table class="timeline-table"><thead><tr>'
      +'<th>Time</th><th>Camera</th><th>Confidence</th><th>Screenshot</th><th>Emotion</th><th>Age</th><th>ID / IC Info</th>'
      +'</tr></thead><tbody>';
    html+=events.map(function(e){
      var conf=parseFloat(e.confidence||0);
      var bc=barCls(e.list_type||lt);
      var barW=Math.round(conf*60);
      var ssUrl=ssToUrl(e.screenshot||'');
      var thumb=ssUrl?'<img class="thumb" loading="lazy" src="'+ssUrl+'" onclick="openModal(this.src)">':'&mdash;';
      var isIC=(e.list_type||'')==='ic_scan';
      var icInfo='';
      if(isIC){
        icInfo=(e.ic_doc_number?esc(e.ic_doc_number):'')+(e.ic_nationality?' '+esc(e.ic_nationality):'');
      }
      return '<tr>'
        +'<td title="'+esc(fmtDate(e.timestamp))+'">'+fmtTime(e.timestamp)+'<br><small style="color:#444466">'+fmtDate(e.timestamp)+'</small></td>'
        +'<td>'+esc(e.camera||'')+'</td>'
        +'<td>'+(conf*100).toFixed(0)+'%<span class="conf-bar-wrap"><span class="conf-bar '+bc+'" style="width:'+barW+'px"></span></span></td>'
        +'<td>'+thumb+'</td>'
        +'<td>'+esc(e.emotion||'&mdash;')+'</td>'
        +'<td>'+(e.age!=null?'~'+e.age:'&mdash;')+'</td>'
        +'<td>'+(icInfo||'&mdash;')+'</td>'
        +'</tr>';
    }).join('');
    html+='</tbody></table>';
  }else{
    html+='<div class="no-data" style="margin-bottom:28px">No detection events recorded.</div>';
  }

  /* Unknown similarity */
  html+='<div class="section-title">Unknown Face Similarity <span class="count">'+simResults.length+'</span></div>';
  if(simResults.length){
    html+='<div class="sim-grid">';
    html+=simResults.map(function(s){
      var pct=s.similarity_pct;
      var pctCls=pct>=60?'high':pct>=30?'medium':'low';
      var isMat=s.is_match;
      var addBtn='<button class="sim-add-btn" data-person="'+esc(person.name)+'" data-file="'+esc(s.filename)+'" onclick="addTrainingPhoto(this.dataset.person,this.dataset.file,this)">+ Add as Training Photo</button>';
      return '<div class="sim-card">'
        +'<img class="sim-photo" src="/unknown_faces/'+encodeURIComponent(s.filename)+'" onclick="openModal(this.src)" loading="lazy">'
        +'<div class="sim-body">'
          +'<div class="sim-pct '+pctCls+'">'+pct.toFixed(1)+'%</div>'
          +'<div class="sim-dist">Cosine distance: '+s.distance.toFixed(4)+'</div>'
          +'<span class="sim-badge '+(isMat?'match':'no-match')+'">'+( isMat?'Match':'No Match')+'</span>'
          +'<div class="sim-file" title="'+esc(s.filename)+'">'+esc(s.filename)+'</div>'
          +addBtn
        +'</div>'
      +'</div>';
    }).join('');
    html+='</div>';
  }else{
    html+='<div class="no-data">No unknown face crops available for comparison.</div>';
  }

  main.innerHTML=html;
}

/* ── Add / Remove training photos ────────────────────────────── */

async function addTrainingPhoto(personName, unknownFilename, btn){
  if(btn.disabled)return;
  btn.disabled=true;
  btn.textContent='Adding...';
  try{
    var r=await fetch('/api/person/'+encodeURIComponent(personName)+'/add-photo',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({filename:unknownFilename})
    });
    var data=await r.json();
    if(!r.ok||data.status!=='ok'){
      btn.textContent='Error: '+(data.message||'failed');
      setTimeout(function(){btn.textContent='+ Add as Training Photo';btn.disabled=false},2500);
      return;
    }
  }catch(e){
    btn.textContent='Error';
    setTimeout(function(){btn.textContent='+ Add as Training Photo';btn.disabled=false},2500);
    return;
  }
  btn.textContent='Added!';
  btn.classList.add('added');
  try{await loadPersons()}catch(e){}
  try{if(_selectedPerson===personName)selectPerson(personName)}catch(e){}
}

async function removeTrainingPhoto(personName, photoFilename){
  if(!confirm('Remove training photo "'+photoFilename+'" from '+personName+'?'))return;
  try{
    var r=await fetch('/api/person/'+encodeURIComponent(personName)+'/remove-photo',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({filename:photoFilename})
    });
    var data=await r.json();
    if(!r.ok||data.status!=='ok'){
      alert('Failed to remove photo: '+(data.message||'unknown error'));
      return;
    }
  }catch(e){
    alert('Failed to remove photo: '+e.message);
    return;
  }
  try{await loadPersons()}catch(e){}
  try{if(_selectedPerson===personName)selectPerson(personName)}catch(e){}
}

/* ── Init ───────────────────────────────────────────────────────── */
loadPersons();
</script>
</body>
</html>
"""


class DashboardServer:
    """
    Lightweight Flask web server running on a daemon thread.
    Provides a live security event dashboard in the browser.
    """

    def __init__(
        self,
        cfg: DashboardConfig,
        events_jsonl_path: str,
        screenshots_dir: str,
        start_time: Optional[datetime] = None,
        db_manager=None,
        matcher=None,
        paths_cfg=None,
    ) -> None:
        self._cfg             = cfg
        self._jsonl_path      = events_jsonl_path
        self._screenshots_dir = screenshots_dir
        self._start_time      = start_time or datetime.now()
        self._camera_count    = 0
        self._thread: Optional[threading.Thread] = None
        # Stats cache — recomputed at most once every 5 s
        self._stats_cache: Optional[dict] = None
        self._stats_cache_ts: float = 0.0
        # Person audit page — face DB + matcher for similarity search
        self._db_manager = db_manager
        self._matcher    = matcher
        self._paths_cfg  = paths_cfg
        self._unknown_cache: Optional[dict] = None
        self._unknown_cache_lock = threading.Lock()

    def set_camera_count(self, n: int) -> None:
        self._camera_count = n

    def start(self) -> None:
        if not self._cfg.enabled:
            return
        self._thread = threading.Thread(
            target=self._serve, daemon=True, name="dashboard"
        )
        self._thread.start()
        log.info(
            "Dashboard running at http://%s:%d",
            self._cfg.host, self._cfg.port,
        )

    # ── Flask app ─────────────────────────────────────────────────────

    def _serve(self) -> None:
        try:
            from flask import Flask, jsonify, send_from_directory, Response

            app = Flask(__name__)
            # Silence Flask request logs — already have our own logger
            import logging as _logging
            _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

            jsonl  = self._jsonl_path
            ss_dir = self._screenshots_dir

            @app.route("/")
            def index():
                return Response(_HTML, mimetype="text/html")

            @app.route("/gallery")
            def gallery():
                from flask import redirect
                return redirect("/?view=gallery")

            @app.route("/api/events")
            def api_events():
                try:
                    from flask import request as freq
                    limit = int(freq.args.get("limit", 60))
                    rows  = _read_jsonl(jsonl, limit=limit)
                    resp  = jsonify(rows)
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    return resp
                except Exception as exc:
                    log.error("api_events error: %s", exc)
                    return jsonify([]), 200

            @app.route("/api/stats")
            def api_stats():
                try:
                    import time as _time
                    now = _time.monotonic()
                    if self._stats_cache is not None and now - self._stats_cache_ts < 5.0:
                        resp = jsonify(self._stats_cache)
                        resp.headers["Access-Control-Allow-Origin"] = "*"
                        return resp
                    rows  = _read_jsonl(jsonl, limit=10_000)
                    today = date.today().isoformat()
                    bl = wl = unk = ic = 0
                    for r in rows:
                        if not r.get("timestamp", "").startswith(today):
                            continue
                        lt = r.get("list_type", "")
                        if lt == "blacklist":   bl  += 1
                        elif lt == "whitelist": wl  += 1
                        elif lt == "ic_scan":   ic  += 1
                        else:                   unk += 1
                    payload = dict(
                        today_blacklist=bl,
                        today_whitelist=wl,
                        today_unknown=unk,
                        today_ic_scan=ic,
                        today_total=bl + wl + unk + ic,
                    )
                    self._stats_cache    = payload
                    self._stats_cache_ts = now
                    resp = jsonify(payload)
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    return resp
                except Exception as exc:
                    log.error("api_stats error: %s", exc)
                    return jsonify(dict(today_blacklist=0, today_whitelist=0,
                                       today_unknown=0, today_total=0)), 200

            @app.route("/api/status")
            def api_status():
                try:
                    elapsed = datetime.now() - self._start_time
                    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
                    m, s    = divmod(rem, 60)
                    resp = jsonify(dict(
                        uptime=f"{h:02d}:{m:02d}:{s:02d}",
                        cameras=self._camera_count,
                        started=self._start_time.isoformat(timespec="seconds"),
                    ))
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    return resp
                except Exception as exc:
                    log.error("api_status error: %s", exc)
                    return jsonify(dict(uptime="--:--:--", cameras=0)), 200

            @app.route("/screenshots/<path:filename>")
            def serve_screenshot(filename):
                resp = send_from_directory(os.path.abspath(ss_dir), filename)
                resp.headers["Cache-Control"] = "public, max-age=86400"
                return resp

            @app.route("/scans/<path:filename>")
            def serve_scan(filename):
                resp = send_from_directory(os.path.abspath("scans"), filename)
                resp.headers["Cache-Control"] = "public, max-age=86400"
                return resp

            # ── Person Audit routes ──────────────────────────────────────

            @app.route("/person")
            def person_page():
                return Response(_PERSON_HTML, mimetype="text/html")

            @app.route("/api/person/reload")
            def api_person_reload():
                try:
                    if self._db_manager and self._paths_cfg:
                        self._db_manager.reload(
                            self._paths_cfg.whitelist_dir,
                            self._paths_cfg.blacklist_dir,
                            self._paths_cfg.cache_dir,
                        )
                    resp = jsonify({"status": "ok"})
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    return resp
                except Exception as exc:
                    log.error("api_person_reload error: %s", exc)
                    return jsonify({"status": "error", "message": str(exc)}), 500

            @app.route("/api/person/list")
            def api_person_list():
                try:
                    persons = self._build_person_list()
                    resp = jsonify(persons)
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    return resp
                except Exception as exc:
                    log.error("api_person_list error: %s", exc)
                    return jsonify([]), 200

            @app.route("/api/person/<name>/events")
            def api_person_events(name):
                try:
                    rows = _read_jsonl(jsonl, limit=10_000)
                    filtered = [
                        r for r in rows
                        if (r.get("name") or "").replace("_", " ") == name
                    ]
                    resp = jsonify({"person": name, "events": filtered})
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    return resp
                except Exception as exc:
                    log.error("api_person_events error: %s", exc)
                    return jsonify({"person": name, "events": []}), 200

            @app.route("/api/person/<name>/similarity")
            def api_person_similarity(name):
                try:
                    results = self._compute_similarity(name)
                    resp = jsonify({"person": name, "results": results})
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    return resp
                except Exception as exc:
                    log.error("api_person_similarity error: %s", exc)
                    return jsonify({"person": name, "results": []}), 200

            @app.route("/known_faces/<list_type>/<path:filename>")
            def serve_known_face(list_type, filename):
                if list_type not in ("whitelist", "blacklist"):
                    return "Not found", 404
                base = os.path.join(
                    self._paths_cfg.whitelist_dir if list_type == "whitelist"
                    else self._paths_cfg.blacklist_dir
                ) if self._paths_cfg else os.path.join("known_faces", list_type)
                resp = send_from_directory(os.path.abspath(base), filename)
                resp.headers["Cache-Control"] = "public, max-age=3600"
                return resp

            @app.route("/unknown_faces/<path:filename>")
            def serve_unknown_face(filename):
                base = (self._paths_cfg.unknown_faces_dir
                        if self._paths_cfg else "known_faces/unknown")
                resp = send_from_directory(os.path.abspath(base), filename)
                resp.headers["Cache-Control"] = "public, max-age=3600"
                return resp

            @app.route("/api/person/<name>/add-photo", methods=["POST"])
            def api_person_add_photo(name):
                from flask import request as freq
                from src.face_db import SUPPORTED_EXTENSIONS
                try:
                    data = freq.get_json(force=True)
                    filename = data.get("filename", "")

                    # Validate filename
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        return jsonify({"status": "error", "message": "Invalid file type"}), 400
                    if os.sep in filename or "/" in filename or ".." in filename:
                        return jsonify({"status": "error", "message": "Invalid filename"}), 400

                    # Find person's list type
                    person_lt = self._find_person_list_type(name)
                    if not person_lt:
                        return jsonify({"status": "error", "message": "Person not found"}), 404

                    # Resolve directories
                    unknown_dir = (self._paths_cfg.unknown_faces_dir
                                   if self._paths_cfg else "known_faces/unknown")
                    target_dir = (
                        self._paths_cfg.whitelist_dir if person_lt == "whitelist"
                        else self._paths_cfg.blacklist_dir
                    ) if self._paths_cfg else f"known_faces/{person_lt}"

                    src_path = os.path.join(unknown_dir, filename)
                    if not os.path.isfile(src_path):
                        return jsonify({"status": "error", "message": "Source file not found"}), 404

                    # Determine next available filename: PersonName_N.ext
                    safe_name = name.replace(" ", "_")
                    existing = [
                        f for f in os.listdir(target_dir)
                        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                    ]
                    # Find existing files for this person
                    max_n = 0
                    has_base = False
                    for ef in existing:
                        stem = os.path.splitext(ef)[0]
                        if stem == safe_name:
                            has_base = True
                            max_n = max(max_n, 1)
                        elif stem.startswith(safe_name + "_"):
                            suffix = stem[len(safe_name) + 1:]
                            if suffix.isdigit():
                                max_n = max(max_n, int(suffix))

                    next_n = max_n + 1
                    if not has_base and max_n == 0:
                        new_fname = f"{safe_name}{ext}"
                    else:
                        new_fname = f"{safe_name}_{next_n}{ext}"

                    dest_path = os.path.join(target_dir, new_fname)
                    shutil.copy2(src_path, dest_path)
                    log.info("Added training photo: %s → %s", filename, new_fname)

                    # Rebuild embeddings
                    if self._db_manager and self._paths_cfg:
                        self._db_manager.reload(
                            self._paths_cfg.whitelist_dir,
                            self._paths_cfg.blacklist_dir,
                            self._paths_cfg.cache_dir,
                        )

                    resp = jsonify({"status": "ok", "saved_as": new_fname})
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    return resp
                except Exception as exc:
                    log.error("api_person_add_photo error: %s", exc)
                    return jsonify({"status": "error", "message": str(exc)}), 500

            @app.route("/api/person/<name>/remove-photo", methods=["POST"])
            def api_person_remove_photo(name):
                from flask import request as freq
                from src.face_db import SUPPORTED_EXTENSIONS
                try:
                    data = freq.get_json(force=True)
                    filename = data.get("filename", "")

                    # Validate filename
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        return jsonify({"status": "error", "message": "Invalid file type"}), 400
                    if os.sep in filename or "/" in filename or ".." in filename:
                        return jsonify({"status": "error", "message": "Invalid filename"}), 400

                    # Find person's list type
                    person_lt = self._find_person_list_type(name)
                    if not person_lt:
                        return jsonify({"status": "error", "message": "Person not found"}), 404

                    target_dir = (
                        self._paths_cfg.whitelist_dir if person_lt == "whitelist"
                        else self._paths_cfg.blacklist_dir
                    ) if self._paths_cfg else f"known_faces/{person_lt}"

                    # Count how many photos this person has
                    safe_name = name.replace(" ", "_")
                    person_files = [
                        f for f in os.listdir(target_dir)
                        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                        and (os.path.splitext(f)[0] == safe_name
                             or (os.path.splitext(f)[0].startswith(safe_name + "_")
                                 and os.path.splitext(f)[0][len(safe_name) + 1:].isdigit()))
                    ]
                    if len(person_files) <= 1:
                        return jsonify({"status": "error", "message": "Cannot remove last photo"}), 400

                    file_path = os.path.join(target_dir, filename)
                    if not os.path.isfile(file_path):
                        return jsonify({"status": "error", "message": "File not found"}), 404

                    os.remove(file_path)
                    log.info("Removed training photo: %s/%s", person_lt, filename)

                    # Rebuild embeddings
                    if self._db_manager and self._paths_cfg:
                        self._db_manager.reload(
                            self._paths_cfg.whitelist_dir,
                            self._paths_cfg.blacklist_dir,
                            self._paths_cfg.cache_dir,
                        )

                    resp = jsonify({"status": "ok"})
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    return resp
                except Exception as exc:
                    log.error("api_person_remove_photo error: %s", exc)
                    return jsonify({"status": "error", "message": str(exc)}), 500

            app.run(
                host=self._cfg.host,
                port=self._cfg.port,
                debug=False,
                use_reloader=False,
                threaded=True,
            )

        except Exception as exc:
            log.error("Dashboard server failed: %s", exc)

    # ── Person Audit helpers ─────────────────────────────────────────

    def _find_person_list_type(self, name: str) -> Optional[str]:
        """Return 'whitelist' or 'blacklist' for a known person, or None."""
        if self._db_manager is None:
            return None
        db = self._db_manager.get()
        if name in db.whitelist.names:
            return "whitelist"
        if name in db.blacklist.names:
            return "blacklist"
        return None

    def _build_person_list(self) -> list:
        """Build list of all known persons with metadata for the sidebar."""
        if self._db_manager is None:
            return []

        from src.face_db import SUPPORTED_EXTENSIONS

        db = self._db_manager.get()
        events = _read_jsonl(self._jsonl_path, limit=10_000)

        # Count detections per person name
        det_counts: dict = {}
        for ev in events:
            name = (ev.get("name") or "").replace("_", " ")
            if name and name != "UNKNOWN VISITOR":
                det_counts[name] = det_counts.get(name, 0) + 1

        persons = []
        for list_model in (db.whitelist, db.blacklist):
            lt = list_model.list_type
            base_dir = (
                self._paths_cfg.whitelist_dir if lt == "whitelist"
                else self._paths_cfg.blacklist_dir
            ) if self._paths_cfg else f"known_faces/{lt}"

            # Group photos by person name
            person_photos: dict = {}
            if os.path.isdir(base_dir):
                for fname in sorted(os.listdir(base_dir)):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        continue
                    stem = os.path.splitext(fname)[0]
                    display = stem.replace("_", " ").replace("-", " ").strip()
                    # Strip trailing number suffix for grouping (e.g. "John Doe 2" → "John Doe")
                    parts = display.rsplit(" ", 1)
                    person = parts[0] if len(parts) == 2 and parts[1].isdigit() else display
                    person_photos.setdefault(person, []).append(fname)

            for name in list_model.names:
                fnames = person_photos.get(name, [])
                photo_urls = [
                    f"/known_faces/{lt}/{f}" for f in fnames
                ]
                persons.append({
                    "name": name,
                    "list_type": lt,
                    "photo_count": len(fnames),
                    "photo_url": photo_urls[0] if photo_urls else "",
                    "photos": photo_urls,
                    "detection_count": det_counts.get(name, 0),
                })

        # Sort: blacklist first, then alphabetical
        persons.sort(key=lambda p: (0 if p["list_type"] == "blacklist" else 1, p["name"]))
        return persons

    def _load_unknown_embeddings(self) -> dict:
        """
        Load or rebuild the unknown face embedding cache.
        Returns dict mapping filename → np.ndarray (512-dim).
        Only processes Unknown_* files (face crops from camera worker).
        """
        import pickle
        import numpy as np

        unknown_dir = (
            self._paths_cfg.unknown_faces_dir
            if self._paths_cfg else "known_faces/unknown"
        )
        cache_dir = self._paths_cfg.cache_dir if self._paths_cfg else "cache"

        if not os.path.isdir(unknown_dir):
            return {}

        from src.face_db import SUPPORTED_EXTENSIONS

        # Current state of unknown dir
        current_files = sorted(
            f for f in os.listdir(unknown_dir)
            if (os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                and f.startswith("Unknown_"))
        )
        if not current_files:
            return {}

        dir_mtime = max(
            os.path.getmtime(unknown_dir),
            max(
                (os.path.getmtime(os.path.join(unknown_dir, f)) for f in current_files),
                default=0.0,
            ),
        )

        # Check if cache is still valid
        pkl_path = os.path.join(cache_dir, "arcface_onnx_unknown_embeddings.pkl")
        meta_path = os.path.join(cache_dir, "arcface_onnx_unknown_meta.txt")

        if os.path.exists(pkl_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    lines = fh.read().splitlines()
                cached_mtime = float(lines[0].split("=", 1)[1])
                cached_files = set(lines[1].split("=", 1)[1].split(",")) if len(lines) > 1 else set()
                if cached_mtime >= dir_mtime and cached_files == set(current_files):
                    with open(pkl_path, "rb") as fh:
                        cache = pickle.load(fh)
                    log.info("[unknown] Embeddings loaded from cache (%d crops).", len(cache))
                    return cache
            except Exception:
                pass  # rebuild

        # Rebuild: extract embeddings for all unknown face crops
        log.info("[unknown] Building embeddings for %d face crop(s)...", len(current_files))
        from src.arcface_onnx import get_arcface
        import cv2

        arcface = get_arcface()
        result = {}
        for fname in current_files:
            path = os.path.join(unknown_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            vec = arcface.get_embedding(img)
            if vec is not None:
                result[fname] = vec

        # Save cache
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(pkl_path, "wb") as fh:
                pickle.dump(result, fh)
            with open(meta_path, "w", encoding="utf-8") as fh:
                fh.write(f"mtime={dir_mtime}\n")
                fh.write(f"files={','.join(sorted(current_files))}\n")
            log.info("[unknown] Embedding cache saved (%d crops).", len(result))
        except Exception as exc:
            log.warning("[unknown] Cache save failed: %s", exc)

        return result

    def _compute_similarity(self, person_name: str, top_n: int = 50) -> list:
        """
        Compare a known person's embeddings against all unknown face crops.
        Returns list of dicts sorted by distance (ascending).
        """
        if self._db_manager is None or self._matcher is None:
            return []

        db = self._db_manager.get()

        # Gather all embeddings for this person across both lists
        person_embeddings = []
        for list_model in (db.whitelist, db.blacklist):
            for pe in list_model.embeddings:
                if pe.name == person_name:
                    person_embeddings.append(pe.embedding)

        if not person_embeddings:
            return []

        # Load unknown embeddings (cached)
        with self._unknown_cache_lock:
            unknown_map = self._load_unknown_embeddings()
            self._unknown_cache = unknown_map

        if not unknown_map:
            return []

        threshold = self._matcher.threshold
        results = []

        for fname, unk_emb in unknown_map.items():
            # Min distance to any of the person's training embeddings
            min_dist = min(
                self._matcher._cosine_distance(unk_emb, pe)
                for pe in person_embeddings
            )
            sim_pct = max(0.0, (1.0 - min_dist / threshold)) * 100
            results.append({
                "filename": fname,
                "distance": round(float(min_dist), 6),
                "similarity_pct": round(float(sim_pct), 1),
                "is_match": min_dist <= threshold,
            })

        results.sort(key=lambda r: r["distance"])
        return results[:top_n]


def _read_jsonl(path: str, limit: int = 100) -> list:
    """Read last `limit` lines from a JSONL file, newest first."""
    if not os.path.isfile(path):
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        return []
    rows.reverse()          # newest first
    return rows[:limit]
