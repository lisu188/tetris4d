"use strict";
(() => {
  const state = {data:null, paused:false, window:"1h", received:null, error:null, busy:false};
  const finite = x => typeof x === "number" && Number.isFinite(x);
  const esc = x => String(x ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const num = (x, digits=0) => finite(x) ? x.toLocaleString(undefined,{maximumFractionDigits:digits}) : "—";
  const stamp = x => {const t = typeof x === "string" ? Date.parse(x) : NaN; return Number.isFinite(t) ? t : null;};
  const clock = x => finite(x) ? [Math.floor(Math.max(0,x)/3600),Math.floor(Math.max(0,x)/60)%60,Math.floor(Math.max(0,x))%60].map(n=>String(n).padStart(2,"0")).join(":") : "—";
  const age = x => {const t=stamp(x); if(t===null)return "Awaiting telemetry";const s=Math.max(0,(Date.now()-t)/1000);return s<60?`${Math.floor(s)}s ago`:s<3600?`${Math.floor(s/60)}m ago`:`${Math.floor(s/3600)}h ${Math.floor(s/60)%60}m ago`;};
  const local = x => stamp(x)===null ? "—" : new Date(x).toLocaleTimeString(undefined,{hour:"2-digit",minute:"2-digit"});
  const $ = id => document.getElementById(id);
  const put = (id,value) => {$(id).textContent=value;};
  const phases = {
    waiting_comparison:["COMPARISON","Comparison in progress","The existing policies are finishing their shared-seed comparison. Production training follows.",0],
    waiting_training:["PREFLIGHT","Waiting for production","The production run has not reported training activity yet. Its budget is shown when the clock begins.",0],
    calibration:["CALIBRATION","Calibrating the critic","The actor is frozen while completed games teach the value head. These games are separate from PPO training.",1],
    training:["PPO TRAINING","Learning through play","Parallel environments collect placements. PPO updates the policy from these rollouts.",2],
    selection:["SELECTION","Evaluating a checkpoint","Greedy selection games measure survival and choose the best checkpoint observed during this run.",2],
    optimizing:["OPTIMIZATION PAUSE","Paused for requested optimization","Training stopped gracefully for the requested performance work and sixteen-environment migration. The total budget clock continues.",2],
    complete:["PRODUCTION COMPLETE","Production budget finished","The production phase is complete. Final comparison and neuron analysis are separate steps.",3],
    failed:["ATTENTION","Run needs attention","The controller or calibration reported a failure. Retained telemetry remains available below.",-1],
    unknown:["AWAITING DATA","Waiting for telemetry","No current training phase can be established from the available source events.",-1]
  };

  function chart(rows,key,{label,percent=false,line=true,xkey=null,rolling=false,calibration=false,step=false,holdTo=null,minYMax=1}={}) {
    const points=rows.map(r=>({r,x:xkey? r[xkey]:stamp(r.timestamp_utc),y:finite(r[key])?r[key]*(percent?100:1):null}));
    const valid=points.filter(p=>finite(p.x)&&finite(p.y));
    if(!valid.length)return `<div class="chart-empty"><span>No ${esc(label.toLowerCase())} in this window</span><small>Only recorded measurements appear here. Try All history for earlier observations.</small></div>`;
    const w=700,h=230,pad={l:51,r:15,t:15,b:32},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;
    let xmin=Math.min(...valid.map(p=>p.x)),xmax=Math.max(...valid.map(p=>p.x));
    if(step&&finite(holdTo))xmax=Math.max(xmax,holdTo);
    if(xmin===xmax){xmin-=xkey?1:30000;xmax+=xkey?1:30000;}
    const ymax=percent?100:Math.max(minYMax,...valid.map(p=>p.y))*1.12;
    const px=x=>pad.l+(x-xmin)/(xmax-xmin)*pw,py=y=>pad.t+ph-y/ymax*ph;
    const xf=x=>xkey?num(x):new Date(x).toLocaleTimeString(undefined,{hour:"2-digit",minute:"2-digit"});
    const yf=y=>ymax<0.1?num(y,Math.min(6,Math.ceil(-Math.log10(ymax))+2)):ymax<2?num(y,2):num(y,ymax<10?1:0);
    let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}: ${valid.length} recorded measurements. ${xkey?"Horizontal axis: rollout updates.":"Horizontal axis: local time."} Vertical axis: ${esc(percent?"percent":key.replaceAll("_"," "))}."><title>${esc(label)}</title>`;
    for(let i=0;i<4;i++){const y=ymax*i/3;svg+=`<line class="plot-grid" x1="${pad.l}" x2="${w-pad.r}" y1="${py(y)}" y2="${py(y)}"/><text class="plot-axis" text-anchor="end" x="${pad.l-10}" y="${py(y)+3}">${esc(yf(y))}</text>`;}
    for(let i=0;i<3;i++){const x=xmin+(xmax-xmin)*i/2;svg+=`<text class="plot-axis" text-anchor="${i===0?"start":i===2?"end":"middle"}" x="${px(x)}" y="${h-8}">${esc(xf(x))}</text>`;}
    const gaps=valid.slice(1).map((p,i)=>stamp(p.r.timestamp_utc)-stamp(valid[i].r.timestamp_utc)).filter(x=>x>0).sort((a,b)=>a-b);
    const gapLimit=Math.max(300000,(gaps[Math.floor(gaps.length/2)]||60000)*5);
    const splits=(a,b)=>!a||!finite(a.y)||!finite(a.x)||(!step&&stamp(b.r.timestamp_utc)-stamp(a.r.timestamp_utc)>gapLimit)||(calibration&&(a.r.calibration_game_index!==b.r.calibration_game_index||a.r.seed!==b.r.seed||b.y<a.y));
    const pathFor=arr=>{let prev=null,path="";for(const p of arr){if(!finite(p.x)||!finite(p.y)){prev=null;continue;}path+=splits(prev,p)?`M${px(p.x).toFixed(2)},${py(p.y).toFixed(2)} `:step?`H${px(p.x).toFixed(2)} V${py(p.y).toFixed(2)} `:`L${px(p.x).toFixed(2)},${py(p.y).toFixed(2)} `;prev=p;}if(step&&prev&&finite(holdTo)&&holdTo>=prev.x)path+=`H${px(holdTo).toFixed(2)}`;return path;};
    if(line)svg+=`<path class="plot-line" stroke="var(--warm)" d="${pathFor(points.filter(p=>!p.r.active_only))}"/>`;
    if(rolling&&valid.length>=5){let streak=[];const means=[];let prev=null;for(const p of points){if(!finite(p.y)||!finite(p.x)||splits(prev,p))streak=[];if(finite(p.y)&&finite(p.x)){streak.push(p.y);if(streak.length>5)streak.shift();means.push({...p,y:streak.length===5?streak.reduce((a,b)=>a+b,0)/5:null});}else means.push({...p,y:null});prev=p;}svg+=`<path class="plot-line" stroke="var(--teal)" d="${pathFor(means)}"/>`;}
    // Keep all line vertices; limit redundant circles in dense telemetry.
    const stride=Math.max(1,Math.ceil(valid.length/400));
    valid.forEach((p,i)=>{if(p.r.record_hold||(i%stride&&i!==valid.length-1&&!p.r.active_only))return;svg+=`<circle class="plot-dot${p.r.active_only?" active-marker":""}" fill="${p.r.active_only?"var(--panel)":"var(--warm)"}" cx="${px(p.x)}" cy="${py(p.y)}" r="${p.r.active_only?5:line?2.3:3}"><title>${p.r.active_only?"Latest active game · ":""}${esc(new Date(p.r.timestamp_utc).toLocaleString())}: ${esc(num(p.y,4))}${percent?"%":""}${p.r.seed!=null?` · seed ${esc(p.r.seed)}`:""}</title></circle>`;});
    return svg+"</svg>";
  }

  function drawCharts() {
    const data=state.data||{},cut=state.window==="all"?-Infinity:Date.now()-(state.window==="15m"?900000:3600000);
    const rows=(Array.isArray(data.series)?data.series:[]).filter(r=>stamp(r.timestamp_utc)!==null&&stamp(r.timestamp_utc)>=cut).sort((a,b)=>stamp(a.timestamp_utc)-stamp(b.timestamp_utc));
    const episodes=rows.filter(r=>r.event==="episode_end");
    const updates=rows.filter(r=>r.event==="update");
    const maximum=data.maximum_survival||{};
    const records=(Array.isArray(maximum.records)?maximum.records:[]).filter(r=>stamp(r.timestamp_utc)!==null).sort((a,b)=>stamp(a.timestamp_utc)-stamp(b.timestamp_utc));
    let recordRows=records.filter(r=>stamp(r.timestamp_utc)>=cut).map(r=>({...r,record_pieces:r.pieces}));
    const before=records.filter(r=>stamp(r.timestamp_utc)<cut).at(-1);
    if(before)recordRows.unshift({...before,timestamp_utc:new Date(cut).toISOString(),record_pieces:before.pieces,record_hold:true});
    if(finite(maximum.active_max_pieces)&&stamp(maximum.active_observed_utc)!==null&&stamp(maximum.active_observed_utc)>=cut)recordRows.push({timestamp_utc:maximum.active_observed_utc,record_pieces:maximum.active_max_pieces,active_only:true});
    const observed=stamp(data.latest?.timestamp_utc);
    $("maximum-chart").innerHTML=chart(recordRows,"record_pieces",{label:"PPO maximum survived pieces",step:true,holdTo:observed!==null&&observed>=cut?observed:null});
    put("maximum-value",num(maximum.current_max_pieces));
    put("maximum-detail",`Completed record ${num(maximum.completed_record_pieces)} · active ${num(maximum.active_max_pieces)} · ${num(maximum.completed_episodes)} completed PPO episodes`);
    $("survival-chart").innerHTML=chart(episodes,"episode_pieces",{label:"Completed training episodes",line:false,rolling:true});
    const perClear=episodes.map(r=>({...r,placements_per_clear:finite(r.episode_pieces)&&finite(r.episode_clears)&&r.episode_clears>0?r.episode_pieces/r.episode_clears:null}));
    $("per-clear-chart").innerHTML=chart(perClear,"placements_per_clear",{label:"Placements per clear",line:false,rolling:true});
    const lastUpdate=(Array.isArray(data.series)?data.series:[]).filter(r=>r.event==="update"&&finite(r.placements)&&finite(r.clears)).at(-1);
    put("per-clear-value",lastUpdate&&lastUpdate.clears>0?num(lastUpdate.placements/lastUpdate.clears):"—");
    const noClear=episodes.filter(r=>r.episode_clears===0).length;
    put("per-clear-detail",lastUpdate&&lastUpdate.clears>0?`Run average ${num(lastUpdate.placements)} placements ÷ ${num(lastUpdate.clears)} clears${noClear?` · ${num(noClear)} episodes without a clear omitted`:""}`:"Run average appears after the first clear");
    $("calibration-chart").innerHTML=chart(rows.filter(r=>["calibration_progress","calibration_game"].includes(r.event)),"calibration_game_pieces",{label:"Calibration observations",calibration:true});
    $("progress-chart").innerHTML=chart(updates,"placements",{label:"PPO placements",xkey:"runner_updates"});
    $("reference-kl-chart").innerHTML=chart(updates,"final_reference_kl",{label:"Reference KL measurements",minYMax:0.0001});
    $("entropy-chart").innerHTML=chart(updates,"entropy",{label:"Entropy measurements"});
    $("exploration-chart").innerHTML=chart(updates.filter(r=>finite(r.lower_primary_probability)),"lower_primary_probability",{label:"Exploration measurements",percent:true,line:false});
    $("saturation-chart").innerHTML=chart(updates.filter(r=>finite(r.h2_saturated_fraction)),"h2_saturated_fraction",{label:"Hidden saturation measurements",percent:true,line:false});
    put("history-note",data.series_sampled?`${num(data.series?.length)} of ${num(data.series_total_count)} events · download for full history`: `${num(data.series_total_count)} recorded events`);
  }

  function drawGames(data) {
    const games=Array.isArray(data.games)?data.games:[];
    put("games-note",games.length?`${games.length} reported environments · ${age(games[0].observed_utc)}`:"No environment snapshots reported yet");
    $("game-grid").innerHTML=games.length?games.map((g,i)=>`<article class="game-card"><div class="game-heading"><span class="game-index">ENV / ${String(i+1).padStart(2,"0")}</span><span class="game-status ${["calibration","optimizing"].includes(data.phase)?"waiting":""}">${data.phase==="optimizing"?"Paused":data.phase==="calibration"?"Queued for PPO":g.game_over?"Finished":"Reported"}</span></div><strong class="game-pieces">${num(g.pieces)}</strong><div class="game-caption">placements this episode</div><div class="game-stats"><span><strong>${num(g.hyperlayers)}</strong>Clears</span><span><strong>${num(g.max_height)}</strong>Height</span><span><strong>${num(g.holes)}</strong>Holes</span></div><p class="game-seed">Seed ${esc(g.seed??"—")} · episode ${num(g.episode)}</p></article>`).join(""):"<p class=\"empty-inline\">No training environments have been reported yet.</p>";
  }

  function render(data) {
    const p=data.progress||{},cal=data.calibration||{},latest=data.latest||{},best=data.best_selection||{};
    const phase=phases[data.phase]||phases.unknown;
    put("phase-badge",phase[0]);put("phase-title",phase[1]);put("phase-copy",phase[2]);
    document.querySelectorAll(".phase-steps li").forEach((el,i)=>{el.classList.toggle("done",phase[3]>i);el.classList.toggle("current",phase[3]===i);});
    let detail="Awaiting source events";
    if(data.phase==="calibration")detail=cal.stage==="fit"?`Critic fit · ${num(cal.fit_updates)} updates`:finite(cal.current_game)?`Game ${num(cal.current_game)} / ${num(cal.total_games)} · ${num(cal.current_pieces)} placements`:`${num(cal.finished_games)} / ${num(cal.total_games)} calibration games finished`;
    else if(data.phase==="training"||data.phase==="selection")detail=`${num(p.runner_updates)} rollout updates · ${num(p.finished_episodes)} completed episodes`;
    else if(data.phase==="complete")detail="Production stopped · review final assessment separately";
    else if(data.phase==="failed")detail="Inspect the event stream and controller diagnostics";
    else if(data.phase==="optimizing")detail="Graceful stop saved · preparing optimized sixteen-environment run";
    put("stage-detail",detail);
    put("placements",num(p.placements));put("updates",num(p.runner_updates));put("episodes",num(p.finished_episodes));
    put("optimizer-detail",`Optimizer steps ${num(p.optimizer_steps)}`);put("clears-detail",`Hyperlayers cleared ${num(p.clears)}`);
    put("best-survival",finite(best.mean_pieces)?num(best.mean_pieces,1):"—");
    put("best-detail",finite(best.mean_pieces)?"Mean placements across selection games":"Awaiting completed selection games");
    put("entropy-value",num(latest.entropy,4));put("kl-value",num(latest.kl,5));
    put("reference-kl-value",num(latest.final_reference_kl,5));put("reference-kl-weight",num(latest.reference_kl_weight,5));
    put("critic-ev-value",num(latest.critic_bootstrapped_explained_variance,4));
    put("exploration-value",finite(latest.lower_primary_probability)?`${num(latest.lower_primary_probability*100,2)}%`:"—");
    put("saturation-value",finite(latest.h2_saturated_fraction)?`${num(latest.h2_saturated_fraction*100,2)}%`:"—");
    const changed=latest.actor_changed_from_initial;
    put("actor-change",typeof changed==="boolean"?(changed?"Yes":"No"):"—");
    const calText=[cal.stage?`Stage: ${cal.stage.replaceAll("_"," ")}`:null,finite(cal.completed_games)?`${num(cal.completed_games)} completed`:null,finite(cal.censored_games)?`${num(cal.censored_games)} censored`:null,finite(cal.validation_mse)?`Validation MSE ${num(cal.validation_mse,3)}`:null].filter(Boolean);
    put("calibration-detail",calText.length?calText.join(" · "):"Calibration observations are separate from PPO episodes.");
    $("selection-section").hidden=!finite(best.mean_pieces);
    $("selection-details").className="selection-details";
    $("selection-details").innerHTML=[[best.mean_pieces,"Mean placements"],[best.minimum_pieces,"Minimum placements"],[best.median_pieces,"Median placements"]].map(([v,label])=>`<div class="selection-metric">${num(v,1)}<span>${esc(label)}</span></div>`).join("");
    const events=Array.isArray(data.events)?data.events:[];
    $("event-list").innerHTML=events.length?events.slice(-10).reverse().map(e=>{const detail=[[e.episode_pieces,"episode placements"],[e.calibration_game_pieces,"calibration placements"],[e.runner_updates,"rollout updates"],[e.selection_mean_pieces,"selection mean"]].filter(([v])=>finite(v)).map(([v,k])=>`${num(v,2)} ${k}`).join(" · ")||e.stop_reason||"Recorded source event";return `<div class="event-row"><time class="event-time" title="${esc(e.timestamp_utc)}">${local(e.timestamp_utc)}</time><span class="event-name">${esc(String(e.event||"Event").replaceAll("_"," "))}</span><span class="event-detail">${esc(detail)}</span></div>`;}).join(""):"<p class=\"empty-inline\">No production events have been recorded yet.</p>";
    drawGames(data);drawCharts();tick();
  }

  function tick() {
    const data=state.data,b=data?.budget||{};
    const deadline=stamp(b.deadline_utc),start=stamp(b.start_utc),active=["training","calibration","selection","optimizing"].includes(data?.phase);
    let remaining=b.remaining_seconds,elapsed=b.elapsed_seconds;
    if(active&&deadline!==null)remaining=Math.max(0,(deadline-Date.now())/1000);
    if(active&&start!==null)elapsed=Math.max(0,(Date.now()-start)/1000);
    $("remaining-time").innerHTML=`${clock(remaining)}<span class="clock-unit">remaining</span>`;
    put("elapsed-time",clock(elapsed));put("deadline",local(b.deadline_utc));$("deadline").title=b.deadline_utc||"Not reported";
    const duration=start!==null&&deadline!==null?(deadline-start)/1000:finite(remaining)&&finite(elapsed)?remaining+elapsed:null;
    const pct=finite(elapsed)&&finite(duration)&&duration>0?Math.min(100,elapsed/duration*100):null;
    $("budget-fill").style.width=`${pct??0}%`;
    if(pct===null){$("budget-progress").removeAttribute("aria-valuenow");$("budget-progress").setAttribute("aria-valuetext","Budget has not been reported");}else{$("budget-progress").setAttribute("aria-valuenow",pct.toFixed(1));$("budget-progress").setAttribute("aria-valuetext",`${num(pct,1)} percent of production budget consumed`);}
    put("budget-label",finite(remaining)?remaining<=0?"Budget exhausted":`${num(pct,1)}% consumed`:"Awaiting start");
    put("budget-copy",finite(remaining)?"Production wall time includes calibration, rollouts and selection.":"The production clock starts after comparison and preflight.");
    const source=data?.latest?.timestamp_utc;
    put("source-age",source?`Latest event ${age(source)}`:"No production events yet");put("signals-age",source?`Source ${age(source)}`:"Awaiting update");
    const seconds=state.received===null?null:(Date.now()-state.received)/1000;
    put("snapshot-age",state.received===null?"Awaiting first snapshot":`Snapshot ${Math.floor(seconds)}s ago`);
    $("connection").className=`connection ${state.error?"error":state.paused?"paused":seconds!==null&&seconds>45?"stale":""}`;
    put("connection-text",state.error?"Connection interrupted":state.paused?"Refresh paused":seconds===null?"Connecting":seconds>45?"Snapshot stale":"Live · localhost");
    const errors=[state.error,...(Array.isArray(data?.errors)?data.errors:[])].filter(Boolean);
    $("error-banner").hidden=!errors.length;$("error-banner").textContent=errors.slice(-3).map(x=>typeof x==="string"?x:JSON.stringify(x)).join(" · ");
  }

  async function refresh() {
    if(state.busy)return;state.busy=true;$("refresh").disabled=true;
    const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),8000);
    try{const response=await fetch("/api/status",{cache:"no-store",signal:abort.signal});if(!response.ok)throw Error(`HTTP ${response.status}`);const data=await response.json();if(!data||typeof data!=="object"||Array.isArray(data))throw Error("Invalid status snapshot");state.data=data;state.received=Date.now();state.error=null;render(data);}
    catch(error){state.error=`Status refresh failed (${error.name==="AbortError"?"8s timeout":error.message}). Previous data is retained.`;tick();}
    finally{clearTimeout(timer);state.busy=false;$("refresh").disabled=false;}
  }

  // Pure formatter/chart exports allow small offline fixture checks without a server.
  globalThis.TrainingDashboard={chart,num,clock,stamp};
  if(typeof document==="undefined")return;
  $("pause").addEventListener("click",()=>{state.paused=!state.paused;$("pause").setAttribute("aria-pressed",String(state.paused));put("pause",state.paused?"Resume":"Pause");if(!state.paused)refresh();tick();});
  $("refresh").addEventListener("click",refresh);
  document.querySelectorAll("[data-window]").forEach(button=>button.addEventListener("click",()=>{state.window=button.dataset.window;document.querySelectorAll("[data-window]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));drawCharts();}));
  drawGames({});drawCharts();refresh();setInterval(()=>{if(!state.paused)refresh();},15000);setInterval(tick,1000);
})();
