/**
 * Isolated official-client geometry oracle. The supplied client is loaded only
 * as source text. Its actual board, spin, kick and clutch primitives run in VM
 * sandboxes; no client bootstrap, account, network or renderer is initialized.
 * The copyrighted client is deliberately not included in this repository.
 */
import fs from 'node:fs';
import vm from 'node:vm';
import crypto from 'node:crypto';

export const AUDITED_CLIENT_SHA256='6b682a1226dd2ae66ce19a08e944df92e7f175bccaa48cca7269bcd78f39a27a';
export const PIECES=['i','o','t','l','j','s','z'];
export const SPINS=['none','T-spins','T-spins+','all','all+','all-mini','all-mini+','mini-only','handheld','stupid'];

// These pinned fragments contain ordinary literals, quoted strings and
// templates. Fail closed on a different build rather than guess new boundaries.
function balanced(source,start) {
  let depth=0,quote=null,escaped=false;
  for(let i=start;i<source.length;i++) {
    const c=source[i];
    if(quote) {
      if(escaped) escaped=false;
      else if(c==='\\') escaped=true;
      else if(c===quote) quote=null;
    } else if(c==='"'||c==="'"||c==='`') quote=c;
    else if(c==='{') depth++;
    else if(c==='}'&&--depth===0) return source.slice(start,i+1);
  }
  throw new Error('Unbalanced pinned source fragment');
}

export function loadOfficialGeometryOracle(path) {
  const source=fs.readFileSync(path,'utf8');
  const sha256=crypto.createHash('sha256').update(source).digest('hex');
  if(sha256!==AUDITED_CLIENT_SHA256)
    throw new Error(`Unaudited client ${sha256}; re-audit extraction and rules before changing the pin`);
  const evaluate=text=>vm.runInNewContext(`(${text})`,Object.create(null),{timeout:1000});
  function literal(name) {
    const start=source.indexOf(`${name}:{`);
    if(start<0) throw new Error(`Missing pinned literal ${name}`);
    return evaluate(balanced(source,start+name.length+1));
  }
  function method(name,parameters) {
    const token=`${name}(${parameters}){`;
    const start=source.indexOf(token);
    if(start<0) throw new Error(`Missing pinned method ${token}`);
    return {start,fn:evaluate(`function(${parameters})${balanced(source,start+token.length-1)}`)};
  }
  const constants={tetrominoes:literal('tetrominoes'),kicksets:literal('kicksets'),
                   spinbonuses_rules:literal('spinbonuses_rules'),cornerTable:literal('cornerTable')};
  const boardStart=source.indexOf('F=function(e){');
  const boardEnd=source.indexOf('},I=function(e)',boardStart);
  if(boardStart<0||boardEnd<=boardStart) throw new Error('Pinned board factory not found');
  const boardFactory=evaluate(source.slice(boardStart+2,boardEnd+1));
  const spinEnd=source.indexOf('};ne.SpinRules=');
  const spinStart=source.lastIndexOf('ne=function(e){',spinEnd);
  if(spinStart<0||spinEnd<=spinStart) throw new Error('Pinned spin factory not found');
  const spinFactory=evaluate(source.slice(spinStart+3,spinEnd+1));
  const computeKick=method('ComputeKick','e,t,n,s,i');
  const considerBlockout=method('ConsiderBlockout','e,t=!1');
  const metadata={client:'TETR.IO Beta 1.7.8 / engine 19',sha256,
    extractedUtf16:{board:boardStart,spin:spinStart,computeKick:computeKick.start,considerBlockout:considerBlockout.start},
    coordinates:'10x40, bottom-first matrix anchor; rotations/spins use representative .96 falling-y phase'};

  function create(rows,{mode=6,kicks=0,spin_180=true,clutch=true,visible_height=20}={}) {
    if(rows.length!==40||rows.some(m=>!Number.isInteger(m)||m<0||m>1023))
      throw new Error('Expected forty 10-bit bottom-first rows');
    const noop=()=>{};
    const e={W:10,H:visible_height,B:40-visible_height,T:40,
      gsm:{board:rows.map(m=>Array.from({length:10},(_,x)=>m&(1<<x)?'t':null)).reverse(),
           setoptions:{spinbonuses:SPINS[mode]},falling:{}},
      c:{OnClient:noop},am:{},hm:{H:{}},wfm:{}};
    e.cm={constants,kickset:constants.kicksets[kicks===0?'SRS+':'SRS'],
      GetPieceOffset:()=>Array.from({length:4},()=>[0,0]),
      GetPieceKickset(piece) {
        const def=constants.tetrominoes[piece];
        if(def.disallow_kick&&!this.kickset.allow_o_kick) return false;
        return def.kickset_special ? this.kickset[`${def.kickset_special}_kicks`]??this.kickset.kicks : this.kickset.kicks;
      }};
    e.fm={GetPieceDefinition(piece,r) {
      const m=constants.tetrominoes[piece].matrix;
      return {data:m.data[r],dx:m.dx,dy:m.dy};
    },HasRotated:()=>true};
    e.bm=boardFactory(e);
    e.rm=spinFactory(e);
    const receiver={self:e,options:{infinite_movement:false,lockresets:15,clutch},
                    S:{totalRotations:0,lastwasclear:false,clutchCount:0},IsServer:()=>true};
    const type=piece=>typeof piece==='number'?PIECES[piece]:piece;
    function pivot(piece,pose,fraction=true) {
      const m=constants.tetrominoes[type(piece)].matrix;
      return [pose[0]+m.dx,39+m.dy-pose[1]-(fraction?.04:0),pose[2]];
    }
    function unpivot(piece,p) {
      const m=constants.tetrominoes[type(piece)].matrix;
      return [p.x-m.dx,39+m.dy-Math.ceil(p.y),p.r];
    }
    function cells(piece,pose) {
      return constants.tetrominoes[type(piece)].matrix.data[pose[2]]
        .map(([x,y])=>[pose[0]+x,pose[1]-y]);
    }
    function legal(piece,pose,fraction=true) {
      return e.bm.IsLegalAtPos(type(piece),...pivot(piece,pose,fraction),true);
    }
    function rotate(piece,pose,move) {
      if(!legal(piece,pose)||move===5&&!spin_180||move<3||move>5) return null;
      const delta=move===3?1:move===4?3:2;
      const [x,y,r]=pivot(piece,pose);
      e.bm.ClearIsLegalAtPosMemo();
      const result=computeKick.fn.call(receiver,type(piece),x,y,r,(r+delta)&3);
      return result ? {pose:unpivot(piece,result),kick:result.kick?.index??-1} : null;
    }
    function spin(piece,pose,kick=-1) {
      if(!legal(piece,pose)) return 0;
      const [x,y,r]=pivot(piece,pose);
      e.gsm.falling={type:type(piece),x,y,r,kick:kick===-1?0:kick};
      e.bm.ClearIsLegalAtPosMemo();
      const result=e.rm.IsTSpin();
      return result==='normal'?2:result==='mini'?1:0;
    }
    function spawn(piece,last_clear=false) {
      const m=constants.tetrominoes[type(piece)].matrix;
      const falling={type:type(piece),x:4,y:40-visible_height-2.04,hy:40-visible_height-2,r:0};
      receiver.S.lastwasclear=last_clear;
      e.bm.ClearIsLegalAtPosMemo();
      const blocked=considerBlockout.fn.call(receiver,falling);
      return blocked ? null : unpivot(piece,falling);
    }
    function placements(piece,last_clear=false) {
      const initial=spawn(piece,last_clear);
      if(!initial) return [];
      const nodes=[{pose:initial,spin:0}],seen=new Set([`${initial}:0`]),emitted=new Set(),result=[];
      const add=(pose,spin)=>{
        const key=`${pose}:${spin}`;
        if(!seen.has(key)) { seen.add(key);nodes.push({pose,spin}); }
      };
      for(let head=0;head<nodes.length;head++) {
        if(nodes.length>20000) throw new Error('Unexpected unbounded placement graph');
        const node=nodes[head],p=node.pose;
        if(!legal(piece,[p[0],p[1]-1,p[2]])) {
          const key=footprint(cells(piece,p),node.spin);
          if(!emitted.has(key)) {emitted.add(key);result.push({pose:p,spin:node.spin});}
        }
        for(const [dx,dy] of [[-1,0],[1,0],[0,-1]]) {
          const q=[p[0]+dx,p[1]+dy,p[2]];
          if(legal(piece,q)) add(q,0);
        }
        for(const move of [3,4,5]) {
          const rotated=rotate(piece,p,move);
          if(rotated) add(rotated.pose,spin(piece,rotated.pose,rotated.kick));
        }
      }
      return result;
    }
    return {cells,legal,rotate,spin,spawn,placements};
  }
  return {metadata,create};
}

export function footprint(cells,spin) {
  return `${spin}:${cells.map(([x,y])=>y*10+x).sort((a,b)=>a-b).join(':')}`;
}
