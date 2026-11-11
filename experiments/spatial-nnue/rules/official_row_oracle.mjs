/**
 * Isolated row-transition oracle for the user-supplied TETR.IO 1.7.8 client.
 * No browser/client initialization, networking, account access, or rendering.
 * Inputs and outputs use bottom-first 10-bit rows; the extracted client is top-first.
 * This does not generate reachable moves or compute spin/attack/timing state.
 */
import fs from 'node:fs';
import vm from 'node:vm';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';
import readline from 'node:readline';

export const AUDITED_CLIENT_SHA256='6b682a1226dd2ae66ce19a08e944df92e7f175bccaa48cca7269bcd78f39a27a';

export function loadOfficialRowOracle(path) {
  const source=fs.readFileSync(path,'utf8');
  const sha256=crypto.createHash('sha256').update(source).digest('hex');
  if(sha256!==AUDITED_CLIENT_SHA256)throw new Error(`Client SHA-256 differs from the audited 1.7.8 build: ${sha256}. Re-audit extraction boundaries and primitive dependencies before updating the pin.`);
  const start=source.indexOf('F=function(e){');
  const end=source.indexOf('},I=function(e)',start);
  if(start<0||end<=start)throw new Error('Pinned board-factory boundaries not found');
  const factory=vm.runInNewContext(`(${source.slice(start+2,end+1)})`,Object.create(null),{timeout:1000});
  const metadata={sha256,factoryStartUtf16:start,factoryEndUtf16:end+1};
  const noop=()=>{};
  return {
    metadata,
    transition({rows, garbageMasks, addCells=[], garbageHoles=[]}) {
      if(!Array.isArray(rows)||rows.length!==40||rows.some(x=>!Number.isInteger(x)||x<0||x>1023))throw new Error('Expected 40 bottom-first 10-bit rows');
      const W=10,H=40;
      const e={W,H:20,B:20,T:H,
        gsm:{board:rows.map((m,y)=>Array.from({length:W},(_,x)=>!(m&(1<<x))?null:garbageMasks?.[y]&(1<<x)?'gb':'t')).reverse(),boardedges:Array.from({length:H},()=>Array(W).fill(255)),setoptions:{lineclear_are:0,garbageholesize:1,zenith_mods:[],usebombs:false},falling:{x:4},actors:{}},
        c:{OnClient:noop},am:{RemoveRow:noop,AddRow:noop,ProgressActors:noop},hm:{H:{}},wfm:{WaitFrames:noop}};
      e.bm=factory(e);
      for(const [x,y]of addCells){
        if(!Number.isInteger(x)||!Number.isInteger(y)||x<0||x>=W||y<0||y>=H||e.gsm.board[H-1-y][x]!==null)throw new Error('Invalid or overlapping added cell');
        e.gsm.board[H-1-y][x]='t';
      }
      const full=e.bm.GetFullLines();
      const clearedRows=Array.from(full,y=>H-1-y).sort((a,b)=>a-b);
      const garbageCleared=full.filter(y=>e.gsm.board[y].includes('gb')).length;
      e.bm.RemoveLinesFromStack(full);
      let inserted=0;
      for(const hole of garbageHoles){
        if(!Number.isInteger(hole)||hole<0||hole>=W)throw new Error('Invalid garbage hole');
        if(!e.bm.PushLine({column:hole}))break;
        inserted++;
      }
      const final=e.gsm.board.slice().reverse();
      return {rows:final.map(r=>r.reduce((m,t,x)=>m|(t!==null?1<<x:0),0)),garbageMasks:final.map(r=>r.reduce((m,t,x)=>m|(t==='gb'?1<<x:0),0)),clearedRows,garbageCleared,garbageRowsInserted:inserted};
    }
  };
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  if(!process.argv[2])throw new Error('Usage: node official_row_oracle.mjs /path/to/tetrio.js < events.jsonl');
  const oracle=loadOfficialRowOracle(process.argv[2]);
  for await(const line of readline.createInterface({input:process.stdin,crlfDelay:Infinity})){
    if(!line.trim())continue;
    const input=JSON.parse(line);
    process.stdout.write(JSON.stringify(oracle.transition(input))+'\n');
  }
}
