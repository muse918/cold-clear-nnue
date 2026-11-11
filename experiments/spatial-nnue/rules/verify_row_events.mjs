// Offline, dependency-free checks. Does not load or run the browser client.
// Extracts only the uploaded board factory and executes selected row methods
// with rendering, actors, and timers replaced by no-op stubs.
import fs from 'node:fs';
import vm from 'node:vm';
import crypto from 'node:crypto';
import { stripTypeScriptTypes } from 'node:module';
import assert from 'node:assert/strict';
import { AUDITED_CLIENT_SHA256 } from './official_row_oracle.mjs';

if (!process.argv[2]) throw new Error('Usage: node verify_row_events.mjs /path/to/tetrio.js');
const officialPath = process.argv[2];
const source = fs.readFileSync(officialPath, 'utf8');
assert.equal(crypto.createHash('sha256').update(source).digest('hex'),AUDITED_CLIENT_SHA256,'Client differs from the audited build; re-audit before updating the pin');
const start = source.indexOf('F=function(e){');
const end = source.indexOf('},I=function(e)', start);
assert(start >= 0 && end > start, 'Expected pinned board factory boundaries');
const factoryText = source.slice(start + 2, end + 1);
const factory = vm.runInNewContext(`(${factoryText})`, Object.create(null), { timeout: 1000 });
const W = 10, H = 40;
const noop = () => {};
function official(bottomRows) {
  const e = {
    W, H: 20, B: 20, T: H,
    gsm: {
      board: bottomRows.map(r => [...r]).reverse(),
      boardedges: Array.from({length:H}, () => Array(W).fill(255)),
      setoptions: {lineclear_are: 0, garbageholesize: 1, zenith_mods: [], usebombs: false},
      falling: {x: 4}, actors: {},
    },
    c: {OnClient: noop}, am: {RemoveRow: noop, AddRow: noop, ProgressActors: noop},
    hm: {H: {}}, wfm: {WaitFrames: noop}
  };
  e.bm = factory(e);
  return e;
}
const Mino = {GARBAGE:'gb', BOMB:'bomb'};
const triangleRaw = fs.readFileSync(new URL('triangle_board.ts', import.meta.url), 'utf8');
const trianglePrepared = triangleRaw
  .replace('import { Mino } from "../queue";', `const Mino = ${JSON.stringify(Mino)};`)
  .replace('export * from ".";', '');
const triangleJS = stripTypeScriptTypes(trianglePrepared, {mode:'transform'}).replace(/\bexport\s+/g, '');
const TriangleBoard = vm.runInNewContext(`${triangleJS}\nBoard`, Object.create(null), {timeout:1000});
const fixedJS = triangleJS.replace('this.state.splice(this.fullHeight - amount - 1, amount);', 'this.state.splice(this.fullHeight, amount);');
assert.notEqual(fixedJS, triangleJS, 'Expected pinned truncation expression');
const FixedTriangleBoard = vm.runInNewContext(`${fixedJS}\nBoard`, Object.create(null), {timeout:1000});
function triangle(bottomRows, Constructor=TriangleBoard) {
  const b = new Constructor({width:W, height:20, buffer:20});
  b.state = bottomRows.map(r => r.map(t => t === null ? null : {mino:t==='gbd'?'bomb':t, connections:0}));
  return b;
}
const asBottom = e => e.gsm.board.map(r=>[...r]).reverse();
const asTiles = b => b.state.map(r => r.map(t => t?.mino==='bomb'?'gbd':t?.mino ?? null));
const normalized = x => JSON.stringify(x);
let rngState = 0x918;
const random = () => {rngState ^= rngState<<13; rngState ^= rngState>>>17; rngState ^= rngState<<5;return (rngState>>>0)/4294967296;};
const rand = n => Math.floor(random()*n);
const rows = () => Array.from({length:H}, () => Array.from({length:W}, () => random()<0.42?null:random()<0.2?'gb':'t'));
let clearCases=0;
for(let i=0;i<1000;i++) {
  const input = rows();
  for(let j=0;j<rand(5);j++) input[rand(H)] = Array.from({length:W},()=>random()<0.4?'gb':'i');
  if(i%3===0) input[rand(H)][rand(W)]='gbd';
  const e=official(input), b=triangle(input);
  const originalTopRows=e.gsm.board.slice();
  const full=e.bm.GetFullLines();
  const fullSet=new Set(full);
  const expectedSurvivors=originalTopRows.filter((_,idx)=>!fullSet.has(idx));
  const reference=Array.from({length:full.length},()=>Array(W).fill(null)).concat(expectedSurvivors);
  const garbageCount=full.filter(y => e.gsm.board[y].includes('gb')).length;
  e.bm.RemoveLinesFromStack(full);
  assert.equal(normalized(e.gsm.board),normalized(reference));
  for(let y=full.length;y<H;y++) assert.equal(e.gsm.board[y],expectedSurvivors[y-full.length]);
  const result=b.clearLines();
  assert.equal(result.lines,full.length);
  assert.equal(result.garbageCleared,garbageCount);
  assert.equal(normalized(asBottom(e)),normalized(asTiles(b)));
  clearCases++;
}
let garbageCases=0, garbageMismatches=0, fixedGarbageMatches=0, firstGarbageMismatch=null;
for(let i=0;i<1000;i++) {
  const input=rows();
  for(let y=30;y<H;y++) input[y]=Array(W).fill(null);
  // Half of cases exercise occupied buffer rows, important for exact parity.
  if(i%2===0) for(let y=30;y<H;y++) input[y][(y+i)%W]='t';
  const column=rand(W),amount=1+rand(8),e=official(input),b=triangle(input),fixed=triangle(input,FixedTriangleBoard);
  for(let j=0;j<amount;j++) e.bm.PushLine({column});
  b.insertGarbage({amount,size:1,column,bombs:false,isBeginning:true,isEnd:true});
  fixed.insertGarbage({amount,size:1,column,bombs:false,isBeginning:true,isEnd:true});
  const expected=asBottom(e),actual=asTiles(b);
  assert.equal(normalized(expected),normalized(asTiles(fixed)));
  fixedGarbageMatches++;
  if(normalized(expected)!==normalized(actual)) {
    garbageMismatches++;
    if(!firstGarbageMismatch) firstGarbageMismatch={amount,column,differingRows:expected.map((r,y)=>normalized(r)===normalized(actual[y])?null:y).filter(x=>x!==null)};
  }
  garbageCases++;
}
const minimalGarbageCases=[];
for(const [name,y,amount] of [
  ['low_marker_one_row',0,1],['low_marker_eight_rows',0,8],
  ['visible_ceiling_eight_rows',19,8],['first_lost_row_at_cap8',23,8],
  ['first_lost_row_at_cap1',37,1],['top_buffer_boundary',39,1]
]) {
  const input=Array.from({length:H},()=>Array(W).fill(null));
  input[y][0]='t';
  const e=official(input),b=triangle(input),fixed=triangle(input,FixedTriangleBoard);
  for(let j=0;j<amount;j++)e.bm.PushLine({column:9});
  const args={amount,size:1,column:9,bombs:false,isBeginning:true,isEnd:true};
  b.insertGarbage(args);fixed.insertGarbage(args);
  const locate=board=>board.flatMap((row,y)=>row.includes('t')?[y]:[]);
  const expected=asBottom(e),actual=asTiles(b);
  assert.equal(normalized(expected),normalized(asTiles(fixed)));
  minimalGarbageCases.push({name,oldMarkerRow:y,amount,officialMarkerRows:locate(expected),triangleMarkerRows:locate(actual),matches:normalized(expected)===normalized(actual)});
}
let lowStackGarbageCases=0;
for(let height=0;height<=20;height++)for(let amount=1;amount<=8;amount++) {
  const input=Array.from({length:H},(_,y)=>Array.from({length:W},(_,x)=>y<height&&x<5?'t':null));
  const e=official(input),b=triangle(input);
  for(let j=0;j<amount;j++)e.bm.PushLine({column:9});
  b.insertGarbage({amount,size:1,column:9,bombs:false,isBeginning:true,isEnd:true});
  assert.equal(normalized(asBottom(e)),normalized(asTiles(b)));
  lowStackGarbageCases++;
}
const output={
  official:{path:officialPath,sha256:crypto.createHash('sha256').update(source).digest('hex'),version:'1.7.8',clientCommit:'e84ebccd2',engineVersion:19,factoryStartCharacter:start,factoryEndCharacter:end+1},
  triangle:{commit:'aa0698ab8b4299b85c5f87e9b54793e1b7d9d380',file:'src/engine/board/index.ts'},
  scope:'Exact isolated row primitives, arbitrary board states; no complete game, movement, replay, timing, or attack equivalence claim.',
  clearCases,clearMatches:clearCases,clearOrderAndSurvivorIdentityMatches:clearCases,
  garbageCases,garbageMismatches,firstGarbageMismatch,fixedGarbageMatches,
  minimalGarbageCases,lowStackGarbageCases,lowStackGarbageMatches:lowStackGarbageCases,
  fixedGarbageChange:'local experimental replacement only: splice(fullHeight - amount - 1, amount) -> splice(fullHeight, amount)'
};
console.log(JSON.stringify(output,null,2));
