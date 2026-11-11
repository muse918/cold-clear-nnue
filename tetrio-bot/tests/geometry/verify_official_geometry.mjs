import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import {loadOfficialGeometryOracle,footprint} from './official_geometry_oracle.mjs';

if(!process.argv[2]||!process.argv[3])
  throw new Error('Usage: node verify_official_geometry.mjs /path/to/tetrio.js /path/to/geometry_tests [report.json]');
const official=loadOfficialGeometryOracle(process.argv[2]);
const empty=()=>Array(40).fill(0),inputs=[],expected=[],contexts=[];
let seed=0x918178;
const rng=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed>>>0;};
const counts={cells:0,collision:0,rotate:0,spin:0,spawn:0,placements:0,official_paths_replayed:0};
const rotationCoverage=Array.from({length:2},()=>Array.from({length:7},()=>({zero:0,kick:[0,0,0,0,0],blocked:0})));
const spinCoverage=Array.from({length:10},()=>[0,0,0]);
function add(input,oracle) {
  inputs.push(input);counts[input.op]++;
  const p=input.piece,q=input.pose;
  let result;
  if(input.op==='cells') result=oracle.cells(p,q);
  else if(input.op==='collision') result=!oracle.legal(p,q,false);
  else if(input.op==='rotate') {
    result=oracle.rotate(p,q,input.move);
    const c=rotationCoverage[input.kicks??0][p];
    if(!result)c.blocked++;else if(result.kick<0)c.zero++;else c.kick[result.kick]++;
  } else if(input.op==='spin') {
    result=oracle.spin(p,q,input.kick);spinCoverage[input.mode??6][result]++;
  } else if(input.op==='spawn') result=oracle.spawn(p,input.last_clear);
  else if(input.op==='placements') {
    result=oracle.placements(p,input.last_clear);
    contexts.push({index:inputs.length-1,oracle});
  } else throw new Error('Unknown op');
  expected.push(result);
}

const e=official.create(empty());
for(let piece=0;piece<7;piece++) for(let r=0;r<4;r++) {
  add({op:'cells',piece,pose:[3,10,r]},e);
  for(const x of [-3,-2,-1,0,7,8,9,10]) for(const y of [0,1,2,3,38,39,40,41])
    add({op:'collision',piece,pose:[x,y,r]},e);
}

// Collision-independent random boards create corner, wall and cavity fixtures.
// Each current mino is removed from the board, making rotation/spin queries
// exercise real occupied surroundings instead of mostly illegal starting poses.
for(let sample=0;sample<900;sample++) {
  const piece=sample%7,pose=[rng()%9-1,3+rng()%36,rng()%4],rows=empty();
  for(let y=Math.max(0,pose[1]-5);y<Math.min(40,pose[1]+4);y++)
    for(let x=0;x<10;x++) if(rng()%100<(sample%3===0?55:28))rows[y]|=1<<x;
  for(const [x,y] of e.cells(piece,pose)) if(x>=0&&x<10&&y>=0&&y<40) rows[y]&=~(1<<x);
  const base={rows,piece,pose};
  for(let kicks=0;kicks<2;kicks++) {
    const oracle=official.create(rows,{kicks});
    add({...base,op:'collision'},oracle);
    for(const move of [3,4,5]) add({...base,op:'rotate',kicks,move},oracle);
  }
  for(let mode=0;mode<10;mode++) {
    const oracle=official.create(rows,{mode});
    for(const kick of [-1,3]) add({...base,op:'spin',mode,kick,half:sample%2===0},oracle);
  }
}

// Pinned semantic regressions: T immobility without three corners, mini/full,
// quarter- and half-turn fourth-kick promotion, O immobility, and ceiling phase.
const fixtures=[];
let rows=empty();rows[3]=1<<4;rows[1]=(1<<2)|(1<<6);rows[0]=1<<4;
fixtures.push({rows,piece:2,pose:[3,2,0]});
rows=empty();rows[2]=1<<3;rows[0]=(1<<3)|(1<<5);
fixtures.push({rows,piece:2,pose:[3,2,0]});
rows=rows.slice();rows[2]|=1<<5;fixtures.push({rows,piece:2,pose:[3,2,0]});
rows=empty();rows[0]=(1<<2)|(1<<5);rows[2]=1<<3;
fixtures.push({rows,piece:1,pose:[3,1,0]});
rows=[0,0,58,844,80,331,726,520,669,...Array(31).fill(0)];
fixtures.push({rows,piece:2,pose:[4,8,1]});
for(const kicks of [0,1]) add({op:'rotate',rows,piece:2,pose:[4,6,3],move:5,kicks},official.create(rows,{kicks}));
for(const fixture of fixtures) for(let mode=0;mode<10;mode++) {
  const oracle=official.create(fixture.rows,{mode});
  for(const kick of [-1,3]) for(const half of [false,true])
    add({...fixture,op:'spin',mode,kick,half},oracle);
}
for(let piece=0;piece<7;piece++) for(let r=0;r<4;r++) for(const y of [37,38,39,40,41]) {
  for(const move of [3,4,5]) add({op:'rotate',piece,pose:[3,y,r],move},e);
}

// Actual ConsiderBlockout method, with clear/no-clear and enabled/disabled clutch.
for(let sample=0;sample<20;sample++) {
  const rows=empty();
  for(let y=18;y<Math.min(40,22+sample);y++)rows[y]=(1<<3)|(1<<4)|(1<<5)|(1<<6);
  for(let piece=0;piece<7;piece++) for(const clutch of [false,true]) for(const last_clear of [false,true]) {
    const oracle=official.create(rows,{clutch});
    add({op:'spawn',rows,piece,clutch,last_clear},oracle);
  }
}

// Complete placement sets and every emitted witness are checked on the actual
// primitives. These are geometric graphs without gravity or lock-reset budgets.
for(let sample=0;sample<12;sample++) {
  const rows=empty();
  for(let x=0;x<10;x++) {
    const height=sample===0?0:rng()%13;
    for(let y=0;y<height;y++) if(rng()%8)rows[y]|=1<<x;
  }
  const mode=sample<2?6:sample%10,kicks=sample%2,spin_180=sample%3!==1;
  const oracle=official.create(rows,{mode,kicks,spin_180});
  for(let piece=0;piece<7;piece++) add({op:'placements',rows,piece,mode,kicks,spin_180},oracle);
}

const run=spawnSync(process.argv[3],['--probe'],{input:inputs.map(j=>JSON.stringify(j)).join('\n')+'\n',encoding:'utf8',maxBuffer:256*1024*1024});
if(run.error||run.status!==0) throw new Error(`Native geometry probe failed: ${run.error??run.stderr}`);
const actual=run.stdout.trim().split('\n').map(s=>JSON.parse(s));
if(actual.length!==inputs.length)throw new Error(`Probe response count ${actual.length} differs from ${inputs.length}`);
let mismatchCount=0;
const mismatches=[];
const canonical=value=>JSON.stringify(value,(_key,item)=>item&&typeof item==='object'&&!Array.isArray(item)
  ? Object.fromEntries(Object.keys(item).sort().map(key=>[key,item[key]])) : item);
function mismatch(index,detail) {
  mismatchCount++;
  if(mismatches.length<8) mismatches.push({index,input:inputs[index],expected:expected[index],actual:actual[index],detail});
}
for(let index=0;index<inputs.length;index++) {
  if(inputs[index].op==='placements')continue;
  if(canonical(expected[index])!==canonical(actual[index]))mismatch(index,'primitive');
}
for(const {index,oracle} of contexts) {
  const input=inputs[index],piece=input.piece;
  const keys=ps=>ps.map(p=>footprint(oracle.cells(piece,p.pose),p.spin)).sort();
  if(JSON.stringify(keys(expected[index]))!==JSON.stringify(keys(actual[index])))mismatch(index,'placement set');
  for(const placement of actual[index]) {
    let pose=oracle.spawn(piece,input.last_clear),spin=0,kick=-1,half=false,last_rotation=false;
    let valid=!!pose;
    for(const move of placement.path) {
      if(!valid)break;
      if(move<=2) {
        pose=[pose[0]+(move===0?-1:move===1?1:0),pose[1]-(move===2?1:0),pose[2]];
        valid=oracle.legal(piece,pose);spin=0;kick=-1;half=false;last_rotation=false;
      } else {
        const rotation=oracle.rotate(piece,pose,move);
        if(!rotation) {valid=false;break;}
        pose=rotation.pose;kick=rotation.kick;half=move===5;last_rotation=true;
        spin=oracle.spin(piece,pose,kick);
      }
    }
    valid&&=JSON.stringify(pose)===JSON.stringify(placement.pose)&&spin===placement.spin&&
      kick===placement.kick&&half===placement.half&&last_rotation===placement.last_rotation&&
      !oracle.legal(piece,[pose[0],pose[1]-1,pose[2]]);
    if(!valid)mismatch(index,'official witness replay');
    counts.official_paths_replayed++;
  }
}
const report={...official.metadata,seed:'0x918178',checks:counts,primitive_and_set_cases:inputs.length,
  mismatches:mismatchCount,rotationCoverage,spinCoverage,
  limitations:['No frame gravity, key timing, lock-reset or movement-limit simulation.',
    'Canonical noninteger falling-y phase is conservative at the topmost buffer row; exact-integer frame phases can differ.',
    'Spawn-to-lock geometric reachability is checked; this does not prove a timing-feasible input script.']};
if(process.argv[4])fs.writeFileSync(process.argv[4],JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
if(mismatches.length) {
  console.error(JSON.stringify(mismatches,null,2));
  process.exitCode=1;
}
