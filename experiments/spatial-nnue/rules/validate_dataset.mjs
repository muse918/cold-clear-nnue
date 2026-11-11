// Dataset geometry checks against isolated primitives from the uploaded client.
// Usage: node validate_dataset.mjs records.bin /path/to/tetrio.js [output.json] [--all]
// Default: at most 10,000 evenly spaced records plus every >=3-line clear and garbage event.
import fs from 'node:fs';
import vm from 'node:vm';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { loadOfficialRowOracle } from './official_row_oracle.mjs';

const [datasetPath,officialPath,outputArg,...flags]=process.argv.slice(2);
if(!datasetPath||!officialPath)throw new Error('Usage: node validate_dataset.mjs records.bin /path/to/tetrio.js [output.json] [--all]');
const outputPath=outputArg&&!outputArg.startsWith('--')?outputArg:new URL('dataset_parity.json',import.meta.url).pathname;
const all=flags.includes('--all')||outputArg==='--all';
const bytes=fs.readFileSync(datasetPath),SIZE=240,W=10,H=40,FULL=1023;
assert.equal(bytes.length%SIZE,0,'Truncated 240-byte record');
const count=bytes.length/SIZE;
const oracle=loadOfficialRowOracle(officialPath);
const source=fs.readFileSync(officialPath,'utf8');
const shapeStart=source.indexOf('tetrominoes:{')+'tetrominoes:'.length;
const shapeEnd=source.indexOf(',tetrominoes_color:',shapeStart);
assert(shapeStart>=12&&shapeEnd>shapeStart,'Official shape-table boundaries missing');
const officialShapes=vm.runInNewContext(`(${source.slice(shapeStart,shapeEnd)})`,Object.create(null),{timeout:1000});
const normalize=cells=>{
  const x0=Math.min(...cells.map(c=>c[0])),y0=Math.min(...cells.map(c=>c[1]));
  return cells.map(([x,y])=>[x-x0,y-y0]).sort((a,b)=>a[1]-b[1]||a[0]-b[0]).map(c=>c.join(',')).join(';');
};
const shapeSets=['i','o','t','l','j','s','z'].map(p=>new Set(officialShapes[p].matrix.data.map(cells=>normalize(cells.map(([x,y])=>[x,-y])))));
const select=new Set();
if(all){for(let i=0;i<count;i++)select.add(i);}else{
  const n=Math.min(10000,count);
  for(let j=0;j<n;j++)select.add(n===1?0:Math.floor(j*(count-1)/(n-1)));
  for(let i=0;i<count;i++){const o=i*SIZE;if(bytes[o+10]===1||bytes[o+14]>0||bytes[o+13]>=3)select.add(i);}
}
const indices=[...select].sort((a,b)=>a-b);
const stat={locks:0,garbage:0,combinedLockGarbage:0,garbageRowsInserted:0,clearHistogram:[0,0,0,0,0],allRareClearsIncluded:0,allGarbageIncluded:0,shapeChecks:0,verticalPathChecks:0,groundContactChecks:0,rowMapChecks:0};
const failures=[];
const started=performance.now();
const compareRows=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
for(const index of indices){
  const o=index*SIZE,event=bytes[o+10],piece=bytes.readInt8(o+11),clears=bytes[o+13],garbage=bytes[o+14];
  const parent=Array.from({length:H},(_,y)=>bytes.readUInt16LE(o+32+2*y));
  const child=Array.from({length:H},(_,y)=>bytes.readUInt16LE(o+112+2*y));
  const rowSource=Array.from({length:H},(_,y)=>bytes.readInt8(o+192+y));
  const clearMask=bytes.readBigUInt64LE(o+16);
  try{
    assert(parent.every(x=>x<=FULL)&&child.every(x=>x<=FULL),'Invalid row bits');
    assert(parent.every(x=>x!==FULL),'Parent already contains a full row');
    assert(garbage<=H,'Invalid garbage amount');
    const holes=[];
    // Each PushLine inserts a new bottom row. Feed the desired bottom rows in
    // reverse order, including when a lock/clear is followed by a garbage rise.
    for(let y=garbage-1;y>=0;y--){
      const mask=(~child[y])&FULL;
      assert(mask!==0&&(mask&(mask-1))===0,'Ordinary garbage row needs exactly one hole');
      holes.push(Math.log2(mask));
    }
    let result,expectedMap;
    if(event===0){
      const cells=Array.from({length:4},(_,j)=>bytes.readUInt16LE(o+232+2*j)).map(n=>[n%W,Math.floor(n/W)]);
      assert.equal(new Set(cells.map(c=>c.join(','))).size,4,'Repeated placed cell');
      assert(piece>=0&&piece<7,'Invalid piece id');
      assert(shapeSets[piece].has(normalize(cells)),'Footprint is not an official rotation of the declared piece');
      stat.shapeChecks++;
      const maxY=Math.max(...cells.map(c=>c[1]));
      for(let dy=0;dy<=H-1-maxY;dy++)for(const [x,y]of cells){
        assert(x>=0&&x<W&&y>=0&&y+dy<H,'Out-of-board path');
        assert(!(parent[y+dy]&(1<<x)),'Vertical hard-drop path intersects parent board');
      }
      stat.verticalPathChecks++;
      assert(cells.some(([x,y])=>y===0||(parent[y-1]&(1<<x))),'Piece is not grounded');
      stat.groundContactChecks++;
      result=oracle.transition({rows:parent,addCells:cells,garbageHoles:holes});
      expectedMap=Array.from({length:H},(_,y)=>y).filter(y=>!(clearMask&(1n<<BigInt(y))));
      while(expectedMap.length<H)expectedMap.push(-1);
      stat.locks++;stat.clearHistogram[clears]++;
      if(garbage)stat.combinedLockGarbage++;
      if(clears>=3)stat.allRareClearsIncluded++;
    }else if(event===1){
      assert(garbage>0&&garbage<=H,'Invalid garbage amount');
      assert.equal(piece,-1,'Garbage record piece id');
      result=oracle.transition({rows:parent,garbageHoles:holes});
      expectedMap=Array.from({length:H},(_,y)=>y);
      stat.garbage++;
    }else throw new Error(`Unknown event ${event}`);
    if(garbage){
      expectedMap=Array(garbage).fill(-1).concat(expectedMap.slice(0,H-garbage));
      stat.allGarbageIncluded++;
    }
    assert.equal(result.garbageRowsInserted,garbage,'Official insertion refused a row');
    stat.garbageRowsInserted+=result.garbageRowsInserted;
    assert(compareRows(result.rows,child),'Official child occupancy differs');
    assert.equal(result.clearedRows.length,clears,'Official clear count differs');
    const actualMask=result.clearedRows.reduce((m,y)=>m|(1n<<BigInt(y)),0n);
    assert.equal(actualMask,clearMask,'Official clear mask differs');
    assert(compareRows(rowSource,expectedMap),'Stored row_source differs from stable compaction mapping');
    stat.rowMapChecks++;
  }catch(err){
    failures.push({record:index,game:bytes.readUInt32LE(o),parentId:bytes.readUInt32LE(o+4),step:bytes.readUInt16LE(o+8),event,piece,clears,garbage,error:err.message});
    if(failures.length>=20)break;
  }
}
const result={
  dataset:datasetPath,datasetSha256:crypto.createHash('sha256').update(bytes).digest('hex'),recordBytes:SIZE,totalRecords:count,
  official:oracle.metadata,officialVersion:'1.7.8',officialEngineVersion:19,
  officialShapeTable:{startUtf16:shapeStart,endUtf16:shapeEnd},
  dimensions:{width:W,visibleHeight:20,bufferHeight:20,totalHeight:H},
  selection:all?'all records':'10,000 evenly spaced indices plus all >=3-line-clear and garbage records',
  selectedRecords:indices.length,validatedRecords:stat.rowMapChecks,failures,...stat,
  elapsedSeconds:(performance.now()-started)/1000,
  scope:'Exact board occupancy/row-clear/mapping/ordinary-garbage primitives; piece footprints from official shape table; collision-free vertical descent from upper buffer in a fixed orientation. No SRS path from actual spawn, hold, spin, attacks, cancellation, timing, topout, or gameplay-strength validation.'
};
fs.writeFileSync(outputPath,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
if(failures.length)process.exitCode=1;
