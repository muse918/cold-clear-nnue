import fs from "node:fs";
import path from "node:path";
import {execFileSync} from "node:child_process";
import {isDeepStrictEqual} from "node:util";
import {loadOfficialOracle} from "./official_battle_oracle.mjs";

const args=process.argv.slice(2);
function arg(name,fallback) {const at=args.indexOf(name);return at<0?fallback:args[at+1];}
const official=arg("--official",process.env.TETRIO_JS);
const executable=path.resolve(arg("--probe","tetrio-bot/build/battle_probe"));
const reportPath=arg("--report","tetrio-bot/tests/rules/battle_parity.json");
const oracle=loadOfficialOracle(official);
let seed=918_2026;
const random=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return(seed>>>0)/4294967296;};
const pick=a=>a[Math.floor(random()*a.length)];
const integer=n=>Math.floor(random()*n);
const cases=[];
for (let lines=0;lines<=4;lines++) for (const spin of ["none","mini","full"])
for (const combo of [0,1,2,3,4,8,16,32]) for (const b2b of [0,1,2,3,4,5,7,8,12,24])
for (const perfect_clear of lines?[false,true]:[false])
for (const garbage_rows of lines?[0,1]:[0]) {
  cases.push({clear:{lines,spin,perfect_clear,garbage_rows},state:{combo,b2b,pieces_placed:15},tank:false});
}
const grid=cases.length;
for (let i=0;i<12000;i++) {
  const lines=integer(5), hole=pick([null,0,1,4,7,9]);
  const options={
    b2bcharging:pick([true,false]),b2bchaining:pick([true,false]),b2bextras:pick([true,false]),
    b2bcharge_at:pick([0,2,4,6]),b2bcharge_base:pick([0,1,3]),
    allclears:pick([true,true,true,false]),allclear_garbage:pick([0,5,10]),allclear_b2b:pick([0,1,2]),
    allclear_b2b_dupes:pick([true,false]),allclear_b2b_sends:pick([true,false]),allclear_charges:pick([true,false]),
    spinbonuses:pick(["all-mini+","all","handheld"]),
    combotable:pick(["multiplier","classic guideline","modern guideline","none"]),
    garbagespecialbonus:pick([true,false]),garbagemultiplier:pick([0,.5,1,1.1,1.25,1.5,2,3.7]),
    cancelmultiplier:pick([.5,1,1.5,2]),roundmode:pick(["down","rng"]),
    garbageblocking:pick(["combo blocking","limited blocking","none"]),
    garbageattackcap:pick([0,0,4,8]),garbagecap:pick([0,2,8,12.5]),garbagecapmax:pick([4,40]),
    openerphase:pick([0,14]),hasgarbage:pick([true,true,true,false]),
    garbageholesize:pick([1,1,2,3]),messiness_change:pick([0,.3,1]),messiness_inner:pick([0,.1,1]),
    messiness_nosame:pick([true,false]),messiness_center:pick([true,false]),messiness_timeout:pick([0,20,100]),
    garbagetargetbonus:pick(["none","offensive","defensive"])
  };
  const state={
    combo:integer(40),b2b:integer(40),pieces_placed:pick([0,1,13,14,15,100]),total_sent:integer(30),
    frame:integer(200),last_tank:0,rng_seed:integer(2147483646)+1,last_hole:hole,
    hole_changed:hole!==null&&pick([false,true]),enemies:integer(8),garbage_bonus:integer(4),
    incoming:Array.from({length:integer(6)},(_,j)=>({id:j+1,amount:integer(15)+1,hole:pick([null,null,0,4,9]),
      width:pick([1,1,2,3]),active:pick([true,true,false]),ready:pick([true,true,false]),
      hardened:random()<.1,shielded:random()<.1}))
  };
  cases.push({options,state,clear:{lines,spin:pick(["none","mini","full"]),piece:pick(["t","i","o","l","j","s","z"]),
    perfect_clear:lines>0&&random()<.2,garbage_rows:integer(lines+1)}});
}
const battleCount=cases.length;
for (let i=0;i<512;i++) {
  const incoming=Array.from({length:integer(5)},(_,j)=>({id:j+1,amount:integer(8)+1,hole:pick([null,0,9]),
    width:pick([1,2,3]),active:pick([true,false]),ready:pick([true,false]),hardened:random()<.1,shielded:random()<.2}));
  cases.push({options:{receivemultiplier:pick([0,.5,1,1.25,2]),garbageabsolutecap:pick([0,4,8,20]),roundmode:pick(["down","rng"])},
    state:{incoming,rng_seed:integer(2147483646)+1},
    enqueue:{id:incoming.length+1,amount:integer(20)+1,hole:pick([null,0,9]),width:pick([1,2,3]),
      active:pick([true,false]),ready:pick([true,false]),hardened:random()<.1,shielded:random()<.1}});
}
const output=execFileSync(executable,[],{input:cases.map(x=>JSON.stringify(x)).join("\n")+"\n",
  encoding:"utf8",maxBuffer:64*1024*1024}).trim().split("\n").map(line=>JSON.parse(line));
if (output.length!==cases.length) throw new Error(`Probe output length ${output.length} != ${cases.length}`);
const mismatch=[];
let mismatchCount=0;
let packets=0,tankRows=0,cancellations=0;
for (let i=0;i<cases.length;i++) {
  let expected;
  try { expected=oracle.resolve(cases[i]); }
  catch(e) { throw new Error(`Official oracle failed on ${i}: ${e.message}`); }
  packets+=expected.outgoing.length;tankRows+=expected.tank.length;cancellations+=expected.cancelled;
  if (!isDeepStrictEqual(output[i],expected)) {
    ++mismatchCount;
    if (mismatch.length<8) mismatch.push({index:i,input:cases[i],expected,actual:output[i]});
  }
}
const report={
  client_sha256:oracle.hash,engine:19,fixture_seed:918_2026,
  deterministic_cases:grid,random_battle_cases:battleCount-grid,enqueue_cases:cases.length-battleCount,total_cases:cases.length,
  passed:cases.length-mismatchCount,mismatches:mismatchCount,
  outgoing_packet_comparisons:packets,tank_row_comparisons:tankRows,cancelled_line_comparisons:cancellations,
  scope:"Official AnnounceLines, AnnounceClear, FightLines, TakeAllDamage, AddPendingGarbage; all output packets, counters, incoming order and RNG state.",
  limitations:"Placement generation, transport acknowledgement reconciliation, frame/input timing, and live gameplay are separate tests.",
  mismatch
};
fs.mkdirSync(path.dirname(reportPath),{recursive:true});
fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+"\n");
console.log(JSON.stringify({...report,mismatch:undefined,report:reportPath},null,2));
if (mismatchCount) process.exitCode=1;
