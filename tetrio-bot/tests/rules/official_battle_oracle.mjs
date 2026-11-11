/**
 * Differential-test oracle for the user-supplied TETR.IO Beta 1.7.8 client.
 * This file contains a harness, not redistributed client code. The caller must
 * provide the original client with TETRIO_JS or --official <path>.
 * Only the original clear/attack factories execute; graphics/network are stubbed.
 */
import fs from "node:fs";
import crypto from "node:crypto";
import vm from "node:vm";
import { pathToFileURL } from "node:url";

export const CLIENT_SHA256 = "6b682a1226dd2ae66ce19a08e944df92e7f175bccaa48cca7269bcd78f39a27a";

function between(source, start, end) {
  const a = source.indexOf(start);
  if (a < 0) throw new Error(`Client locator not found: ${start}`);
  const b = source.indexOf(end, a + start.length);
  if (b < 0) throw new Error(`Client end locator not found: ${end}`);
  return source.slice(a + start.length, b);
}

function objectLiteral(source, marker) {
  const start = source.indexOf(marker) + marker.length;
  if (start < marker.length || source[start] !== "{") throw new Error(`Missing ${marker}`);
  let depth = 0, quoted = null, escaped = false;
  for (let i = start; i < source.length; i++) {
    const c = source[i];
    if (quoted) {
      if (escaped) escaped = false;
      else if (c === "\\") escaped = true;
      else if (c === quoted) quoted = null;
    } else if (c === '"' || c === "'") quoted = c;
    else if (c === "{") depth++;
    else if (c === "}" && --depth === 0) return source.slice(start, i + 1);
  }
  throw new Error(`Unclosed ${marker}`);
}

const optionNames = [
  "hasgarbage", "garbagefavor", "garbageholesize", "garbagemultiplier",
  "cancelmultiplier", "receivemultiplier", "garbageblocking", "garbagecap",
  "garbagecapmax", "garbageabsolutecap", "garbageattackcap", "garbageentry",
  "garbagespecialbonus", "garbagetargetbonus", "garbagephase", "garbagequeue",
  "garbageare", "garbagearebump", "garbagespeed", "garbageincrease",
  "garbagemargin", "garbagecapincrease", "garbagecapmargin", "openerphase",
  "passthrough", "roundmode", "messiness_change", "messiness_inner",
  "messiness_nosame", "messiness_center", "messiness_timeout", "spinbonuses",
  "combotable", "b2bcharging", "b2bchaining", "b2bcharge_at", "b2bcharge_base",
  "b2bextras", "allclears", "allclear_garbage", "allclear_b2b",
  "allclear_b2b_sends", "allclear_b2b_dupes", "allclear_charges", "are",
  "lineclear_are", "clutch", "nolockout", "usebombs"
];

function scalarDefaults(source, context) {
  const options = {};
  for (const name of optionNames) {
    const m = source.match(new RegExp(`(?:^|[,{}])${name}:\\{default:([^,}]+)`));
    if (!m) throw new Error(`Missing option default: ${name}`);
    options[name] = vm.runInContext(`(${m[1]})`, context);
  }
  const preset = source.match(/"tetra league":"([^\"]+)"/);
  if (!preset) throw new Error("Missing Tetra League preset");
  for (const assignment of preset[1].split(";")) {
    const m = assignment.match(/^options\.([^=]+)=(.*)$/);
    if (!m || !optionNames.includes(m[1])) continue;
    const raw = m[2];
    options[m[1]] = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(raw) ? Number(raw) : raw;
  }
  return options;
}

function noops() { return new Proxy({}, { get: () => () => {} }); }
function zeros() { return new Proxy({}, { get: (o, k) => o[k] ?? 0 }); }

export function loadOfficialOracle(path) {
  if (!path) throw new Error("Supply TETRIO_JS or --official <path>");
  const raw = fs.readFileSync(path);
  const hash = crypto.createHash("sha256").update(raw).digest("hex");
  if (hash !== CLIENT_SHA256) throw new Error(`Client hash mismatch: ${hash}`);
  const source = raw.toString("utf8");
  const context = vm.createContext({ Math, setTimeout: () => {} });
  const garbage = vm.runInContext(`(${objectLiteral(source, "garbage:")})`, context);
  const attackFactory = vm.runInContext(`(${between(source, "const z=", ";class T extends k{")})`, context);
  const clearFactory = vm.runInContext(`(${between(source, "const K=", ",Y=function(e)")})`, context);
  const Rng = vm.runInContext(`(${between(source, "class C", "const y=19") .replace(/;$/, "") .replace(/^/, "class C")})`, context);
  const defaults = scalarDefaults(source, context);

  function resolve(input = {}) {
    const st = input.state ?? {}, clear = input.clear ?? {};
    const options = {
      ...defaults, hasgarbage: true, zenith: false, zenith_mods: [], levels: false, zenlevels: false,
      ...input.options
    };
    const incoming = (st.incoming ?? []).map((p, i) => ({
      id: p.id ?? i + 1, type: "garbage", amt: p.amount ?? p.amt,
      size: p.width ?? p.size ?? 1, column: p.hole ?? p.column ?? null,
      active: p.active ?? true, status: p.status ?? (p.ready === false ? "caution" : "spawn"),
      queued: false, hardened: p.hardened ?? false, shielded: p.shielded ?? false
    }));
    const outgoing = [], tank = [];
    const t = {
      setoptions: options,
      stats: {
        combo: st.combo ?? 0, btb: st.b2b ?? 0, piecesplaced: st.pieces_placed ?? 1,
        lines: 0, score: 0, level: 1, topcombo: 0, topbtb: 0, tspins: 0,
        combopower: 0, btbpower: 0, clears: zeros(),
        garbage: { attack: 0, sent: st.total_sent ?? 0, sent_nomult: 0, cleared: 0 }
      },
      rngex: new Rng(st.rng_seed ?? 1),
      board: Array.from({length: 40}, () => Array(10).fill(null)),
      falling: {type: (clear.piece ?? "t").toLowerCase(), x: 4, y: 20, r: 0},
      hold: null, holdlocked: false, clearsbypiece: {}, quadcolumns: Array(10).fill(0),
      cancelstreak: 0, lasttanktime: st.last_tank ?? 0,
      lastcolumn: st.last_hole ?? null, haschangedcolumn: st.hole_changed ?? false,
      impendingdamage: incoming, garbageareentries: [], lastreceivedcount: 0,
      lastoffensive: {offence: 0, defense: 0, surge: 0, x: 0, y: 0},
      interactionid: 0, notyetreceivedattacks: 0,
      garbageid: incoming.reduce((maximum, p) => Math.max(maximum, p.id), 0),
      garbageacknowledgements: {incoming: {}, outgoing: {}},
      garbagebonus: st.garbage_bonus ?? 0, enemies: Array(st.enemies ?? 1).fill(0),
      zenith: {
        snapshotting: {current: {atk_out_send: 0, atk_out_cancel: 0}},
        columnsicknessbuckets_i: [], columnsicknessbuckets_lj: [],
        columnsicknessbuckets_sz: [], szsickness: [], spinbuckets: [], quadbuckets: []
      }
    };
    const e = {
      gsm: t, W: 10, H: 20, B: 20, T: 40, Wh: 5, Hh: 10,
      esm: {frame: st.frame ?? 0},
      c: {IsServer: () => true, OnClient: () => {}},
      cm: {constants: {garbage, scoring: zeros(), Strings: {}}},
      fm: {
        IsSpin: () => clear.spin === "full" || clear.spin === "mini",
        IsSpinMini: () => clear.spin === "mini", WasMoved: () => true
      },
      hm: {H: {holderstate: {ds: 0, dy: 0}}}, lm: {H: {lastdT: 0}},
      bm: {
        ColumnWidth: () => Math.max(0, 10 - (options.garbageholesize - 1)),
        AreWeToppedYet: () => {},
        PushLine: p => tank.push({hole: p.column, width: p.size ?? 1}),
        IsUncolored: () => false
      },
      tm: {GetTargets: () => ["opponent"]},
      elm: {emit: (event, data) => { if (event === "interaction") outgoing.push(data.amt); }},
      zthm: noops(), sfx: noops(), am: noops(), fim: noops(), gpm: noops()
    };
    e.atm = attackFactory(e);
    const lineManager = clearFactory(e);
    if ((clear.garbage_rows ?? 0) > 0) t.lasttanktime = e.esm.frame;
    let blocked = false;
    if (input.enqueue) {
      const p = input.enqueue;
      if (p.id != null) t.garbageid = p.id - 1;
      e.atm.AddPendingGarbage({
        cid: 1, amt: p.amount, column: p.hole ?? null, size: p.width ?? 1,
        active: p.active ?? false, delay: p.ready === false ? 1 : 0, queued: false,
        hardened: p.hardened ?? false, shielded: p.shielded ?? false
      });
    } else {
      const pc = !!clear.perfect_clear && options.allclears !== false && options.allclears !== 0;
      blocked = lineManager.AnnounceLines(clear.lines ?? 0, pc, clear.garbage_rows ?? 0);
      if (options.allclears !== false && options.allclears !== 0) lineManager.AnnounceClear(pc);
      if (!blocked && input.tank !== false) e.atm.TakeAllDamage();
    }
    return {
      combo: t.stats.combo, b2b: t.stats.btb,
      attack: t.stats.garbage.attack, sent: t.lastoffensive.offence,
      cancelled: t.lastoffensive.defense, received: t.lastreceivedcount,
      surge: t.lastoffensive.surge, outgoing, tank, blocked,
      total_sent: t.stats.garbage.sent, rng_seed: t.rngex.seed,
      last_hole: t.lastcolumn, hole_changed: t.haschangedcolumn,
      last_tank: t.lasttanktime,
      incoming: t.impendingdamage.map(p => ({
        id: p.id, amount: p.amt, hole: p.column, width: p.size,
        active: p.active, ready: p.status === "spawn",
        hardened: p.hardened, shielded: p.shielded
      }))
    };
  }
  return {resolve, defaults, hash, engine: 19};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2), index = args.indexOf("--official");
  const oracle = loadOfficialOracle(index >= 0 ? args[index + 1] : process.env.TETRIO_JS);
  const raw = fs.readFileSync(0, "utf8").trim();
  if (!raw) process.stdout.write(JSON.stringify({hash: oracle.hash, engine: 19, defaults: oracle.defaults}, null, 2) + "\n");
  else for (const line of raw.split(/\n/)) process.stdout.write(JSON.stringify(oracle.resolve(JSON.parse(line))) + "\n");
}
