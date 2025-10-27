import React, { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectTrigger, SelectContent, SelectValue, SelectItem } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { motion } from "framer-motion";
import { FlaskConical, FireExtinguisher, Flame, ThermometerSun, Sparkles, Calculator, CheckCircle2, AlertCircle } from "lucide-react";

/**
 * ChemReactions Interactive Simulator (Web)
 * ---------------------------------------------------------
 * This single-file React app gives students a hands-on way to explore:
 * 1) Balancing chemical equations
 * 2) Classifying reaction types (precipitation, acid–base, gas evolution, redox, combustion)
 * 3) Limiting reactant, theoretical yield & percent yield
 * 4) Heat transfer (ΔH, exothermic vs endothermic) visualised
 *
 * Notes
 * - Uses shadcn/ui components (Card, Tabs, etc.) and Tailwind for styling.
 * - All computations run locally in the browser.
 * - Several common reactions are preloaded, but you can also type your own for the balancer.
 */

// --- Utilities: Chemistry helpers ---

// Very small periodic table for example molar masses (g/mol). Add more as needed.
const MOLAR_MASS: Record<string, number> = {
  H: 1.008,
  C: 12.011,
  N: 14.007,
  O: 15.999,
  Na: 22.990,
  Mg: 24.305,
  Al: 26.982,
  S: 32.06,
  Cl: 35.45,
  K: 39.098,
  Ca: 40.078,
  Ba: 137.327,
  Pb: 207.2,
  Ag: 107.8682,
  I: 126.90447,
  Li: 6.94,
};

// Regex to parse formula pieces (e.g., Ca(OH)2 -> Ca, O, H with counts)
function parseFormula(formula: string): Record<string, number> {
  // Remove state labels like (s),(l),(g),(aq)
  formula = formula.replace(/\((s|l|g|aq)\)/g, "");
  // Expand parentheses recursively
  function expand(form: string): string {
    const paren = /\(([^()]+)\)(\d*)/g;
    while (paren.test(form)) {
      form = form.replace(paren, (_, inner, times) => {
        const t = Number(times || 1);
        return inner
          .replace(/([A-Z][a-z]?)(\d*)/g, (__, el, n) => `${el}${Number(n || 1) * t}`)
          .replace(/([A-Z][a-z]?)(?=[A-Z(]|$)/g, "$1");
      });
    }
    return form;
  }
  const flat = expand(formula);
  const counts: Record<string, number> = {};
  flat.replace(/([A-Z][a-z]?)(\d*)/g, (_, el, n) => {
    counts[el] = (counts[el] || 0) + Number(n || 1);
    return "";
  });
  return counts;
}

// Compute molar mass of a chemical formula string using MOLAR_MASS map
function molarMass(formula: string): number {
  const parts = parseFormula(formula);
  let mass = 0;
  for (const el of Object.keys(parts)) {
    if (!(el in MOLAR_MASS)) throw new Error(`Molar mass for element ${el} not in table`);
    mass += MOLAR_MASS[el] * parts[el];
  }
  return mass;
}

// Gaussian elimination to solve linear system Ax=0 (stoichiometric balancing)
function balanceEquation(reactants: string[], products: string[]): number[] | null {
  // Collect unique elements
  const elementSet = new Set<string>();
  const species = [...reactants, ...products];
  const speciesParsed = species.map((s) => parseFormula(s));
  speciesParsed.forEach((sp) => Object.keys(sp).forEach((e) => elementSet.add(e)));
  const elements = Array.from(elementSet);

  const m = elements.length;
  const n = species.length;
  // Build matrix A (m x n): reactants positive, products negative -> Ax = 0
  const A = Array.from({ length: m }, (_, i) => Array(n).fill(0));
  elements.forEach((el, i) => {
    reactants.forEach((r, j) => (A[i][j] = speciesParsed[j][el] || 0));
    products.forEach((p, k) => (A[i][reactants.length + k] = - (speciesParsed[reactants.length + k][el] || 0)));
  });

  // Solve integer nullspace vector (simple approach: add constraint x_n = 1 and solve least squares)
  // We'll set last variable to 1 and solve A' x' = b.
  const vars = n - 1;
  const Aprime = A.map((row) => row.slice(0, vars));
  const b = A.map((row) => -row[n - 1]);

  // Solve Aprime x = b via Gaussian elimination
  function gaussianSolve(mat: number[][], rhs: number[]): number[] | null {
    const r = mat.length, c = mat[0].length;
    // Augmented matrix
    const M = mat.map((row, i) => [...row, rhs[i]]);
    let row = 0;
    for (let col = 0; col < c && row < r; col++) {
      // Find pivot
      let pivot = row;
      for (let i = row; i < r; i++) if (Math.abs(M[i][col]) > Math.abs(M[pivot][col])) pivot = i;
      if (Math.abs(M[pivot][col]) < 1e-12) continue; // no pivot in this column
      [M[row], M[pivot]] = [M[pivot], M[row]];
      // Normalize
      const div = M[row][col];
      for (let j = col; j <= c; j++) M[row][j] /= div;
      // Eliminate
      for (let i = 0; i < r; i++) if (i !== row) {
        const factor = M[i][col];
        for (let j = col; j <= c; j++) M[i][j] -= factor * M[row][j];
      }
      row++;
    }
    // Extract solution (last column)
    const x = Array(c).fill(0);
    for (let i = 0; i < c; i++) x[i] = 0;
    // Back-substitution from RREF
    for (let i = 0; i < r; i++) {
      const lead = M[i].findIndex((v, idx) => idx < c && Math.abs(v - 1) < 1e-9);
      if (lead !== -1) x[lead] = M[i][c];
    }
    if (x.some((v) => isNaN(v))) return null;
    return x;
  }

  const sol = gaussianSolve(Aprime, b);
  if (!sol) return null;
  const withLast = [...sol, 1];
  // Scale to smallest integers
  const denomLCM = (arr: number[]) => {
    const fracs = arr.map((x) => {
      const s = x.toString();
      if (!s.includes(".")) return 1;
      const dec = s.split(".")[1].length;
      return 10 ** dec;
    });
    const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : Math.abs(a));
    const lcm = (a: number, b: number) => (a * b) / gcd(a, b);
    return fracs.reduce((acc, f) => lcm(acc, f), 1);
  };
  const lcm = denomLCM(withLast);
  const scaled = withLast.map((x) => Math.round(x * lcm + 1e-9));
  // Make all positive
  const sign = scaled.some((v) => v < 0) ? -1 : 1;
  const ints = scaled.map((v) => Math.abs(v * sign));
  // Reduce by GCD
  const g = ints.reduce((acc, v) => {
    const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : Math.abs(a));
    return gcd(acc, v);
  });
  return ints.map((v) => v / g);
}

// Pretty print equation with coefficients
function prettyEquation(reactants: string[], products: string[], coeffs: number[]) {
  const ra = reactants.map((r, i) => `${coeffs[i] === 1 ? "" : coeffs[i] + " "}${r}`).join(" + ");
  const pa = products.map((p, i) => `${coeffs[reactants.length + i] === 1 ? "" : coeffs[reactants.length + i] + " "}${p}`).join(" + ");
  return `${ra} → ${pa}`;
}

// Built-in example reactions with metadata
const PRESET_REACTIONS = [
  {
    id: "combust_methane",
    name: "Combustion: CH4 + O2 → CO2 + H2O",
    reactants: ["CH4(g)", "O2(g)"], // FIX: comma (was a stray semicolon)
    products: ["CO2(g)", "H2O(l)"],
    classes: ["Combustion", "Redox"],
    enthalpy_kJ: -802.3,
    notes: "Exothermic; flame/heat/light produced.",
  },
  {
    id: "precipitation_lead_iodide",
    name: "Precipitation: Pb(NO3)2 + NaI → PbI2(s) + NaNO3",
    reactants: ["Pb(NO3)2(aq)", "NaI(aq)"],
    products: ["PbI2(s)", "NaNO3(aq)"],
    classes: ["Double displacement", "Precipitation"],
    enthalpy_kJ: undefined,
    notes: "Bright yellow PbI2(s) precipitate forms.",
  },
  {
    id: "acid_base_neutralisation",
    name: "Acid–base: HCl + NaOH → H2O + NaCl",
    reactants: ["HCl(aq)", "NaOH(aq)"],
    products: ["H2O(l)", "NaCl(aq)"],
    classes: ["Double displacement", "Acid–base"],
    enthalpy_kJ: -57.1, // approx per mol for strong acid/base
    notes: "Neutralisation forms water and salt.",
  },
  {
    id: "gas_evolution_bicarb",
    name: "Gas evolution: HCl + NaHCO3 → H2O + CO2 + NaC2H3O2",
    reactants: ["HCl(aq)", "NaHCO3(s)"],
    products: ["H2O(l)", "CO2(g)", "NaC2H3O2(aq)"],
    classes: ["Gas evolution", "Acid–base"],
    enthalpy_kJ: undefined,
    notes: "CO2 gas bubbles; also an acid–base reaction via H2CO3 intermediate.",
  },
  {
    id: "single_displacement",
    name: "Single displacement: Cu + AgNO3 → Cu(NO3)2 + Ag",
    reactants: ["Cu(s)", "AgNO3(aq)"],
    products: ["Cu(NO3)2(aq)", "Ag(s)"],
    classes: ["Single displacement", "Redox"],
    enthalpy_kJ: undefined,
    notes: "Silver plates out; blue Cu²⁺ solution forms.",
  },
  {
    id: "combust_propane",
    name: "Combustion: C3H8 + O2 → CO2 + H2O",
    reactants: ["C3H8(g)", "O2(g)"],
    products: ["CO2(g)", "H2O(g)"],
    classes: ["Combustion", "Redox"],
    enthalpy_kJ: -2044,
    notes: "Exothermic combustion of propane.",
  },
];

// --- UI Components ---

function Badge({ children }: { children: React.ReactNode }) {
  return <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 text-xs border">{children}</span>;
}

function SectionTitle({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle?: string }) {
  return (
    <div className="flex items-center gap-3 mb-3">
      <div className="p-2 rounded-2xl bg-slate-50 shadow-sm">{icon}</div>
      <div>
        <h3 className="text-lg font-semibold">{title}</h3>
        {subtitle && <p className="text-sm text-slate-500">{subtitle}</p>}
      </div>
    </div>
  );
}

function EquationBalancer() {
  const [lhs, setLhs] = useState("C2H6 + O2");
  const [rhs, setRhs] = useState("CO2 + H2O");
  const [result, setResult] = useState<string>("");
  const [error, setError] = useState<string>("");

  const onBalance = () => {
    setError("");
    try {
      const reactants = lhs.split("+").map((s) => s.trim());
      const products = rhs.split("+").map((s) => s.trim());
      const coeffs = balanceEquation(reactants, products);
      if (!coeffs) {
        setResult("");
        setError("Could not balance this equation. Check your formulas.");
        return;
      }
      setResult(prettyEquation(reactants, products, coeffs));
    } catch (e: any) {
      setError(e.message || "Parsing error. Check chemical formulas.");
    }
  };

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><FlaskConical className="w-5 h-5"/> Balance an Equation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <Label>Reactants (separate with +)</Label>
            <Input value={lhs} onChange={(e) => setLhs(e.target.value)} placeholder="e.g., C2H6 + O2"/>
          </div>
          <div>
            <Label>Products (separate with +)</Label>
            <Input value={rhs} onChange={(e) => setRhs(e.target.value)} placeholder="e.g., CO2 + H2O"/>
          </div>
        </div>
        <div className="flex gap-2">
          <Button onClick={onBalance}><Sparkles className="w-4 h-4 mr-1"/>Auto‑balance</Button>
          <Button variant="secondary" onClick={() => { setLhs("CH4 + O2"); setRhs("CO2 + H2O"); setResult(""); setError(""); }}>Load CH4 combustion</Button>
        </div>
        {error && <p className="text-red-600 text-sm">{error}</p>}
        {result && (
          <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="p-3 bg-emerald-50 border rounded-xl">
            <div className="text-sm text-slate-600 mb-1">Balanced:</div>
            <div className="font-mono text-base">{result}</div>
          </motion.div>
        )}
        <div className="text-xs text-slate-500">
          Tip: Include states like (s), (l), (g), (aq) if you want — the balancer will ignore them when counting atoms.
        </div>
      </CardContent>
    </Card>
  );
}

function ReactionExplorer() {
  const [selected, setSelected] = useState(PRESET_REACTIONS[0].id);
  const rxn = useMemo(() => PRESET_REACTIONS.find(r => r.id === selected)!, [selected]);
  const coeffs = useMemo(() => balanceEquation(rxn.reactants, rxn.products)!, [rxn]);
  const pretty = useMemo(() => prettyEquation(rxn.reactants, rxn.products, coeffs), [rxn, coeffs]);

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><FireExtinguisher className="w-5 h-5"/> Classify Reaction Types</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <Label>Choose a reaction</Label>
            <Select value={selected} onValueChange={setSelected}>
              <SelectTrigger><SelectValue placeholder="Select"/></SelectTrigger>
              <SelectContent>
                {PRESET_REACTIONS.map((r) => (
                  <SelectItem key={r.id} value={r.id}>{r.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label>Classification</Label>
            <div className="flex flex-wrap gap-2 mt-2">
              {rxn.classes.map((c) => <Badge key={c}>{c}</Badge>)}
            </div>
          </div>
        </div>
        <div className="p-3 rounded-xl bg-indigo-50 border">
          <div className="text-sm text-slate-600 mb-1">Balanced:</div>
          <div className="font-mono">{pretty}</div>
        </div>
        {rxn.notes && <p className="text-sm text-slate-600">Observation: {rxn.notes}</p>}
      </CardContent>
    </Card>
  );
}

function LimitingReactant() {
  const [selected, setSelected] = useState("combust_methane");
  const rxn = useMemo(() => PRESET_REACTIONS.find(r => r.id === selected)!, [selected]);
  const coeffs = useMemo(() => balanceEquation(rxn.reactants, rxn.products)!, [rxn]);

  const [massA, setMassA] = useState("16"); // grams reactant 1
  const [massB, setMassB] = useState("64"); // grams reactant 2
  const [productIndex, setProductIndex] = useState(0);
  const [actualYield, setActualYield] = useState("");

  // Compute molar masses and moles
  const info = useMemo(() => {
    const [rA, rB] = rxn.reactants;
    const mA = molarMass(rA);
    const mB = molarMass(rB);
    const nA = parseFloat(massA || "0") / mA;
    const nB = parseFloat(massB || "0") / mB;

    const cA = coeffs[0];
    const cB = coeffs[1];

    const extentA = nA / cA; // how many "reaction units" each can supply
    const extentB = nB / cB;
    const extent = Math.min(extentA, extentB);
    const limiting = extentA < extentB ? 0 : 1;

    // Theoretical moles of chosen product
    const pCoeff = coeffs[rxn.reactants.length + productIndex];
    const nP = extent * pCoeff;
    const pFormula = rxn.products[productIndex];
    const mP = molarMass(pFormula);
    const massTheo = nP * mP;

    let percent = undefined as number | undefined;
    if (actualYield) {
      const ay = parseFloat(actualYield);
      if (isFinite(ay) && massTheo > 0) percent = (ay / massTheo) * 100;
    }

    return {
      rA, rB, mA, mB, nA, nB, cA, cB, limiting,
      pFormula, pCoeff, mP, nP, massTheo, percent,
    };
  }, [rxn, coeffs, massA, massB, productIndex, actualYield]);

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Calculator className="w-5 h-5"/> Limiting Reactant & Yields</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid md:grid-cols-2 gap-3">
          <div>
            <Label>Choose a reaction</Label>
            <Select value={selected} onValueChange={setSelected}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {PRESET_REACTIONS.filter(r => r.reactants.length >= 2).map((r) => (
                  <SelectItem key={r.id} value={r.id}>{r.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="text-xs text-slate-500 mt-1">We’ll consider the first two reactants for the calculation.</div>
          </div>
          <div>
            <Label>Product to track</Label>
            <Select value={String(productIndex)} onValueChange={(v) => setProductIndex(Number(v))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {rxn.products.map((p, i) => (
                  <SelectItem key={i} value={String(i)}>{p}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-3">
          <div className="p-3 rounded-xl border bg-slate-50">
            <div className="font-medium mb-2">Reactant amounts</div>
            <div className="grid grid-cols-2 gap-3 items-end">
              <div>
                <Label>{info.rA} mass (g)</Label>
                <Input value={massA} onChange={(e) => setMassA(e.target.value)} />
                <p className="text-xs text-slate-500 mt-1">Molar mass ≈ {info.mA.toFixed(2)} g/mol</p>
              </div>
              <div>
                <Label>{info.rB} mass (g)</Label>
                <Input value={massB} onChange={(e) => setMassB(e.target.value)} />
                <p className="text-xs text-slate-500 mt-1">Molar mass ≈ {info.mB.toFixed(2)} g/mol</p>
              </div>
            </div>
            <div className="mt-3 text-sm">
              Limiting reactant: <span className="font-semibold">{info.limiting === 0 ? info.rA : info.rB}</span>
            </div>
          </div>

          <div className="p-3 rounded-xl border bg-emerald-50">
            <div className="font-medium mb-2">Theoretical yield</div>
            <div className="text-sm">{info.pFormula} moles ≈ <span className="font-semibold">{info.nP.toFixed(3)}</span></div>
            <div className="text-sm">{info.pFormula} mass ≈ <span className="font-semibold">{info.massTheo.toFixed(2)} g</span></div>
            <div className="mt-3 grid grid-cols-2 gap-2 items-end">
              <div>
                <Label>Actual yield (g)</Label>
                <Input value={actualYield} onChange={(e) => setActualYield(e.target.value)} placeholder="optional" />
              </div>
              <div>
                <Label>% yield</Label>
                <div className="p-2 rounded-lg bg-white border text-center font-semibold">
                  {info.percent === undefined || isNaN(info.percent) ? "—" : `${info.percent.toFixed(1)} %`}
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function HeatTransfer() {
  const [selected, setSelected] = useState("combust_propane");
  const rxn = useMemo(() => PRESET_REACTIONS.find(r => r.id === selected)!, [selected]);
  const coeffs = useMemo(() => balanceEquation(rxn.reactants, rxn.products)!, [rxn]);
  const pretty = useMemo(() => prettyEquation(rxn.reactants, rxn.products, coeffs), [rxn, coeffs]);

  const [scale, setScale] = useState([1]); // slider multiplier for scaling reaction amount
  const enthalpy = (rxn.enthalpy_kJ ?? 0) * scale[0];
  const exo = enthalpy < 0;

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><ThermometerSun className="w-5 h-5"/> Heat Transfer (ΔH)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid md:grid-cols-2 gap-3">
          <div>
            <Label>Choose a reaction</Label>
            <Select value={selected} onValueChange={setSelected}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {PRESET_REACTIONS.filter(r => typeof r.enthalpy_kJ === 'number').map((r) => (
                  <SelectItem key={r.id} value={r.id}>{r.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label>Scale (x stoichiometric amount)</Label>
            <div className="px-2">
              <Slider min={0.25} max={4} step={0.25} value={scale} onValueChange={setScale} />
              <div className="text-xs text-slate-500 mt-1">Scaling: ×{scale[0].toFixed(2)}</div>
            </div>
          </div>
        </div>

        <div className={`p-3 rounded-xl border flex items-center justify-between ${exo ? 'bg-orange-50' : 'bg-sky-50'}`}>
          <div>
            <div className="text-sm text-slate-600 mb-1">Balanced (per stoichiometric unit shown):</div>
            <div className="font-mono">{pretty}</div>
            <div className="mt-2 text-sm">ΔH ≈ <span className="font-semibold">{enthalpy.toLocaleString(undefined, { maximumFractionDigits: 1 })} kJ</span> ({exo ? 'exothermic' : 'endothermic'})</div>
          </div>
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="ml-4 mr-2"
          >
            {exo ? <Flame className="w-12 h-12"/> : <ThermometerSun className="w-12 h-12"/>}
          </motion.div>
        </div>
        <p className="text-xs text-slate-500">The displayed ΔH scales linearly with the slider. Negative means heat released to surroundings (exothermic); positive means heat absorbed (endothermic).</p>
      </CardContent>
    </Card>
  );
}

// --- Lightweight in-app tests ---

type TestCase = { name: string; run: () => { pass: boolean; detail?: string } };

const TESTS: TestCase[] = [
  {
    name: "Balance CH4 combustion",
    run: () => {
      const coeffs = balanceEquation(["CH4", "O2"], ["CO2", "H2O"]);
      const expect = [1, 2, 1, 2];
      const pass = !!coeffs && coeffs.length === 4 && coeffs.every((v, i) => v === expect[i]);
      return { pass, detail: `got ${coeffs}` };
    },
  },
  {
    name: "Balance C2H6 combustion",
    run: () => {
      const coeffs = balanceEquation(["C2H6", "O2"], ["CO2", "H2O"]);
      const expect = [2, 7, 4, 6];
      const pass = !!coeffs && coeffs.length === 4 && coeffs.every((v, i) => v === expect[i]);
      return { pass, detail: `got ${coeffs}` };
    },
  },
  {
    name: "Molar mass of H2O ≈ 18.015",
    run: () => {
      const mm = molarMass("H2O");
      return { pass: Math.abs(mm - 18.015) < 0.05, detail: `got ${mm.toFixed(3)}` };
    },
  },
  {
    name: "Parse Pb(NO3)2 atom counts",
    run: () => {
      const c = parseFormula("Pb(NO3)2");
      const pass = c.Pb === 1 && c.N === 2 && c.O === 6;
      return { pass, detail: `counts ${JSON.stringify(c)}` };
    },
  },
  {
    name: "Limiting reactant for CH4 (16 g) & O2 (40 g) is O2",
    run: () => {
      const coeffs = balanceEquation(["CH4", "O2"], ["CO2", "H2O"])!;
      const mCH4 = molarMass("CH4");
      const mO2 = molarMass("O2");
      const nA = 16 / mCH4; // ≈1 mol
      const nB = 40 / mO2; // ≈1.25 mol O atoms -> 1.25 mol O2? actually 40/32=1.25 mol O2
      const extentA = nA / coeffs[0];
      const extentB = nB / coeffs[1];
      const limiting = extentA < extentB ? 0 : 1; // 0 -> CH4, 1 -> O2
      return { pass: limiting === 1, detail: `extentA=${extentA.toFixed(2)} extentB=${extentB.toFixed(2)}` };
    },
  },
];

function TestRunner() {
  const results = TESTS.map(t => ({ name: t.name, ...t.run() }));
  const passed = results.filter(r => r.pass).length;
  const failed = results.length - passed;
  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Sparkles className="w-5 h-5"/> Self‑tests</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-sm text-slate-600">{passed} passed, {failed} failed.</div>
        <ul className="space-y-2">
          {results.map((r, i) => (
            <li key={i} className={`flex items-start gap-2 p-2 rounded-lg border ${r.pass ? 'bg-emerald-50' : 'bg-red-50'}`}>
              {r.pass ? <CheckCircle2 className="w-4 h-4 mt-0.5"/> : <AlertCircle className="w-4 h-4 mt-0.5"/>}
              <div>
                <div className="font-medium text-sm">{r.name}</div>
                {r.detail && <div className="text-xs text-slate-600">{r.detail}</div>}
              </div>
            </li>
          ))}
        </ul>
        <div className="text-xs text-slate-500">Tests are lightweight and run in‑app to catch regressions (balancer, parser, basic stoichiometry). Add more as you expand features.</div>
      </CardContent>
    </Card>
  );
}

export default function ChemReactionsSimulator() {
  return (
    <div className="p-6 md:p-10 max-w-6xl mx-auto space-y-6">
      <header className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Chemical Reactions — Interactive Simulator</h1>
          <p className="text-slate-600 mt-1 max-w-3xl">Explore equations, reaction types, limiting reactants, yields, and heat transfer through an interactive, visual tool designed to match your Week 6 content.</p>
        </div>
      </header>

      <Tabs defaultValue="balance" className="space-y-4">
        <TabsList className="grid grid-cols-2 md:grid-cols-5 lg:grid-cols-5 gap-2">
          <TabsTrigger value="balance">Balance</TabsTrigger>
          <TabsTrigger value="classify">Classify</TabsTrigger>
          <TabsTrigger value="limiting">Limiting & Yields</TabsTrigger>
          <TabsTrigger value="heat">Heat (ΔH)</TabsTrigger>
          <TabsTrigger value="tests">Self‑tests</TabsTrigger>
        </TabsList>

        <TabsContent value="balance" className="space-y-4">
          <SectionTitle icon={<FlaskConical className="w-5 h-5"/>} title="Balance chemical equations" subtitle="Auto‑balance or practice manually using your own equations." />
          <EquationBalancer />
        </TabsContent>

        <TabsContent value="classify" className="space-y-4">
          <SectionTitle icon={<FireExtinguisher className="w-5 h-5"/>} title="Classify reactions" subtitle="See precipitation, acid–base, gas evolution, displacement, and combustion examples." />
          <ReactionExplorer />
        </TabsContent>

        <TabsContent value="limiting" className="space-y-4">
          <SectionTitle icon={<Calculator className="w-5 h-5"/>} title="Limiting reactant & yields" subtitle="Enter masses to identify the limiting reagent and compute theoretical & percent yields." />
          <LimitingReactant />
        </TabsContent>

        <TabsContent value="heat" className="space-y-4">
          <SectionTitle icon={<ThermometerSun className="w-5 h-5"/>} title="Heat transfer in reactions" subtitle="Visualise exothermic vs endothermic reactions and scale ΔH with reaction size." />
          <HeatTransfer />
        </TabsContent>

        <TabsContent value="tests" className="space-y-4">
          <SectionTitle icon={<Sparkles className="w-5 h-5"/>} title="Built‑in checks" subtitle="Quick regression tests for parsing, balancing, and stoichiometry." />
          <TestRunner />
        </TabsContent>
      </Tabs>

      <footer className="text-xs text-slate-500 pt-2">
        Built for your CHEM1006 Week 6 module. Add more reactions by editing <code>PRESET_REACTIONS</code> and expand molar masses in <code>MOLAR_MASS</code> as desired.
      </footer>
    </div>
  );
}
