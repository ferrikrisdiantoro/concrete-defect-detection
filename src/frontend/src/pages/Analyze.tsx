import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  UploadCloud, Download, FileSpreadsheet, FileJson, Box, FileText,
  ImagePlus, Info, CheckCircle2, History as HistoryIcon, HelpCircle, Save, Star, ChevronDown, ChevronUp
} from "lucide-react";
import Field from "../components/Field";
import SelectField from "../components/SelectField";

type SeverityEN = "Minor" | "Moderate" | "Severe";
type SeverityAny = SeverityEN | "ringan" | "sedang" | "berat";

const sevNormalize = (s: SeverityAny): SeverityEN => {
  const m = String(s).toLowerCase();
  if (m === "ringan") return "Minor";
  if (m === "sedang") return "Moderate";
  if (m === "berat") return "Severe";
  return (m.charAt(0).toUpperCase() + m.slice(1)) as SeverityEN;
};

const sevClass = (s: SeverityAny) => {
  const n = sevNormalize(s);
  return n === "Severe" ? "bg-red-600/90 text-white"
    : n === "Moderate" ? "bg-amber-500/90 text-black"
      : "bg-emerald-600/90 text-white";
};

type Item = {
  bbox: [number, number, number, number];
  conf: number;
  cls: string;
  severity: SeverityAny;
  solution: string;
  keterangan: string;
  project_id: string;
  column_id: string;
  level?: string;
  grid?: string;
  location?: string;
  timestamp: string;
};
type AnalyzeResp = { image_width: number; image_height: number; items: Item[]; };

type MetaPreset = {
  name: string;
  project_id: string;
  column_id: string;
  level: string;
  grid: string;
  location: string;
};

const BACKEND_URL = (() => {
  const fromEnv = (import.meta.env.VITE_BACKEND_URL ?? "").trim();
  return fromEnv !== "" ? fromEnv.replace(/\/+$/, "") : "";
})();

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [imgUrl, setImgUrl] = useState<string>("");
  const [data, setData] = useState<AnalyzeResp | null>(null);
  const [loading, setLoading] = useState(false);
  const [showMeta, setShowMeta] = useState(true);

  // metadata core
  const [projectId, setProject] = useState("PRJ-001");
  const [columnId, setColumn] = useState("K-01");
  const [level, setLevel] = useState("");
  const [grid, setGrid] = useState("");
  const [location, setLocation] = useState("");

  // metadata extra
  const [projectName, setProjectName] = useState("");
  const [materialSpec, setMaterialSpec] = useState("");
  const [yearOfConstruction, setYearOfConstruction] = useState<number | "">("");
  const [defectSide, setDefectSide] = useState<"" | "North" | "South" | "East" | "West">("");
  const [inspectionDate, setInspectionDate] = useState<string>("");
  const [inspector, setInspector] = useState("");
  const [columnType, setColumnType] = useState("");
  const [notes, setNotes] = useState("");

  // dimensions & coordinates
  const [dimW, setDimW] = useState<number | "">("");
  const [dimD, setDimD] = useState<number | "">("");
  const [coordX, setCoordX] = useState<number | "">("");
  const [coordY, setCoordY] = useState<number | "">("");
  const [coordZ, setCoordZ] = useState<number | "">("");

  // autosuggest
  const [suggestProjects, setSuggestProjects] = useState<string[]>([]);
  const [suggestColumns, setSuggestColumns] = useState<string[]>([]);

  // presets
  const [presets, setPresets] = useState<MetaPreset[]>([]);

  // single ref
  const imgRef = useRef<HTMLImageElement>(null);

  // ---- load suggestions ----
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${BACKEND_URL}/history?limit=250`);
        const js = await r.json();
        if (Array.isArray(js)) {
          const p = new Set<string>();
          const c = new Set<string>();
          js.forEach((rec: any) => {
            if (rec.project_id) p.add(rec.project_id);
            if (rec.column_id) c.add(rec.column_id);
          });
          setSuggestProjects([...p].slice(0, 50));
          setSuggestColumns([...c].slice(0, 50));
        }
      } catch { /* ignore */ }
    })();
  }, []);

  // ---- load presets ----
  useEffect(() => {
    const raw = localStorage.getItem("cd_meta_presets");
    if (raw) setPresets(JSON.parse(raw));
  }, []);

  // ---- preset handlers ----
  const savePreset = useCallback(() => {
    const name = window.prompt("Preset Name (e.g., 'L1-East'):");
    if (!name) return;
    const next = [
      ...presets.filter((p) => p.name !== name),
      { name, project_id: projectId, column_id: columnId, level, grid, location },
    ];
    setPresets(next);
    localStorage.setItem("cd_meta_presets", JSON.stringify(next));
  }, [presets, projectId, columnId, level, grid, location]);

  const applyPreset = useCallback((p: MetaPreset) => {
    setProject(p.project_id);
    setColumn(p.column_id);
    setLevel(p.level);
    setGrid(p.grid);
    setLocation(p.location);
  }, []);

  // ---- file handlers ----
  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    setFile(f);
    setData(null);
    setImgUrl(URL.createObjectURL(f));
  }, []);

  const onPick = useCallback((f: File | null) => {
    setFile(f);
    setData(null);
    setImgUrl(f ? URL.createObjectURL(f) : "");
  }, []);

  // ---- composed meta ----
  const buildMeta = () => {
    const meta: any = {
      project_id: projectId,
      project_name: projectName || undefined,
      column_id: columnId,
      level: level || undefined,
      grid: grid || undefined,
      location: location || undefined,
      material_spec: materialSpec || undefined,
      year_of_construction: yearOfConstruction === "" ? undefined : Number(yearOfConstruction),
      defect_side: defectSide || undefined,
      inspection_date: inspectionDate || undefined,
      inspector: inspector || undefined,
      column_type: columnType || undefined,
      notes: notes || undefined,
    };
    if (dimW !== "" || dimD !== "") {
      meta.dimensions_mm = {
        width: dimW === "" ? 0 : Number(dimW),
        depth: dimD === "" ? 0 : Number(dimD),
      };
    }
    if (coordX !== "" || coordY !== "" || coordZ !== "") {
      meta.coordinates_mm = {
        x: coordX === "" ? 0 : Number(coordX),
        y: coordY === "" ? 0 : Number(coordY),
        z: coordZ === "" ? 0 : Number(coordZ),
      };
    }
    return meta;
  };

  // ---- analyze ----
  const onAnalyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("image", file);
      fd.append("project_id", projectId);
      fd.append("column_id", columnId);
      fd.append("level", level);
      fd.append("grid", grid);
      fd.append("location", location);
      fd.append("meta", JSON.stringify(buildMeta()));
      const r = await fetch(`${BACKEND_URL}/analyze`, { method: "POST", body: fd });
      if (!r.ok) {
        alert(await r.text());
        return;
      }
      setData(await r.json());
    } finally {
      setLoading(false);
    }
  }, [
    file, projectId, columnId, level, grid, location,
    projectName, materialSpec, yearOfConstruction, defectSide, inspectionDate,
    inspector, columnType, notes, dimW, dimD, coordX, coordY, coordZ,
  ]);

  // ---- export ----
  const exportBatch = useCallback(
    async (path: "/export/csv" | "/export/json" | "/export/ifc-overlay" | "/export/pdf") => {
      if (!data) return;
      const resp = await fetch(`${BACKEND_URL}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId, items: data.items }),
      });
      if (!resp.ok) {
        alert(await resp.text());
        return;
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download =
        path === "/export/csv"
          ? "export_damage.csv"
          : path === "/export/json"
            ? "export_damage.json"
            : path === "/export/ifc-overlay"
              ? "damage_ifc_overlay.json"
              : "damage_report.pdf";
      a.click();
      URL.revokeObjectURL(url);
    },
    [data, projectId]
  );

  // ---- overlay boxes ----
  const overlayBoxes = useMemo(() => {
    if (!data || !imgRef.current) return null;
    const sx = imgRef.current.clientWidth / data.image_width;
    const sy = imgRef.current.clientHeight / data.image_height;
    return data.items.map((it, i) => {
      const [x1, y1, x2, y2] = it.bbox;
      const left = x1 * sx, top = y1 * sy, w = (x2 - x1) * sx, h = (y2 - y1) * sy;
      const sev = sevNormalize(it.severity);
      const border = sev === "Severe" ? "border-red-500" : sev === "Moderate" ? "border-amber-500" : "border-emerald-500";
      return (
        <motion.div
          key={i}
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          className={`absolute rounded-sm border-2 ${border} bg-white/10`}
          style={{ left, top, width: w, height: h }}
        >
          <div className={`absolute -top-7 left-0 rounded px-1.5 py-0.5 text-[10px] font-bold text-white shadow-sm ${sev === "Severe" ? "bg-red-500" : sev === "Moderate" ? "bg-amber-500" : "bg-emerald-500"}`}>
            {it.cls} • {sev} • {(it.conf * 100).toFixed(0)}%
          </div>
        </motion.div>
      );
    });
  }, [data, imgUrl]);

  return (
    <main className="py-6 md:py-8 lg:max-w-7xl mx-auto px-4 sm:px-6">
      <section className="bg-white rounded-xl border border-zinc-200 shadow-sm p-4 md:p-6 mb-6">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-800">AI Crack Detection Analysis</h1>
            <p className="text-slate-500 text-sm mt-1">
              Analyze column images for structural defects using AI.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="text-xs font-medium text-slate-500 mr-2">Severity Legend:</div>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-700 uppercase tracking-wide">Minor</span>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-700 uppercase tracking-wide">Moderate</span>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-100 text-red-700 uppercase tracking-wide">Severe</span>
          </div>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[1fr,400px]">
        {/* Left Column: Input & Upload */}
        <section className="space-y-6">
          {/* Metadata Card */}
          <div className="bg-white rounded-xl border border-zinc-200 shadow-sm overflow-hidden">
            <div
              className="px-4 py-3 bg-slate-50 border-b border-zinc-100 flex items-center justify-between cursor-pointer hover:bg-slate-100 transition-colors"
              onClick={() => setShowMeta(!showMeta)}
            >
              <h3 className="font-semibold text-slate-700 text-sm flex items-center gap-2">
                <Info className="h-4 w-4" /> Column Metadata
              </h3>
              {showMeta ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
            </div>

            <AnimatePresence>
              {showMeta && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden"
                >
                  <div className="p-4 md:p-6 space-y-4">
                    {/* Presets & Autosuggest */}
                    <div className="flex flex-wrap items-center gap-2 mb-4">
                      <button className="btn-outline text-xs h-8" onClick={savePreset}><Save className="h-3.5 w-3.5 mr-1" /> Save Preset</button>
                      {presets.length > 0 && (
                        <div className="inline-flex items-center gap-2 text-xs">
                          <span className="text-slate-400">Load:</span>
                          <select
                            className="input h-8 py-0 pl-2 text-xs w-auto"
                            onChange={(e) => {
                              const p = presets.find((x) => x.name === e.target.value);
                              if (p) applyPreset(p);
                            }}
                            defaultValue=""
                          >
                            <option value="" disabled>Select Preset...</option>
                            {presets.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
                          </select>
                        </div>
                      )}
                      <datalist id="projects">{suggestProjects.map((p) => <option key={p} value={p} />)}</datalist>
                      <datalist id="columns">{suggestColumns.map((c) => <option key={c} value={c} />)}</datalist>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <Field label="Project ID" listId="projects" value={projectId} onChange={setProject} />
                      <Field label="Column ID" listId="columns" value={columnId} onChange={setColumn} />
                      <Field label="Level" value={level} onChange={setLevel} />
                      <Field label="Grid" value={grid} onChange={setGrid} />
                    </div>

                    <div className="pt-4 border-t border-dashed border-zinc-200">
                      <div className="text-xs font-medium text-slate-400 mb-3 uppercase tracking-wider">Detailed Info (Optional)</div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <Field label="Location" value={location} onChange={setLocation} />
                        <Field label="Material" value={materialSpec} onChange={setMaterialSpec} />
                        <SelectField<"" | "North" | "South" | "East" | "West">
                          label="Side"
                          value={defectSide}
                          onChange={setDefectSide}
                          options={["", "North", "South", "East", "West"] as const}
                        />
                        <Field label="Inspector" value={inspector} onChange={setInspector} />
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
                        <Field label="Insp. Date" type="date" value={inspectionDate} onChange={setInspectionDate} />
                        <Field label="Dim W (mm)" type="number" value={dimW} onChange={(v) => setDimW(v as any)} />
                        <Field label="Dim D (mm)" type="number" value={dimD} onChange={(v) => setDimD(v as any)} />
                        <div className="md:col-span-1">
                          <Field label="Notes" value={notes} onChange={setNotes} />
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Upload & Preview */}
          <div
            className="bg-white rounded-xl border border-zinc-200 shadow-sm p-4 md:p-6"
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
          >
            <div className="mb-4 flex items-center justify-between">
              <h3 className="font-semibold text-slate-800 text-sm">Image Analysis</h3>
              {file && <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">{file.name}</span>}
            </div>

            <div className="relative min-h-[300px] bg-zinc-50 rounded-lg border-2 border-dashed border-zinc-300 flex flex-col items-center justify-center overflow-hidden">
              {!imgUrl ? (
                <div className="text-center p-8">
                  <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <UploadCloud className="h-8 w-8 text-slate-400" />
                  </div>
                  <p className="text-sm font-medium text-slate-900">Drag and drop your image here</p>
                  <p className="text-xs text-slate-500 mt-1 mb-4">or click to browse from credentials</p>
                  <label className="btn inline-flex cursor-pointer">
                    Select Image
                    <input type="file" accept="image/*" className="hidden" onChange={(e) => onPick(e.target.files?.[0] || null)} />
                  </label>
                </div>
              ) : (
                <>
                  <img ref={imgRef} src={imgUrl} alt="preview" className="max-h-[500px] w-full object-contain" />
                  <div className="absolute inset-0 pointer-events-none">{overlayBoxes}</div>

                  <div className="absolute top-2 right-2 flex gap-2 pointer-events-auto">
                    <label className="btn-sm bg-white/90 backdrop-blur text-slate-700 hover:bg-white shadow-sm cursor-pointer">
                      Change
                      <input type="file" accept="image/*" className="hidden" onChange={(e) => onPick(e.target.files?.[0] || null)} />
                    </label>
                  </div>
                </>
              )}
            </div>

            <div className="mt-4 flex justify-end">
              <button
                className="btn px-6 py-2.5 shadow-lg shadow-primary/20"
                disabled={!file || loading}
                onClick={onAnalyze}
              >
                {loading ? <span className="loader mr-2" /> : <Download className="h-4 w-4 mr-2" />}
                {loading ? "Analyzing..." : "Run Analysis"}
              </button>
            </div>
          </div>
        </section>

        {/* Right Column: Results */}
        <motion.aside
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-4"
        >
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-800">Analysis Results</h2>
            {data && (
              <div className="flex gap-2">
                <button className="p-2 rounded-md border border-zinc-200 hover:bg-slate-50 text-slate-600" title="Export JSON" onClick={() => exportBatch("/export/json")}><FileJson className="h-4 w-4" /></button>
                <button className="p-2 rounded-md border border-zinc-200 hover:bg-slate-50 text-slate-600" title="Export CSV" onClick={() => exportBatch("/export/csv")}><FileSpreadsheet className="h-4 w-4" /></button>
                <button className="p-2 rounded-md border border-zinc-200 hover:bg-slate-50 text-slate-600" title="Export PDF" onClick={() => exportBatch("/export/pdf")}><FileText className="h-4 w-4" /></button>
                <button className="p-2 rounded-md border border-zinc-200 hover:bg-slate-50 text-slate-600" title="IFC Overlay" onClick={() => exportBatch("/export/ifc-overlay")}><Box className="h-4 w-4" /></button>
              </div>
            )}
          </div>

          {!data && (
            <div className="h-64 rounded-xl border border-dashed border-zinc-300 bg-slate-50/50 flex flex-col items-center justify-center text-slate-400 p-8 text-center">
              <Info className="h-8 w-8 mb-3 opacity-50" />
              <p className="text-sm">Results will appear here after analysis.</p>
            </div>
          )}

          <div className="space-y-3">
            {data?.items.map((it, i) => {
              const sev = sevNormalize(it.severity);
              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="bg-white rounded-lg border border-zinc-200 p-4 shadow-sm hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex flex-col">
                      <span className="text-sm font-bold text-slate-800 uppercase tracking-tight">{it.cls}</span>
                      <span className="text-[10px] text-slate-400 font-mono mt-0.5">CONF: {(it.conf * 100).toFixed(1)}%</span>
                    </div>
                    <span className={`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide ${sev === "Severe" ? "bg-red-100 text-red-700" :
                      sev === "Moderate" ? "bg-amber-100 text-amber-700" :
                        "bg-emerald-100 text-emerald-700"
                      }`}>
                      {sev}
                    </span>
                  </div>

                  <div className="space-y-2 mt-3">
                    <div className="text-xs">
                      <span className="font-semibold text-slate-700 block mb-0.5 flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-emerald-500" /> Recommended Solution</span>
                      <p className="text-slate-600 leading-relaxed pl-4.5">{it.solution}</p>
                    </div>

                    {it.keterangan && (
                      <div className="text-xs bg-slate-50 p-2 rounded border border-slate-100 text-slate-500 mt-2">
                        {it.keterangan}
                      </div>
                    )}

                    <div className="pt-2 mt-2 border-t border-zinc-100 grid grid-cols-2 text-[10px] text-slate-400">
                      <div>BBOX: [{it.bbox.join(',')}]</div>
                      <div className="text-right">{it.timestamp.split('T')[1].split('.')[0]}</div>
                    </div>
                  </div>
                </motion.div>
              );
            })}

            {data && (
              <a href="/history" className="block w-full py-3 text-center text-xs font-medium text-primary hover:text-accent hover:bg-slate-50 rounded-lg border border-transparent hover:border-zinc-200 transition-all">
                View in History →
              </a>
            )}
          </div>
        </motion.aside>
      </div>
    </main>
  );
}
