import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  FileSpreadsheet,
  FileJson,
  Filter,
  Clock,
  Info,
  Upload,
  Save,
  Trash2,
  CheckCircle2,
  MapPin,
  Calendar,
  MoreVertical,
} from "lucide-react";

type SeverityEN = "Minor" | "Moderate" | "Severe";
type SeverityAny = SeverityEN | "ringan" | "sedang" | "berat";

const sevNormalize = (s: SeverityAny): SeverityEN => {
  const m = String(s).toLowerCase();
  if (m === "ringan") return "Minor";
  if (m === "sedang") return "Moderate";
  if (m === "berat") return "Severe";
  return (m.charAt(0).toUpperCase() + m.slice(1)) as SeverityEN;
};

type Monitoring = {
  status: "pending" | "in_progress" | "fixed";
  notes?: string;
  photos: string[]; // url relatif "/uploads/xxx.jpg"
  updated_at?: string;
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
  monitoring?: Monitoring;
};

type HistoryRecord = {
  id: string;
  project_id: string;
  column_id: string;
  timestamp: string;
  image_url?: string;
  image_width: number;
  image_height: number;
  items: Item[];
  meta?: any;
};

const BACKEND_URL = (() => {
  const fromEnv = (import.meta.env.VITE_BACKEND_URL ?? "").trim();
  return fromEnv !== "" ? fromEnv.replace(/\/+$/, "") : "";
})();

export default function HistoryPage() {
  const [list, setList] = useState<HistoryRecord[]>([]);
  const [projectId, setProject] = useState("");
  const [columnId, setColumn] = useState("");
  const [limit, setLimit] = useState(100);
  const [loading, setLoading] = useState(false);

  const [savingKey, setSavingKey] = useState<string | null>(null); // "recordId#index"
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [bulkDeleting, setBulkDeleting] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const qs = new URLSearchParams();
      if (projectId) qs.set("project_id", projectId);
      if (columnId) qs.set("column_id", columnId);
      if (limit) qs.set("limit", String(limit));
      const r = await fetch(`${BACKEND_URL}/history?${qs.toString()}`);
      const js = await r.json().catch(() => []);
      setList(Array.isArray(js) ? (js as HistoryRecord[]) : []);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const exportAll = async (type: "csv" | "json") => {
    const items = list.flatMap((rec) => rec.items);
    if (items.length === 0) {
      alert("Tidak ada data untuk diekspor.");
      return;
    }
    const endpoint = type === "csv" ? "/export/csv" : "/export/json";
    const resp = await fetch(`${BACKEND_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: projectId || "ALL", items }),
    });
    if (!resp.ok) {
      alert(await resp.text());
      return;
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = type === "csv" ? "history_export.csv" : "history_export.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  // ------- Monitoring helpers -------
  const patchMonitoring = async (
    recordId: string,
    itemIndex: number,
    data: FormData
  ) => {
    setSavingKey(`${recordId}#${itemIndex}`);
    try {
      const resp = await fetch(`${BACKEND_URL}/history/monitor`, {
        method: "POST",
        body: data,
      });
      if (!resp.ok) {
        const t = await resp.text();
        alert(t || "Gagal menyimpan monitoring");
        return;
      }
      const updated: HistoryRecord = await resp.json();
      setList((prev) => prev.map((r) => (r.id === updated.id ? updated : r)));
    } finally {
      setSavingKey(null);
    }
  };

  const handleStatusChange = async (
    recordId: string,
    itemIndex: number,
    status: Monitoring["status"]
  ) => {
    const fd = new FormData();
    fd.append("record_id", recordId);
    fd.append("item_index", String(itemIndex));
    fd.append("status", status);
    await patchMonitoring(recordId, itemIndex, fd);
  };

  const handleSaveNotes = async (
    recordId: string,
    itemIndex: number,
    notes: string
  ) => {
    const fd = new FormData();
    fd.append("record_id", recordId);
    fd.append("item_index", String(itemIndex));
    fd.append("notes", notes);
    await patchMonitoring(recordId, itemIndex, fd);
  };

  const handleUploadPhotos = async (
    recordId: string,
    itemIndex: number,
    files: FileList | null
  ) => {
    if (!files || files.length === 0) return;
    const fd = new FormData();
    fd.append("record_id", recordId);
    fd.append("item_index", String(itemIndex));
    Array.from(files).forEach((f) => fd.append("photos", f));
    await patchMonitoring(recordId, itemIndex, fd);
  };

  // ------- Delete helpers -------
  const deleteRecord = async (recordId: string) => {
    const ok = window.confirm(
      "Hapus history ini? Tindakan tidak bisa dibatalkan."
    );
    if (!ok) return;
    setDeletingId(recordId);
    try {
      const resp = await fetch(`${BACKEND_URL}/history/${recordId}`, {
        method: "DELETE",
      });
      if (resp.status === 204) {
        setList((prev) => prev.filter((r) => r.id !== recordId));
      } else {
        const msg = await resp.text();
        alert(msg || "Gagal menghapus history.");
      }
    } finally {
      setDeletingId(null);
    }
  };

  const clearFiltered = async () => {
    if (!projectId && !columnId) {
      alert("Isi Project ID atau Column ID untuk menghapus secara terfilter.");
      return;
    }
    const ok = window.confirm(
      `Hapus SEMUA history yang cocok dengan filter saat ini?\nProject: ${projectId || "(semua)"} | Column: ${columnId || "(semua)"
      }`
    );
    if (!ok) return;
    setBulkDeleting(true);
    try {
      const qs = new URLSearchParams();
      if (projectId) qs.set("project_id", projectId);
      if (columnId) qs.set("column_id", columnId);
      const resp = await fetch(`${BACKEND_URL}/history?${qs.toString()}`, {
        method: "DELETE",
      });
      if (!resp.ok) {
        const t = await resp.text();
        alert(t || "Gagal menghapus data terfilter.");
      } else {
        await load();
      }
    } finally {
      setBulkDeleting(false);
    }
  };

  return (
    <main className="py-6 md:py-8 lg:max-w-7xl mx-auto px-4 sm:px-6">
      <section className="bg-white rounded-xl border border-zinc-200 shadow-sm p-4 md:p-6 mb-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-800">Inspection History</h1>
            <p className="text-slate-500 text-sm mt-1">Manage and track column inspection records.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button className="btn-outline justify-center" onClick={() => exportAll("csv")}>
              <FileSpreadsheet className="h-4 w-4 mr-2" /> CSV
            </button>
            <button className="btn-outline justify-center" onClick={() => exportAll("json")}>
              <FileJson className="h-4 w-4 mr-2" /> JSON
            </button>
            {(projectId || columnId) && (
              <button
                className="btn-outline border-red-200 text-red-600 hover:bg-red-50 justify-center"
                onClick={clearFiltered}
                disabled={bulkDeleting}
                title="Delete matching records"
              >
                <Trash2 className="h-4 w-4 mr-2" />
                {bulkDeleting ? "Deleting..." : "Delete Filtered"}
              </button>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 items-end bg-slate-50 p-4 rounded-lg border border-slate-100">
          <label className="flex flex-col gap-1.5">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Project ID</span>
            <input
              className="input bg-white h-10"
              placeholder="e.g. PRJ-001"
              value={projectId}
              onChange={(e) => setProject(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Column ID</span>
            <input
              className="input bg-white h-10"
              placeholder="e.g. K-01"
              value={columnId}
              onChange={(e) => setColumn(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Limit</span>
            <input
              className="input bg-white h-10"
              type="number"
              value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value || "100", 10))}
            />
          </label>
          <div className="flex gap-2">
            <button className="btn h-10 flex-1 justify-center" onClick={load}>
              <Filter className="h-4 w-4 mr-2" /> Filter
            </button>
            <button className="btn-outline h-10 flex-1 justify-center bg-white" onClick={load}>
              <Clock className="h-4 w-4 mr-2" /> {loading ? "..." : "Refresh"}
            </button>
          </div>
        </div>
      </section>

      <section className="space-y-6">
        {(!list || list.length === 0) && (
          <div className="flex flex-col items-center justify-center py-16 text-slate-400 bg-white rounded-xl border border-dashed border-slate-200">
            <Info className="h-10 w-10 mb-3 opacity-50" />
            <p>No history records found.</p>
          </div>
        )}

        {list.map((h) => (
          <motion.article
            key={h.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col md:flex-row gap-6 bg-white rounded-xl border border-zinc-200 shadow-sm p-5 md:p-6 overflow-hidden"
          >
            {/* Left: Image */}
            <div className="flex-shrink-0 md:w-64 lg:w-72">
              <div className="relative aspect-[4/3] rounded-lg overflow-hidden border border-zinc-100 bg-zinc-50">
                {h.image_url ? (
                  <img
                    src={`${BACKEND_URL}${h.image_url}`}
                    alt="Inspection"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-xs text-muted">No Preview</div>
                )}
                {/* Overlay Metadata */}
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/60 to-transparent p-3 pt-6 text-white">
                  <div className="text-xs font-medium flex items-center gap-1.5"><Calendar className="h-3 w-3" /> {h.timestamp.split('T')[0]}</div>
                </div>
              </div>

              <button
                className="mt-3 w-full btn-outline border-red-200 text-red-600 hover:bg-red-50 text-xs py-1.5"
                onClick={() => deleteRecord(h.id)}
                disabled={deletingId === h.id}
              >
                <Trash2 className="h-3.5 w-3.5 mr-2" />
                {deletingId === h.id ? "Deleting..." : "Delete Record"}
              </button>
            </div>

            {/* Right: Details */}
            <div className="flex-1 min-w-0">
              {/* Header Info */}
              <div className="mb-4 pb-4 border-b border-zinc-100">
                <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 mb-1">
                  <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                    {h.project_id} <span className="text-slate-300">/</span> {h.column_id}
                  </h3>
                  {h.meta?.level && <span className="text-sm font-medium text-slate-500 bg-slate-100 px-2 py-0.5 rounded-full">Lvl: {h.meta.level}</span>}
                  {h.meta?.grid && <span className="text-sm font-medium text-slate-500 bg-slate-100 px-2 py-0.5 rounded-full">Grid: {h.meta.grid}</span>}
                </div>

                {/* Dense Meta Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2 text-xs text-slate-600 mt-2">
                  {h.meta?.project_name && <div className="col-span-2"><span className="text-slate-400">Project:</span> {h.meta.project_name}</div>}
                  {h.meta?.location && <div className="flex items-center gap-1"><MapPin className="h-3 w-3 text-slate-400" /> {h.meta.location}</div>}
                  {h.meta?.inspector && <div><span className="text-slate-400">Insp:</span> {h.meta.inspector}</div>}
                  {h.meta?.defect_side && <div><span className="text-slate-400">Side:</span> {h.meta.defect_side}</div>}
                </div>
              </div>

              {/* Defect List */}
              <div className="space-y-3">
                {h.items.map((it, i) => {
                  const sev = sevNormalize(it.severity);
                  const m = it.monitoring ?? { status: "pending", notes: "", photos: [] };
                  const saving = savingKey === `${h.id}#${i}`;

                  return (
                    <div key={i} className="flex flex-col sm:flex-row gap-4 p-3 rounded-lg border border-zinc-100 bg-slate-50/50 hover:bg-slate-50 transition-colors">
                      {/* Defect Info */}
                      <div className="flex-1 space-y-2">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className={`px-2 py-0.5 text-[11px] font-bold uppercase tracking-wide rounded-sm ${sev === "Severe" ? "bg-red-100 text-red-700 border border-red-200" :
                              sev === "Moderate" ? "bg-amber-100 text-amber-700 border border-amber-200" :
                                "bg-emerald-100 text-emerald-700 border border-emerald-200"
                            }`}>
                            {sev}
                          </span>
                          <span className="text-sm font-semibold text-slate-700">{it.cls}</span>
                          <span className="text-xs text-slate-400">{(it.conf * 100).toFixed(0)}%</span>
                        </div>

                        <p className="text-xs text-slate-600 line-clamp-2"><span className="font-medium text-slate-900">Solusi:</span> {it.solution}</p>
                      </div>

                      {/* Monitoring Control */}
                      <div className="flex-shrink-0 sm:w-[240px] flex flex-col gap-2 pt-2 sm:pt-0 sm:border-l sm:border-zinc-200 sm:pl-4">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-uppercase font-bold text-slate-400 tracking-wider">STATUS</span>
                          {m.updated_at && <span className="text-[9px] text-slate-400" title={m.updated_at}>Updated</span>}
                        </div>

                        <div className="flex gap-2">
                          <select
                            className={`flex-1 text-xs border-0 ring-1 ring-inset ring-zinc-300 rounded-md py-1.5 pl-2 bg-white focus:ring-2 focus:ring-primary sm:text-xs leading-5
                                     ${m.status === 'fixed' ? 'text-emerald-700 font-medium' : m.status === 'in_progress' ? 'text-blue-700 font-medium' : 'text-slate-600'}
                                   `}
                            value={m.status}
                            onChange={(e) => handleStatusChange(h.id, i, e.target.value as any)}
                            disabled={saving}
                          >
                            <option value="pending">Pending</option>
                            <option value="in_progress">On Progress</option>
                            <option value="fixed">Fixed</option>
                          </select>
                        </div>

                        <div className="relative">
                          <textarea
                            className="w-full text-xs bg-white border-zinc-200 rounded-md py-1.5 px-2 min-h-[50px] focus:border-primary focus:ring-primary"
                            placeholder="Add repair notes..."
                            defaultValue={m.notes || ""}
                            disabled={saving}
                            onBlur={(e) => handleSaveNotes(h.id, i, e.target.value)}
                          />
                        </div>

                        <div className="flex items-center justify-between gap-2">
                          <label className="cursor-pointer inline-flex items-center justify-center p-1.5 rounded hover:bg-zinc-200 transition-colors text-slate-500" title="Upload Repair Photos">
                            <Upload className="h-4 w-4" />
                            <input type="file" multiple accept="image/*" className="hidden" onChange={(e) => handleUploadPhotos(h.id, i, e.target.files)} />
                          </label>

                          {m.photos?.length > 0 && (
                            <div className="flex -space-x-2">
                              {m.photos.slice(0, 3).map((p, idx) => (
                                <img key={idx} src={`${BACKEND_URL}${p}`} className="w-6 h-6 rounded-full ring-1 ring-white object-cover" alt="" />
                              ))}
                              {m.photos.length > 3 && <span className="w-6 h-6 rounded-full bg-slate-100 flex items-center justify-center text-[9px] font-bold text-slate-500 ring-1 ring-white">+{m.photos.length - 3}</span>}
                            </div>
                          )}

                          {saving && <span className="loader h-3 w-3 border-2" />}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </motion.article>
        ))}
      </section>
    </main>
  );
}
