type Props = {
  label: string;
  value: any;
  onChange: (v: any) => void;
  listId?: string;
  type?: string;
};

export default function Field({ label, value, onChange, listId, type = "text" }: Props) {
  return (
    <label className="flex flex-col gap-1">
      <span className="label">{label}</span>
      <input className="input" type={type} value={value} onChange={(e)=>onChange((e.target as HTMLInputElement).value)} list={listId}/>
    </label>
  );
}
