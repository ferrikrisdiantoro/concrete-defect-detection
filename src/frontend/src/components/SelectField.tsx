import React from "react";

type SelectFieldProps<T extends string = string> = {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: readonly T[] | string[];
};

export default function SelectField<T extends string = string>({
  label,
  value,
  onChange,
  options,
}: SelectFieldProps<T>) {
  return (
    <label className="flex flex-col gap-1">
      <span className="label">{label}</span>
      <select
        className="input"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((opt) => (
          <option key={String(opt)} value={String(opt)}>
            {String(opt) || "(blank)"}
          </option>
        ))}
      </select>
    </label>
  );
}
