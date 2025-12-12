import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/analyze": "http://localhost:8000",
      "/history": "http://localhost:8000",
      "/uploads": "http://localhost:8000",
      "/export": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/diag": "http://localhost:8000",
    },
  },
});
