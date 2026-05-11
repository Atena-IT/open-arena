import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.OPEN_ARENA_GUI_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
});
