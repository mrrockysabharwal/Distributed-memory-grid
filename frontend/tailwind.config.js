/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        dmg: {
          page: "rgb(var(--dmg-page) / <alpha-value>)",
          surface: "rgb(var(--dmg-surface) / <alpha-value>)",
          "surface-alt": "rgb(var(--dmg-surface-alt) / <alpha-value>)",
          text: "rgb(var(--dmg-text) / <alpha-value>)",
          muted: "rgb(var(--dmg-muted) / <alpha-value>)",
          subtle: "rgb(var(--dmg-subtle) / <alpha-value>)",
          line: "rgb(var(--dmg-line) / <alpha-value>)",
          accent: "rgb(var(--dmg-accent) / <alpha-value>)",
          "accent-hover": "rgb(var(--dmg-accent-hover) / <alpha-value>)",
          "accent-soft": "rgb(var(--dmg-accent-soft) / <alpha-value>)",
          "accent-text": "rgb(var(--dmg-accent-text) / <alpha-value>)",
          indicator: "rgb(var(--dmg-indicator) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: ["Segoe UI Variable", "Aptos", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: [
          "Segoe UI Variable Display",
          "Aptos Display",
          "Segoe UI Variable",
          "Inter",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};
