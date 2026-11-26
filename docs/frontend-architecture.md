# Frontend architecture

## Technology

The frontend is a React single-page application built with Vite. Its declared package versions are:

- React and React DOM `^18.2.0`
- Vite `^8.4.0` and `@vitejs/plugin-react` `^6.3.0`
- React Router DOM `^7.18.7`
- Tailwind CSS `^3.4.1`, PostCSS, and Autoprefixer
- Lucide React `^0.344.0`

Vite 8 requires Node.js `^20.19.0` or `>=22.12.0`. Tailwind is configured with the v3 PostCSS plugin.

## Directory structure

```text
frontend/
├── index.html
├── package.json
├── package-lock.json
├── postcss.config.js
├── tailwind.config.js
├── vite.config.js
└── src/
    ├── App.jsx
    ├── main.jsx
    ├── index.css
    ├── components/
    │   ├── common/
    │   │   ├── NetworkBackground.jsx
    │   │   └── PageHeader.jsx
    │   ├── dashboard/
    │   │   └── MetricCard.jsx
    │   └── layout/
    │       └── AppShell.jsx
    ├── pages/
    │   ├── Dashboard.jsx
    │   ├── Workers.jsx
    │   ├── Models.jsx
    │   ├── Inference.jsx
    │   └── JobHistory.jsx
    └── services/
        └── api.js
```

There are no separate `assets`, `hooks`, or `context` directories in the current frontend.

## Application routing

`src/main.jsx` mounts the app under `BrowserRouter`. `src/App.jsx` defines the routes inside the shared `AppShell`; unknown paths redirect to `/`.

| Path | Page | Current content |
| --- | --- | --- |
| `/` | Dashboard | DMG introduction and static “Not connected” / “No data” status summaries |
| `/workers` | Workers | Empty state; no live worker list |
| `/models` | Models | Empty state; no connected model catalog |
| `/inference` | Inference | Local prompt workspace; it does not submit requests |
| `/jobs` | Job History | Empty state; no frontend job records |

## Shared components

- **`AppShell`** — Provides the responsive top navigation, brand, mobile menu, theme toggle, cluster status indicator, routed content outlet, and footer. The cluster indicator is always the neutral “Not connected” state; it is not backed by a health check.
- **`PageHeader`** — Renders a page eyebrow, title, description, and optional action for the standard content pages. The Dashboard uses its own hero layout.
- **`NetworkBackground`** — Renders the shared decorative SVG node-and-link grid behind all routes. CSS animates the graph drift and node opacity; `prefers-reduced-motion: reduce` disables those animations.
- **`MetricCard`** — A reusable presentational row for a label, value, and detail. It is not currently used by the routed pages and does not fetch or calculate metrics.

## Theme

The theme is defined by CSS custom properties in `src/index.css`. The light palette is the stylesheet fallback; `AppShell` initializes the app in dark mode unless `localStorage` contains the value `light` under the key `dmg-theme-v2`. The theme toggle switches between `dark` and `light`, updates `document.documentElement.dataset.theme`, and persists the selection in localStorage. The browser theme-color metadata is also updated when the theme changes.

## API service

`src/services/api.js` exports one constant, `API_BASE_URL`. It reads `import.meta.env.VITE_API_BASE_URL`, falls back to `http://localhost:8000/api`, and removes trailing slashes from the selected value.

There are currently **no API request functions** in this service, and the routed pages do not import or call it. No frontend authentication, endpoint client, or backend integration is implemented. The repository's root `.env.example` contains backend settings and does not define `VITE_API_BASE_URL`; an optional frontend override can be provided through Vite's environment configuration.

## Current implementation status

Functional frontend behavior includes client-side routing, responsive navigation, theme switching and persistence, the shared decorative network background, and local prompt input validation. The non-empty inference form displays a notice that the prompt has not been sent; it does not contact the backend.

Worker, model, cluster, metric, generated-output, and job-history content is absent or explicitly presented as unavailable. There is no live cluster connectivity, backend authentication flow, real-time streaming, or API-driven dashboard data in the current frontend. Backend routes exist separately, but their presence does not mean they are connected to these pages.

## Development

From the `frontend/` directory, using a supported Node.js version:

```sh
npm install
npm run dev
npm run build
```

`npm run dev` starts the Vite development server; `npm run build` creates the production bundle. The frontend package currently has no test script.
