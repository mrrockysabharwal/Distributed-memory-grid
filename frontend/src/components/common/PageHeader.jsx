export default function PageHeader({ eyebrow, title, description, action }) {
  return (
    <div className="mb-10 flex flex-col justify-between gap-6 sm:flex-row sm:items-end">
      <div>
        {eyebrow && (
          <p className="mb-3 text-xs font-bold uppercase tracking-[0.2em] text-dmg-accent">
            {eyebrow}
          </p>
        )}
        <h1 className="max-w-3xl font-display text-4xl font-semibold leading-[1.08] tracking-[-0.045em] text-dmg-text sm:text-5xl">
          {title}
        </h1>
        {description && (
          <p className="mt-4 max-w-2xl text-base leading-7 text-dmg-muted">{description}</p>
        )}
      </div>
      {action}
    </div>
  );
}
