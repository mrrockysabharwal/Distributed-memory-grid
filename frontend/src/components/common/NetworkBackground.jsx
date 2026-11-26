const nodes = [
  [190, 160],
  [310, 235],
  [440, 130],
  [520, 285],
  [660, 150],
  [790, 250],
  [900, 120],
  [250, 410],
  [420, 470],
  [650, 430],
  [820, 500],
  [1030, 360],
  [1040, 540],
  [110, 340],
  [570, 70],
  [730, 80],
];

const links = [
  [0, 1],
  [0, 13],
  [1, 2],
  [1, 3],
  [1, 7],
  [2, 3],
  [2, 14],
  [2, 4],
  [3, 4],
  [3, 8],
  [3, 9],
  [4, 5],
  [4, 14],
  [4, 15],
  [5, 6],
  [5, 9],
  [5, 10],
  [5, 11],
  [6, 15],
  [7, 8],
  [8, 9],
  [9, 10],
  [9, 11],
  [10, 11],
  [10, 12],
  [11, 12],
  [13, 1],
  [14, 15],
];

export default function NetworkBackground() {
  return (
    <div aria-hidden="true" className="network-background pointer-events-none fixed inset-0 z-0 overflow-hidden">
      <div className="network-wash absolute inset-0" />
      <svg
        viewBox="0 0 1200 620"
        preserveAspectRatio="xMidYMid slice"
        className="network-lines absolute -left-[7.5%] -top-[7.5%] h-[115%] w-[115%] text-dmg-accent"
      >
        <g fill="none" stroke="currentColor" strokeWidth="1.25">
          {links.map(([from, to]) => (
            <line
              key={`${from}-${to}`}
              x1={nodes[from][0]}
              y1={nodes[from][1]}
              x2={nodes[to][0]}
              y2={nodes[to][1]}
            />
          ))}
        </g>
        <g fill="currentColor">
          {nodes.map(([cx, cy], index) => (
            <circle
              key={`${cx}-${cy}`}
              className="network-node"
              cx={cx}
              cy={cy}
              r={index % 4 === 0 ? 4 : 2.5}
              style={{ animationDelay: `${(index % 6) * 0.45}s` }}
            />
          ))}
        </g>
      </svg>
    </div>
  );
}
