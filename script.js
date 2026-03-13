const PALETTE = ["#177dff", "#ff8a62", "#1b9978", "#f0b34a", "#7c69ff", "#0fb9c8"];
const NEUTRAL = "#9aa3af";
const SURFACE = "rgba(255,255,255,0.58)";
const GRID = "rgba(98,105,117,0.16)";

function mulberry32(seed) {
  let t = seed >>> 0;
  return function next() {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function generateGaussianSet(specs, seed) {
  const rng = mulberry32(seed);
  const points = [];
  specs.forEach((spec) => {
    for (let index = 0; index < spec.count; index += 1) {
      points.push([
        spec.center[0] + gaussian(rng) * spec.spread[0],
        spec.center[1] + gaussian(rng) * spec.spread[1],
      ]);
    }
  });
  return points;
}

function clonePoint(point) {
  return point.slice();
}

function meanPoint(points) {
  const dims = points[0].length;
  const sums = Array(dims).fill(0);
  points.forEach((point) => {
    point.forEach((value, index) => {
      sums[index] += value;
    });
  });
  return sums.map((value) => value / points.length);
}

function squaredDistance(a, b) {
  let total = 0;
  for (let index = 0; index < a.length; index += 1) {
    total += (a[index] - b[index]) ** 2;
  }
  return total;
}

function euclideanDistance(a, b) {
  return Math.sqrt(squaredDistance(a, b));
}

function correlationDistance(a, b) {
  const meanA = a.reduce((sum, value) => sum + value, 0) / a.length;
  const meanB = b.reduce((sum, value) => sum + value, 0) / b.length;
  const centeredA = a.map((value) => value - meanA);
  const centeredB = b.map((value) => value - meanB);
  const normA = Math.sqrt(centeredA.reduce((sum, value) => sum + value * value, 0));
  const normB = Math.sqrt(centeredB.reduce((sum, value) => sum + value * value, 0));
  if (!normA || !normB) return 1;
  const correlation =
    centeredA.reduce((sum, value, index) => sum + value * centeredB[index], 0) / (normA * normB);
  return 1 - correlation;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function formatPercent(value, digits = 0) {
  return `${formatNumber(value * 100, digits)}%`;
}

function standardize(points) {
  const dims = points[0].length;
  const means = Array(dims).fill(0);
  const sds = Array(dims).fill(0);

  points.forEach((point) => {
    point.forEach((value, index) => {
      means[index] += value;
    });
  });

  for (let index = 0; index < dims; index += 1) {
    means[index] /= points.length;
  }

  points.forEach((point) => {
    point.forEach((value, index) => {
      sds[index] += (value - means[index]) ** 2;
    });
  });

  for (let index = 0; index < dims; index += 1) {
    sds[index] = Math.sqrt(sds[index] / Math.max(points.length - 1, 1)) || 1;
  }

  const scaled = points.map((point) =>
    point.map((value, index) => (value - means[index]) / sds[index]),
  );
  return { scaled, means, sds };
}

function sampleUnique(points, k, rng) {
  const pool = points.map((_, index) => index);
  const chosen = [];
  for (let count = 0; count < k; count += 1) {
    const pick = Math.floor(rng() * pool.length);
    chosen.push(clonePoint(points[pool[pick]]));
    pool.splice(pick, 1);
  }
  return chosen;
}

function nearestCenterIndex(point, centers) {
  let bestIndex = 0;
  let bestDistance = Infinity;
  centers.forEach((center, index) => {
    const distance = squaredDistance(point, center);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function clusterSizes(assignments, k) {
  const sizes = Array(k).fill(0);
  assignments.forEach((cluster) => {
    sizes[cluster] += 1;
  });
  return sizes;
}

function computeCentroids(points, assignments, k, rng) {
  const clusters = Array.from({ length: k }, () => []);
  points.forEach((point, index) => {
    clusters[assignments[index]].push(point);
  });

  return clusters.map((cluster) => {
    if (cluster.length) return meanPoint(cluster);
    return clonePoint(points[Math.floor(rng() * points.length)]);
  });
}

function computeInertia(points, assignments, centroids) {
  return points.reduce((total, point, index) => {
    return total + squaredDistance(point, centroids[assignments[index]]);
  }, 0);
}

function countDifferences(left, right) {
  let count = 0;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) count += 1;
  }
  return count;
}

function kmeansTrace(points, k, seed, maxIterations = 8) {
  const rng = mulberry32(seed);
  const startAssignments = points.map(() => Math.floor(rng() * k));
  const frames = [
    { title: "Choose K", step: 0, assignments: null, centroids: [], moved: 0, loop: 0, inertia: null },
    {
      title: "Random start",
      step: 1,
      assignments: startAssignments.slice(),
      centroids: [],
      moved: points.length,
      loop: 0,
      inertia: null,
    },
  ];

  let assignments = startAssignments.slice();
  let centroids = computeCentroids(points, assignments, k, rng);

  frames.push({
    title: "Compute centroids",
    step: 2,
    assignments: assignments.slice(),
    centroids: centroids.map(clonePoint),
    moved: 0,
    loop: 0,
    inertia: computeInertia(points, assignments, centroids),
  });

  for (let loop = 1; loop <= maxIterations; loop += 1) {
    const nextAssignments = points.map((point) => nearestCenterIndex(point, centroids));
    const moved = countDifferences(assignments, nextAssignments);

    frames.push({
      title: loop === 1 ? "Reassign points" : "Repeat",
      step: 3,
      assignments: nextAssignments.slice(),
      centroids: centroids.map(clonePoint),
      moved,
      loop,
      inertia: computeInertia(points, nextAssignments, centroids),
    });

    centroids = computeCentroids(points, nextAssignments, k, rng);
    assignments = nextAssignments.slice();

    frames.push({
      title: moved === 0 ? "Converged" : "Repeat",
      step: 4,
      assignments: assignments.slice(),
      centroids: centroids.map(clonePoint),
      moved,
      loop,
      inertia: computeInertia(points, assignments, centroids),
    });

    if (moved === 0) break;
  }

  return {
    frames,
    finalAssignments: assignments.slice(),
    finalCentroids: centroids.map(clonePoint),
  };
}

function kmeansRun(points, k, seed, maxIterations = 50) {
  const rng = mulberry32(seed);
  let centroids = sampleUnique(points, k, rng);
  const initialCentroids = centroids.map(clonePoint);
  let assignments = Array(points.length).fill(-1);

  for (let loop = 0; loop < maxIterations; loop += 1) {
    const nextAssignments = points.map((point) => nearestCenterIndex(point, centroids));
    if (countDifferences(assignments, nextAssignments) === 0) {
      assignments = nextAssignments;
      break;
    }
    assignments = nextAssignments;
    centroids = computeCentroids(points, assignments, k, rng);
  }

  const inertia = computeInertia(points, assignments, centroids);
  return { assignments, centroids, inertia, seed, initialCentroids };
}

function silhouetteScore(points, assignments) {
  const groups = new Map();
  assignments.forEach((cluster, index) => {
    if (!groups.has(cluster)) groups.set(cluster, []);
    groups.get(cluster).push(index);
  });

  if (groups.size < 2) return 0;

  const distances = points.map((point, row) =>
    points.map((other, col) => (row === col ? 0 : euclideanDistance(point, other))),
  );

  let total = 0;
  points.forEach((_, index) => {
    const cluster = assignments[index];
    const ownGroup = groups.get(cluster);
    const intra =
      ownGroup.length <= 1
        ? 0
        : ownGroup.reduce((sum, member) => (member === index ? sum : sum + distances[index][member]), 0) /
          (ownGroup.length - 1);

    let nearestOther = Infinity;
    groups.forEach((members, key) => {
      if (key === cluster) return;
      const candidate =
        members.reduce((sum, member) => sum + distances[index][member], 0) / Math.max(members.length, 1);
      if (candidate < nearestOther) nearestOther = candidate;
    });

    const denominator = Math.max(intra, nearestOther);
    total += denominator === 0 ? 0 : (nearestOther - intra) / denominator;
  });

  return total / points.length;
}

function clusterSummaries(points, assignments) {
  const k = Math.max(...assignments) + 1;
  return Array.from({ length: k }, (_, index) => {
    const members = points.filter((_, pointIndex) => assignments[pointIndex] === index);
    const centroid = meanPoint(members);
    const spread = Math.sqrt(
      members.reduce((sum, point) => sum + squaredDistance(point, centroid), 0) / Math.max(members.length, 1),
    );
    return {
      index,
      size: members.length,
      share: members.length / points.length,
      centroid,
      spread,
      members,
    };
  });
}

function hierarchicalClustering(points, options) {
  const linkage = options.linkage || "average";
  const distanceKind = options.distance || "euclidean";
  const distanceFn = distanceKind === "correlation" ? correlationDistance : euclideanDistance;
  let nextId = points.length;
  let step = 0;

  function clusterDistance(clusterA, clusterB) {
    if (linkage === "centroid") {
      return distanceFn(clusterA.centroid, clusterB.centroid);
    }

    const distances = [];
    clusterA.members.forEach((memberA) => {
      clusterB.members.forEach((memberB) => {
        distances.push(distanceFn(points[memberA], points[memberB]));
      });
    });

    if (linkage === "single") return Math.min(...distances);
    if (linkage === "complete") return Math.max(...distances);
    return distances.reduce((sum, value) => sum + value, 0) / distances.length;
  }

  let clusters = points.map((point, index) => ({
    id: index,
    members: [index],
    centroid: clonePoint(point),
    left: null,
    right: null,
    height: 0,
    step: 0,
    minMember: index,
  }));

  const merges = [];
  const clusterStates = [clusters.map((cluster) => cluster.members.slice())];

  while (clusters.length > 1) {
    let bestPair = { distance: Infinity, i: 0, j: 1 };

    for (let row = 0; row < clusters.length; row += 1) {
      for (let col = row + 1; col < clusters.length; col += 1) {
        const distance = clusterDistance(clusters[row], clusters[col]);
        if (distance < bestPair.distance) {
          bestPair = { distance, i: row, j: col };
        }
      }
    }

    const first = clusters[bestPair.i];
    const second = clusters[bestPair.j];
    const [left, right] = first.minMember <= second.minMember ? [first, second] : [second, first];
    step += 1;

    const members = left.members.concat(right.members);
    const centroid = meanPoint(members.map((member) => points[member]));
    const merged = {
      id: nextId,
      members,
      centroid,
      left,
      right,
      height: bestPair.distance,
      step,
      minMember: Math.min(left.minMember, right.minMember),
    };

    nextId += 1;
    merges.push({
      step,
      height: bestPair.distance,
      members: members.slice(),
      left: left.members.slice(),
      right: right.members.slice(),
    });

    clusters = clusters.filter((_, index) => index !== bestPair.i && index !== bestPair.j);
    clusters.push(merged);
    clusters.sort((a, b) => a.minMember - b.minMember);
    clusterStates.push(clusters.map((cluster) => cluster.members.slice()));
  }

  const root = clusters[0];
  const leafOrder = [];

  function collectLeaves(node) {
    if (!node.left || !node.right) {
      leafOrder.push(node.id);
      return;
    }
    collectLeaves(node.left);
    collectLeaves(node.right);
  }

  collectLeaves(root);

  let inversion = false;
  function checkInversions(node) {
    if (!node.left || !node.right) return;
    if (node.height < node.left.height || node.height < node.right.height) inversion = true;
    checkInversions(node.left);
    checkInversions(node.right);
  }

  checkInversions(root);

  return {
    root,
    merges,
    clusterStates,
    leafOrder,
    maxHeight: root.height || 1,
    inversion,
  };
}

function cutTree(root, cutHeight) {
  if (!root.left || !root.right) return [root.members.slice()];
  if (root.height <= cutHeight) return [root.members.slice()];
  return cutTree(root.left, cutHeight).concat(cutTree(root.right, cutHeight));
}

function groupsToAssignments(groups, total) {
  const assignments = Array(total).fill(-1);
  const orderedGroups = groups.slice().sort((a, b) => Math.min(...a) - Math.min(...b));
  orderedGroups.forEach((group, index) => {
    group.forEach((member) => {
      assignments[member] = index;
    });
  });
  return assignments;
}

function getCssVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim() || "#177dff";
}

function getBounds(points) {
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padX = (maxX - minX || 1) * 0.16;
  const padY = (maxY - minY || 1) * 0.16;
  return {
    minX: minX - padX,
    maxX: maxX + padX,
    minY: minY - padY,
    maxY: maxY + padY,
  };
}

function mapScatterPoints(points, width, height, padding) {
  const bounds = getBounds(points);
  return points.map((point) => {
    const x = padding + ((point[0] - bounds.minX) / (bounds.maxX - bounds.minX || 1)) * (width - padding * 2);
    const y =
      height -
      padding -
      ((point[1] - bounds.minY) / (bounds.maxY - bounds.minY || 1)) * (height - padding * 2);
    return [x, y];
  });
}

function mapSinglePoint(point, bounds, width, height, padding) {
  const x = padding + ((point[0] - bounds.minX) / (bounds.maxX - bounds.minX || 1)) * (width - padding * 2);
  const y =
    height -
    padding -
    ((point[1] - bounds.minY) / (bounds.maxY - bounds.minY || 1)) * (height - padding * 2);
  return [x, y];
}

function scatterSvg(options) {
  const width = options.width || 620;
  const height = options.height || 440;
  const padding = options.padding || 48;
  const allPoints = []
    .concat(options.points || [])
    .concat(options.centroids || [])
    .concat(options.ghostCentroids || [])
    .concat(options.initialCentroids || []);
  const bounds = getBounds(allPoints.length ? allPoints : [[0, 0]]);
  const mappedPoints = (options.points || []).map((point) =>
    mapSinglePoint(point, bounds, width, height, padding),
  );
  const accent = getCssVar("--accent");
  const accentAlt = getCssVar("--accent-alt");
  const axes = options.axes;

  const gridLines = Array.from({ length: 4 }, (_, index) => {
    const x = padding + ((width - padding * 2) / 3) * index;
    const y = padding + ((height - padding * 2) / 3) * index;
    return `
      <line x1="${x}" y1="${padding}" x2="${x}" y2="${height - padding}" stroke="${GRID}" stroke-width="1" />
      <line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="${GRID}" stroke-width="1" />
    `;
  }).join("");

  const axisLabels = axes
    ? `
      <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="${GRID}" stroke-width="1.2" />
      <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="${GRID}" stroke-width="1.2" />
      <text x="${width - padding}" y="${height - padding + 28}" text-anchor="end" fill="${NEUTRAL}" font-size="12" font-weight="600">${axes.xLabel}</text>
      <text x="${padding - 28}" y="${padding}" text-anchor="start" fill="${NEUTRAL}" font-size="12" font-weight="600" transform="rotate(-90 ${padding - 28} ${padding})">${axes.yLabel}</text>
      <text x="${padding}" y="${height - padding + 28}" fill="${NEUTRAL}" font-size="11">${formatNumber(bounds.minX, 0)}</text>
      <text x="${width - padding}" y="${height - padding + 28}" text-anchor="end" fill="${NEUTRAL}" font-size="11">${formatNumber(bounds.maxX, 0)}</text>
      <text x="${padding - 10}" y="${height - padding}" text-anchor="end" fill="${NEUTRAL}" font-size="11">${formatNumber(bounds.minY, 1)}</text>
      <text x="${padding - 10}" y="${padding + 4}" text-anchor="end" fill="${NEUTRAL}" font-size="11">${formatNumber(bounds.maxY, 1)}</text>
    `
    : "";

  const ghostCentroids = (options.ghostCentroids || [])
    .map((center, index) => {
      const [x, y] = mapSinglePoint(center, bounds, width, height, padding);
      const color = PALETTE[index % PALETTE.length];
      return `
        <circle cx="${x}" cy="${y}" r="14" fill="none" stroke="${color}" stroke-width="2.2" stroke-dasharray="6 6" opacity="0.5" />
      `;
    })
    .join("");

  const initialCentroids = (options.initialCentroids || [])
    .map((center, index) => {
      const [x, y] = mapSinglePoint(center, bounds, width, height, padding);
      const color = PALETTE[index % PALETTE.length];
      return `
        <circle cx="${x}" cy="${y}" r="7" fill="none" stroke="${color}" stroke-width="2" opacity="0.45" />
      `;
    })
    .join("");

  const pointsLayer = mappedPoints
    .map(([x, y], index) => {
      const cluster = options.assignments ? options.assignments[index] : -1;
      const highlighted =
        options.highlightCluster === null ||
        options.highlightCluster === undefined ||
        cluster === options.highlightCluster;
      const fill = cluster >= 0 ? PALETTE[cluster % PALETTE.length] : "rgba(154,163,175,0.85)";
      const opacity = highlighted ? 0.95 : 0.16;
      const radius = options.outlierIndex === index ? 8 : 6.4;
      const stroke = options.outlierIndex === index ? "#11151d" : "rgba(255,255,255,0.9)";
      return `
        <circle cx="${x}" cy="${y}" r="${radius}" fill="${fill}" opacity="${opacity}" stroke="${stroke}" stroke-width="1.3" />
      `;
    })
    .join("");

  const centroids = (options.centroids || [])
    .map((center, index) => {
      const [x, y] = mapSinglePoint(center, bounds, width, height, padding);
      const color = PALETTE[index % PALETTE.length];
      return `
        <circle cx="${x}" cy="${y}" r="11.5" fill="${color}" fill-opacity="0.18" stroke="${color}" stroke-width="2.2" />
        <path d="M ${x - 8} ${y} H ${x + 8} M ${x} ${y - 8} V ${y + 8}" stroke="${color}" stroke-width="2.2" stroke-linecap="round" />
      `;
    })
    .join("");

  const annotations = options.note
    ? `<text x="${padding}" y="${padding - 14}" fill="${accent}" font-size="12" font-weight="700">${options.note}</text>`
    : "";

  const legend = options.legend
    ? `
      <g transform="translate(${width - padding - 150}, ${padding - 8})">
        <rect x="0" y="0" width="150" height="24" rx="12" fill="rgba(255,255,255,0.82)" />
        <circle cx="18" cy="12" r="5" fill="${accent}" />
        <text x="30" y="16" fill="${NEUTRAL}" font-size="11">${options.legend}</text>
      </g>
    `
    : "";

  return `
    <rect x="0" y="0" width="${width}" height="${height}" rx="28" fill="${SURFACE}" />
    ${gridLines}
    ${axisLabels}
    ${annotations}
    ${legend}
    ${ghostCentroids}
    ${initialCentroids}
    ${pointsLayer}
    ${centroids}
    ${
      options.outlierPoint
        ? (() => {
            const [x, y] = mapSinglePoint(options.outlierPoint, bounds, width, height, padding);
            return `<text x="${x + 12}" y="${y - 10}" fill="${accentAlt}" font-size="12" font-weight="700">outlier</text>`;
          })()
        : ""
    }
  `;
}

function lineChartSvg(values, options = {}) {
  const width = options.width || 300;
  const height = options.height || 180;
  const padding = 26;
  const accent = options.color || getCssVar("--accent");
  const minY = options.minY !== undefined ? options.minY : Math.min(...values);
  const maxY = options.maxY !== undefined ? options.maxY : Math.max(...values);
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;

  const coords = values.map((value, index) => {
    const x = padding + (index / Math.max(values.length - 1, 1)) * usableWidth;
    const y = height - padding - ((value - minY) / (maxY - minY || 1)) * usableHeight;
    return [x, y];
  });

  const path = coords.map(([x, y], index) => `${index === 0 ? "M" : "L"} ${x} ${y}`).join(" ");

  const dots = coords
    .map(([x, y], index) => {
      const selected = options.selectedIndex === index;
      const radius = selected ? 5.5 : 4;
      const fill = selected ? accent : "rgba(255,255,255,0.92)";
      const stroke = selected ? accent : "rgba(98,105,117,0.34)";
      return `<circle cx="${x}" cy="${y}" r="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="2" />`;
    })
    .join("");

  const labels = (options.labels || [])
    .map((label, index) => {
      const [x] = coords[index];
      return `<text x="${x}" y="${height - 6}" text-anchor="middle" fill="${NEUTRAL}" font-size="11">${label}</text>`;
    })
    .join("");

  return `
    <rect x="0" y="0" width="${width}" height="${height}" rx="22" fill="${SURFACE}" />
    <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="${GRID}" stroke-width="1.1" />
    <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="${GRID}" stroke-width="1.1" />
    <path d="${path}" fill="none" stroke="${accent}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" />
    ${dots}
    ${labels}
    ${
      options.title
        ? `<text x="${padding}" y="18" fill="${NEUTRAL}" font-size="12" font-weight="700">${options.title}</text>`
        : ""
    }
  `;
}

function layoutDendrogram(root, width, height, padding = 26) {
  const leaves = [];
  function collectLeaves(node) {
    if (!node.left || !node.right) {
      leaves.push(node.id);
      return;
    }
    collectLeaves(node.left);
    collectLeaves(node.right);
  }
  collectLeaves(root);

  const positions = new Map();
  const stepX = leaves.length > 1 ? (width - padding * 2) / (leaves.length - 1) : 0;
  const baseY = height - padding;
  const maxHeight = root.height || 1;

  function place(node) {
    if (!node.left || !node.right) {
      const x = padding + leaves.indexOf(node.id) * stepX;
      const position = { x, y: baseY };
      positions.set(node.id, position);
      return position;
    }

    const left = place(node.left);
    const right = place(node.right);
    const y = baseY - (node.height / maxHeight) * (height - padding * 2);
    const position = { x: (left.x + right.x) / 2, y };
    positions.set(node.id, position);
    return position;
  }

  place(root);
  return { positions, leaves, baseY, maxHeight, padding };
}

function dendrogramSvg(clustering, options = {}) {
  const width = options.width || 290;
  const height = options.height || 280;
  const accent = getCssVar("--accent");
  const accentAlt = getCssVar("--accent-alt");
  const layout = layoutDendrogram(clustering.root, width, height, 28);
  const { positions, leaves, baseY, maxHeight, padding } = layout;
  const visibleStep = options.visibleStep ?? Infinity;
  const currentStep = options.currentStep ?? null;
  const assignments = options.assignments || null;
  const leafLabels = options.labels || leaves.map((leaf) => `${leaf + 1}`);

  const segments = [];
  function collectSegments(node) {
    if (!node.left || !node.right) return;
    const nodePos = positions.get(node.id);
    const leftPos = positions.get(node.left.id);
    const rightPos = positions.get(node.right.id);
    segments.push({
      x1: leftPos.x,
      y1: leftPos.y,
      x2: leftPos.x,
      y2: nodePos.y,
      step: node.step,
    });
    segments.push({
      x1: rightPos.x,
      y1: rightPos.y,
      x2: rightPos.x,
      y2: nodePos.y,
      step: node.step,
    });
    segments.push({
      x1: leftPos.x,
      y1: nodePos.y,
      x2: rightPos.x,
      y2: nodePos.y,
      step: node.step,
      mergeHeight: node.height,
    });
    collectSegments(node.left);
    collectSegments(node.right);
  }
  collectSegments(clustering.root);

  const lines = segments
    .map((segment) => {
      const active = segment.step <= visibleStep;
      const highlighted = currentStep !== null && segment.step === currentStep;
      return `
        <line
          x1="${segment.x1}"
          y1="${segment.y1}"
          x2="${segment.x2}"
          y2="${segment.y2}"
          stroke="${highlighted ? accentAlt : active ? accent : GRID}"
          stroke-width="${highlighted ? 3.4 : active ? 2.3 : 1.4}"
          opacity="${active ? 1 : 0.28}"
          stroke-linecap="round"
        />
      `;
    })
    .join("");

  const leafDots = leaves
    .map((leafId, index) => {
      const pos = positions.get(leafId);
      const color = assignments ? PALETTE[assignments[leafId] % PALETTE.length] : "rgba(154,163,175,0.8)";
      const focused =
        options.focusCluster === null ||
        options.focusCluster === undefined ||
        !assignments ||
        assignments[leafId] === options.focusCluster;
      return `
        <circle cx="${pos.x}" cy="${baseY}" r="5.2" fill="${color}" opacity="${focused ? 1 : 0.22}" stroke="rgba(255,255,255,0.95)" stroke-width="1.4" />
        <text x="${pos.x}" y="${height - 6}" text-anchor="middle" fill="${NEUTRAL}" opacity="${focused ? 1 : 0.32}" font-size="10">${leafLabels[index]}</text>
      `;
    })
    .join("");

  const cutLine =
    options.cutHeight !== null && options.cutHeight !== undefined
      ? (() => {
          const y = baseY - (options.cutHeight / (maxHeight || 1)) * (height - padding * 2);
          return `
            <line x1="${padding - 4}" y1="${y}" x2="${width - padding + 4}" y2="${y}" stroke="${accentAlt}" stroke-width="2" stroke-dasharray="7 7" />
            <text x="${width - padding}" y="${Math.max(y - 8, 16)}" text-anchor="end" fill="${accentAlt}" font-size="11" font-weight="700">cut = ${formatNumber(options.cutHeight, 2)}</text>
          `;
        })()
      : "";

  return `
    <rect x="0" y="0" width="${width}" height="${height}" rx="24" fill="${SURFACE}" />
    ${lines}
    ${cutLine}
    ${leafDots}
  `;
}

function profilePath(values, width, height, padding) {
  const maxValue = Math.max(...values);
  const minValue = Math.min(...values);
  return values
    .map((value, index) => {
      const x = padding + (index / Math.max(values.length - 1, 1)) * (width - padding * 2);
      const y = height - padding - ((value - minValue) / (maxValue - minValue || 1)) * (height - padding * 2);
      return `${index === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");
}

function renderProfileCards(container, profiles, assignments, labels) {
  container.innerHTML = profiles
    .map((profile, index) => {
      const color = PALETTE[assignments[index] % PALETTE.length];
      return `
        <div class="profile-card">
          <header>
            <strong>${labels[index]}</strong>
            <span>cluster ${assignments[index] + 1}</span>
          </header>
          <svg viewBox="0 0 160 90" class="plot plot-small" role="img" aria-label="Profile ${labels[index]}">
            <rect x="0" y="0" width="160" height="90" rx="18" fill="${SURFACE}" />
            <path d="${profilePath(profile, 160, 90, 16)}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </div>
      `;
    })
    .join("");
}

function setSvg(svg, markup) {
  svg.innerHTML = markup;
}

function buildDatasets() {
  const kmeansMain = generateGaussianSet(
    [
      { center: [-2.8, -1.2], count: 14, spread: [0.45, 0.35] },
      { center: [0.4, 2.5], count: 12, spread: [0.55, 0.45] },
      { center: [2.8, -0.4], count: 14, spread: [0.5, 0.4] },
    ],
    12,
  );

  const hierMain = generateGaussianSet(
    [
      { center: [-2.6, -1.1], count: 4, spread: [0.28, 0.26] },
      { center: [-0.1, 2.15], count: 4, spread: [0.32, 0.28] },
      { center: [2.6, -0.3], count: 4, spread: [0.28, 0.26] },
    ],
    27,
  );

  const linkageData = [
    [-3.3, 0.05],
    [-3.0, 0.45],
    [-2.85, -0.42],
    [-1.4, 0.16],
    [-0.35, 0.06],
    [0.85, -0.08],
    [1.9, 0.06],
    [3.05, 0.38],
    [3.35, -0.18],
    [3.65, 0.08],
  ];

  const scaleData = [
    [15, 0.8],
    [26, 1.0],
    [38, 1.2],
    [54, 1.1],
    [70, 1.35],
    [88, 0.95],
    [18, 3.55],
    [31, 3.95],
    [47, 4.15],
    [60, 3.7],
    [77, 4.02],
    [94, 3.88],
  ];

  const outlierBase = generateGaussianSet(
    [
      { center: [-2.0, -0.05], count: 12, spread: [0.45, 0.36] },
      { center: [2.05, 0.1], count: 12, spread: [0.45, 0.36] },
    ],
    9,
  );

  const outlierPoint = [6.5, 2.6];

  const profileData = [
    [1, 2, 3, 4],
    [5, 10, 15, 20],
    [4, 3, 2, 1],
    [20, 15, 10, 5],
    [2, 2, 2, 2],
    [9, 9, 9, 9],
  ];

  return {
    kmeansMain,
    hierMain,
    linkageData,
    scaleData,
    outlierBase,
    outlierPoint,
    profileData,
    profileLabels: ["A", "B", "C", "D", "E", "F"],
    hierLabels: Array.from({ length: hierMain.length }, (_, index) => `${index + 1}`),
  };
}

function buildCache(data) {
  const kDiagnostics = [];
  for (let k = 1; k <= 8; k += 1) {
    let best = null;
    for (let seed = 1; seed <= 28; seed += 1) {
      const run = kmeansRun(data.kmeansMain, k, seed);
      if (!best || run.inertia < best.inertia) best = run;
    }
    kDiagnostics.push({
      k,
      run: best,
      inertia: best.inertia,
      silhouette: k === 1 ? 0 : silhouetteScore(data.kmeansMain, best.assignments),
      summaries: clusterSummaries(data.kmeansMain, best.assignments),
    });
  }

  const randomCases = [];
  for (let seed = 1; seed <= 24; seed += 1) {
    randomCases.push(kmeansRun(data.kmeansMain, 3, seed));
  }
  randomCases.sort((left, right) => left.inertia - right.inertia);

  const scaleInfo = standardize(data.scaleData);

  const hierAverage = hierarchicalClustering(data.hierMain, { linkage: "average", distance: "euclidean" });
  const hierComplete = hierarchicalClustering(data.hierMain, { linkage: "complete", distance: "euclidean" });

  const linkageResults = {
    complete: hierarchicalClustering(data.linkageData, { linkage: "complete", distance: "euclidean" }),
    single: hierarchicalClustering(data.linkageData, { linkage: "single", distance: "euclidean" }),
    average: hierarchicalClustering(data.linkageData, { linkage: "average", distance: "euclidean" }),
    centroid: hierarchicalClustering(data.linkageData, { linkage: "centroid", distance: "euclidean" }),
  };

  const distanceResults = {
    euclidean: hierarchicalClustering(data.profileData, { linkage: "average", distance: "euclidean" }),
    correlation: hierarchicalClustering(data.profileData, { linkage: "average", distance: "correlation" }),
  };

  const scaleRawHier = hierarchicalClustering(data.scaleData, { linkage: "average", distance: "euclidean" });
  const scaleScaledHier = hierarchicalClustering(scaleInfo.scaled, { linkage: "average", distance: "euclidean" });

  const outlierCleanHier = hierarchicalClustering(data.outlierBase, { linkage: "complete", distance: "euclidean" });
  const outlierDirtyHier = hierarchicalClustering(data.outlierBase.concat([data.outlierPoint]), {
    linkage: "complete",
    distance: "euclidean",
  });

  return {
    kDiagnostics,
    bestKRuns: Object.fromEntries(kDiagnostics.map((entry) => [entry.k, entry.run])),
    randomCases,
    scaleInfo,
    hierAverage,
    hierComplete,
    linkageResults,
    distanceResults,
    scaleRawHier,
    scaleScaledHier,
    outlierCleanHier,
    outlierDirtyHier,
  };
}

const data = buildDatasets();
const cache = buildCache(data);

const state = {
  activeTab: "kmeans",
  kmeansReveal: false,
  kmeansAlgorithm: {
    k: 3,
    step: 0,
    playing: false,
    timer: null,
  },
  kmeansSelectedK: 3,
  kmeansRandomIndex: Math.max(cache.randomCases.length - 1, 0),
  kmeansScaleMode: "raw",
  kmeansOutlierMode: "hide",
  kmeansFocusCluster: 0,
  hierMergeStep: 3,
  hierCutPercent: 55,
  hierLinkage: "complete",
  hierDistance: "euclidean",
  hierScaleMode: "raw",
  hierOutlierMode: "hide",
  hierFinalPercent: 58,
  hierFocusCluster: 0,
};

const heroTrace = kmeansTrace(data.kmeansMain, 3, 30);

const refs = {};

function bindElements() {
  [
    "kmeans-hero-plot",
    "kmeans-unsupervised-plot",
    "kmeans-unsupervised-toggle",
    "kmeans-algorithm-k",
    "kmeans-algorithm-k-value",
    "kmeans-step-prev",
    "kmeans-step-next",
    "kmeans-step-play",
    "kmeans-step-list",
    "kmeans-algorithm-title",
    "kmeans-algorithm-subtitle",
    "kmeans-algorithm-plot",
    "kmeans-algorithm-moved",
    "kmeans-algorithm-inertia",
    "kmeans-algorithm-loop",
    "kmeans-k-slider",
    "kmeans-k-slider-value",
    "kmeans-k-plot",
    "kmeans-elbow-plot",
    "kmeans-silhouette-plot",
    "kmeans-k-inertia",
    "kmeans-k-silhouette",
    "kmeans-k-balance",
    "kmeans-random-reroll",
    "kmeans-random-current",
    "kmeans-random-best",
    "kmeans-random-summary",
    "kmeans-random-current-inertia",
    "kmeans-random-best-inertia",
    "kmeans-random-gap",
    "kmeans-scale-controls",
    "kmeans-scale-plot",
    "kmeans-scale-note",
    "kmeans-scale-basis",
    "kmeans-scale-story",
    "kmeans-outlier-controls",
    "kmeans-outlier-plot",
    "kmeans-outlier-note",
    "kmeans-outlier-shift",
    "kmeans-outlier-inertia",
    "kmeans-outlier-story",
    "kmeans-final-plot",
    "kmeans-cluster-cards",
    "hier-hero-plot",
    "hier-intro-plot",
    "hier-merge-slider",
    "hier-merge-value",
    "hier-merge-note",
    "hier-merge-scatter",
    "hier-merge-dendrogram",
    "hier-merge-clusters",
    "hier-merge-height",
    "hier-cut-slider",
    "hier-cut-value",
    "hier-cut-note",
    "hier-cut-dendrogram",
    "hier-cut-scatter",
    "hier-cut-clusters",
    "hier-cut-height-label",
    "hier-cut-story",
    "hier-linkage-controls",
    "hier-linkage-note",
    "hier-linkage-scatter",
    "hier-linkage-dendrogram",
    "hier-linkage-active",
    "hier-linkage-height",
    "hier-linkage-warning",
    "hier-distance-controls",
    "hier-distance-note",
    "hier-distance-dendrogram",
    "hier-distance-profiles",
    "hier-distance-active",
    "hier-distance-story",
    "hier-scale-controls",
    "hier-scale-note",
    "hier-scale-scatter",
    "hier-scale-dendrogram",
    "hier-scale-basis",
    "hier-scale-story",
    "hier-outlier-controls",
    "hier-outlier-note",
    "hier-outlier-scatter",
    "hier-outlier-dendrogram",
    "hier-outlier-height",
    "hier-outlier-story",
    "hier-outlier-singleton",
    "hier-final-slider",
    "hier-final-value",
    "hier-final-dendrogram",
    "hier-final-cluster-cards",
  ].forEach((id) => {
    refs[id] = document.getElementById(id);
  });
}

function toggleActiveButton(container, attribute, value) {
  container.querySelectorAll(`[data-${attribute}]`).forEach((button) => {
    button.classList.toggle("is-active", button.dataset[attribute] === value);
  });
}

function renderKmeansUnsupervised() {
  const previewAssignments = cache.bestKRuns[3].assignments;
  const markup = scatterSvg({
    points: data.kmeansMain,
    assignments: state.kmeansReveal ? previewAssignments : null,
    centroids: state.kmeansReveal ? cache.bestKRuns[3].centroids : [],
    note: state.kmeansReveal ? "distance creates the labels" : "no labels are provided",
    legend: state.kmeansReveal ? "clusters are imposed from distance" : "raw observations only",
  });
  setSvg(refs["kmeans-unsupervised-plot"], markup);
  refs["kmeans-unsupervised-toggle"].textContent = state.kmeansReveal ? "Hide structure" : "Reveal structure";
}

function getAlgorithmTrace() {
  return kmeansTrace(data.kmeansMain, state.kmeansAlgorithm.k, 14 + state.kmeansAlgorithm.k * 3);
}

function renderAlgorithm() {
  const trace = getAlgorithmTrace();
  state.kmeansAlgorithm.step = Math.min(state.kmeansAlgorithm.step, trace.frames.length - 1);
  const frame = trace.frames[state.kmeansAlgorithm.step];

  refs["kmeans-algorithm-k-value"].textContent = String(state.kmeansAlgorithm.k);
  refs["kmeans-algorithm-title"].textContent = frame.title;
  refs["kmeans-algorithm-subtitle"].textContent = `step ${state.kmeansAlgorithm.step + 1} of ${trace.frames.length}`;
  refs["kmeans-algorithm-moved"].textContent = String(frame.moved);
  refs["kmeans-algorithm-inertia"].textContent = formatNumber(frame.inertia, 2);
  refs["kmeans-algorithm-loop"].textContent = String(frame.loop);

  const steps = refs["kmeans-step-list"].querySelectorAll("li");
  steps.forEach((item, index) => {
    item.classList.toggle("is-active", index === frame.step);
  });

  setSvg(
    refs["kmeans-algorithm-plot"],
    scatterSvg({
      points: data.kmeansMain,
      assignments: frame.assignments,
      centroids: frame.centroids,
      note:
        frame.title === "Choose K"
          ? "pick a target number of groups"
          : frame.title === "Random start"
            ? "a random configuration begins the search"
            : frame.title === "Compute centroids"
              ? "means become provisional centers"
              : "points switch to their nearest center",
    }),
  );
}

function renderChooseK() {
  const diagnostic = cache.kDiagnostics[state.kmeansSelectedK - 1];
  const labels = cache.kDiagnostics.map((entry) => String(entry.k));
  const elbowValues = cache.kDiagnostics.map((entry) => entry.inertia);
  const silhouetteValues = cache.kDiagnostics.slice(1).map((entry) => entry.silhouette);

  refs["kmeans-k-slider-value"].textContent = String(state.kmeansSelectedK);
  refs["kmeans-k-inertia"].textContent = formatNumber(diagnostic.inertia, 2);
  refs["kmeans-k-silhouette"].textContent = formatNumber(diagnostic.silhouette, 2);
  refs["kmeans-k-balance"].textContent = formatPercent(
    Math.max(...diagnostic.summaries.map((summary) => summary.share)),
    0,
  );

  setSvg(
    refs["kmeans-k-plot"],
    scatterSvg({
      points: data.kmeansMain,
      assignments: diagnostic.run.assignments,
      centroids: diagnostic.run.centroids,
      note: `K = ${state.kmeansSelectedK}`,
    }),
  );

  setSvg(
    refs["kmeans-elbow-plot"],
    lineChartSvg(elbowValues, {
      labels,
      selectedIndex: state.kmeansSelectedK - 1,
      title: "Elbow",
      color: getCssVar("--accent"),
    }),
  );

  setSvg(
    refs["kmeans-silhouette-plot"],
    lineChartSvg(silhouetteValues, {
      labels: labels.slice(1),
      selectedIndex: Math.max(state.kmeansSelectedK - 2, 0),
      title: "Silhouette",
      color: getCssVar("--accent-alt"),
      minY: Math.min(...silhouetteValues, 0),
      maxY: Math.max(...silhouetteValues, 0.65),
    }),
  );
}

function renderRandomStarts() {
  const current = cache.randomCases[state.kmeansRandomIndex];
  const best = cache.randomCases[0];
  const improvement = 1 - best.inertia / current.inertia;

  refs["kmeans-random-summary"].textContent = `seed ${current.seed} versus the best of ${cache.randomCases.length} starts`;
  refs["kmeans-random-current-inertia"].textContent = formatNumber(current.inertia, 2);
  refs["kmeans-random-best-inertia"].textContent = formatNumber(best.inertia, 2);
  refs["kmeans-random-gap"].textContent = formatPercent(improvement, 0);

  setSvg(
    refs["kmeans-random-current"],
    scatterSvg({
      width: 290,
      height: 280,
      padding: 30,
      points: data.kmeansMain,
      assignments: current.assignments,
      centroids: current.centroids,
      initialCentroids: current.initialCentroids,
      note: `seed ${current.seed}`,
    }),
  );

  setSvg(
    refs["kmeans-random-best"],
    scatterSvg({
      width: 290,
      height: 280,
      padding: 30,
      points: data.kmeansMain,
      assignments: best.assignments,
      centroids: best.centroids,
      initialCentroids: best.initialCentroids,
      note: "lowest inertia",
    }),
  );
}

function renderScaleDemo() {
  const rawRun = kmeansRun(data.scaleData, 2, 1);
  const scaledRun = kmeansRun(cache.scaleInfo.scaled, 2, 1);
  const assignments = state.kmeansScaleMode === "raw" ? rawRun.assignments : scaledRun.assignments;
  const centroids = clusterSummaries(data.scaleData, assignments).map((summary) => summary.centroid);

  refs["kmeans-scale-note"].textContent =
    state.kmeansScaleMode === "raw" ? "raw units emphasize socks" : "scaled features restore balance";
  refs["kmeans-scale-basis"].textContent = state.kmeansScaleMode;
  refs["kmeans-scale-story"].textContent =
    state.kmeansScaleMode === "raw" ? "socks dominate" : "computer intensity appears";

  setSvg(
    refs["kmeans-scale-plot"],
    scatterSvg({
      points: data.scaleData,
      assignments,
      centroids,
      axes: { xLabel: "Socks", yLabel: "Computers" },
      note: state.kmeansScaleMode === "raw" ? "distance driven by bigger numbers" : "distance driven by z-scores",
    }),
  );
}

function renderOutlierDemo() {
  const baseRun = kmeansRun(data.outlierBase, 2, 1);
  const withOutlierRun = kmeansRun(data.outlierBase.concat([data.outlierPoint]), 2, 1);
  const showOutlier = state.kmeansOutlierMode === "show";
  const points = showOutlier ? data.outlierBase.concat([data.outlierPoint]) : data.outlierBase;
  const run = showOutlier ? withOutlierRun : baseRun;
  const shift = euclideanDistance(baseRun.centroids[1], withOutlierRun.centroids[1]);

  refs["kmeans-outlier-note"].textContent = showOutlier
    ? "the right centroid is pulled toward the outlier"
    : "clean fit without the stray point";
  refs["kmeans-outlier-shift"].textContent = formatNumber(showOutlier ? shift : 0, 2);
  refs["kmeans-outlier-inertia"].textContent = showOutlier
    ? formatPercent((withOutlierRun.inertia - baseRun.inertia) / baseRun.inertia, 0)
    : "0%";
  refs["kmeans-outlier-story"].textContent = showOutlier ? "distorted" : "stable";

  setSvg(
    refs["kmeans-outlier-plot"],
    scatterSvg({
      points,
      assignments: run.assignments,
      centroids: run.centroids,
      ghostCentroids: showOutlier ? baseRun.centroids : [],
      outlierIndex: showOutlier ? points.length - 1 : null,
      outlierPoint: showOutlier ? data.outlierPoint : null,
      note: showOutlier ? "dashed marks show the clean centroids" : "baseline centroids",
    }),
  );
}

function renderKmeansInterpretation() {
  const diagnostic = cache.kDiagnostics[state.kmeansSelectedK - 1];
  const summaries = diagnostic.summaries;
  state.kmeansFocusCluster = Math.min(state.kmeansFocusCluster, summaries.length - 1);

  setSvg(
    refs["kmeans-final-plot"],
    scatterSvg({
      width: 620,
      height: 340,
      padding: 42,
      points: data.kmeansMain,
      assignments: diagnostic.run.assignments,
      centroids: diagnostic.run.centroids,
      highlightCluster: state.kmeansFocusCluster,
      note: `cluster ${state.kmeansFocusCluster + 1} highlighted`,
    }),
  );

  refs["kmeans-cluster-cards"].innerHTML = summaries
    .map((summary) => {
      const color = PALETTE[summary.index % PALETTE.length];
      return `
        <button
          type="button"
          class="cluster-card ${summary.index === state.kmeansFocusCluster ? "is-active" : ""}"
          data-kmeans-cluster="${summary.index}"
        >
          <div class="cluster-title">
            <span class="cluster-swatch" style="background:${color}"></span>
            Cluster ${summary.index + 1}
          </div>
          <div class="cluster-copy">
            <span>${summary.size} points</span>
            <span>center (${formatNumber(summary.centroid[0], 2)}, ${formatNumber(summary.centroid[1], 2)})</span>
            <span>spread ${formatNumber(summary.spread, 2)}</span>
          </div>
        </button>
      `;
    })
    .join("");
}

function renderHierIntro() {
  const assignments = groupsToAssignments(cutTree(cache.hierAverage.root, cache.hierAverage.maxHeight * 0.58), data.hierMain.length);
  setSvg(
    refs["hier-intro-plot"],
    dendrogramSvg(cache.hierAverage, {
      width: 620,
      height: 440,
      cutHeight: cache.hierAverage.maxHeight * 0.58,
      assignments,
      labels: data.hierLabels,
    }),
  );
}

function renderHierMerge() {
  const maxStep = cache.hierAverage.merges.length;
  refs["hier-merge-slider"].max = String(maxStep);
  state.hierMergeStep = Math.min(state.hierMergeStep, maxStep);
  const groups = cache.hierAverage.clusterStates[state.hierMergeStep];
  const assignments = groupsToAssignments(groups, data.hierMain.length);
  const height = cache.hierAverage.merges[state.hierMergeStep - 1].height;

  refs["hier-merge-value"].textContent = String(state.hierMergeStep);
  refs["hier-merge-note"].textContent = `fusion ${state.hierMergeStep} of ${maxStep}`;
  refs["hier-merge-clusters"].textContent = String(groups.length);
  refs["hier-merge-height"].textContent = formatNumber(height, 2);

  setSvg(
    refs["hier-merge-scatter"],
    scatterSvg({
      width: 290,
      height: 280,
      padding: 30,
      points: data.hierMain,
      assignments,
      note: `${groups.length} clusters remain`,
    }),
  );

  setSvg(
    refs["hier-merge-dendrogram"],
    dendrogramSvg(cache.hierAverage, {
      width: 290,
      height: 280,
      visibleStep: state.hierMergeStep,
      currentStep: state.hierMergeStep,
      assignments,
      labels: data.hierLabels,
    }),
  );
}

function cutHeightFromPercent(clustering, percent) {
  return clustering.maxHeight * (percent / 100);
}

function renderHierCut() {
  const cutHeight = cutHeightFromPercent(cache.hierAverage, state.hierCutPercent);
  const groups = cutTree(cache.hierAverage.root, cutHeight);
  const assignments = groupsToAssignments(groups, data.hierMain.length);

  refs["hier-cut-value"].textContent = `${state.hierCutPercent}%`;
  refs["hier-cut-note"].textContent = groups.length <= 3 ? "broad structure" : "fine-grained split";
  refs["hier-cut-clusters"].textContent = String(groups.length);
  refs["hier-cut-height-label"].textContent = formatNumber(cutHeight, 2);
  refs["hier-cut-story"].textContent = groups.length <= 3 ? "broad split" : "finer segmentation";

  setSvg(
    refs["hier-cut-dendrogram"],
    dendrogramSvg(cache.hierAverage, {
      width: 290,
      height: 280,
      cutHeight,
      assignments,
      labels: data.hierLabels,
    }),
  );

  setSvg(
    refs["hier-cut-scatter"],
    scatterSvg({
      width: 290,
      height: 280,
      padding: 30,
      points: data.hierMain,
      assignments,
      note: `${groups.length} clusters at this cut`,
    }),
  );
}

function renderHierLinkage() {
  const clustering = cache.linkageResults[state.hierLinkage];
  const cutHeight = clustering.maxHeight * 0.48;
  const assignments = groupsToAssignments(cutTree(clustering.root, cutHeight), data.linkageData.length);
  const noteMap = {
    complete: "complete keeps groups compact",
    single: "single linkage can chain through bridges",
    average: "average balances compactness and continuity",
    centroid: "centroid compares cluster means directly",
  };

  refs["hier-linkage-note"].textContent = noteMap[state.hierLinkage];
  refs["hier-linkage-active"].textContent = state.hierLinkage;
  refs["hier-linkage-height"].textContent = formatNumber(clustering.maxHeight, 2);
  refs["hier-linkage-warning"].textContent = clustering.inversion ? "inversion risk" : "none";

  setSvg(
    refs["hier-linkage-scatter"],
    scatterSvg({
      width: 290,
      height: 280,
      padding: 30,
      points: data.linkageData,
      assignments,
      note: state.hierLinkage,
    }),
  );

  setSvg(
    refs["hier-linkage-dendrogram"],
    dendrogramSvg(clustering, {
      width: 290,
      height: 280,
      cutHeight,
      assignments,
    }),
  );
}

function renderHierDistance() {
  const clustering = cache.distanceResults[state.hierDistance];
  const cutHeight = clustering.maxHeight * 0.5;
  const assignments = groupsToAssignments(cutTree(clustering.root, cutHeight), data.profileData.length);

  refs["hier-distance-note"].textContent =
    state.hierDistance === "euclidean"
      ? "euclidean groups similar amounts"
      : "correlation groups similar shapes";
  refs["hier-distance-active"].textContent = state.hierDistance;
  refs["hier-distance-story"].textContent =
    state.hierDistance === "euclidean" ? "magnitude" : "shape";

  setSvg(
    refs["hier-distance-dendrogram"],
    dendrogramSvg(clustering, {
      width: 620,
      height: 180,
      cutHeight,
      assignments,
      labels: data.profileLabels,
    }),
  );
  renderProfileCards(refs["hier-distance-profiles"], data.profileData, assignments, data.profileLabels);
}

function renderHierScale() {
  const clustering = state.hierScaleMode === "raw" ? cache.scaleRawHier : cache.scaleScaledHier;
  const cutHeight = clustering.maxHeight * 0.43;
  const assignments = groupsToAssignments(cutTree(clustering.root, cutHeight), data.scaleData.length);

  refs["hier-scale-note"].textContent =
    state.hierScaleMode === "raw" ? "raw units emphasize socks" : "scaled values recover computer intensity";
  refs["hier-scale-basis"].textContent = state.hierScaleMode;
  refs["hier-scale-story"].textContent =
    state.hierScaleMode === "raw" ? "socks dominate" : "computer intensity appears";

  setSvg(
    refs["hier-scale-scatter"],
    scatterSvg({
      width: 290,
      height: 280,
      padding: 30,
      points: data.scaleData,
      assignments,
      axes: { xLabel: "Socks", yLabel: "Computers" },
    }),
  );
  setSvg(
    refs["hier-scale-dendrogram"],
    dendrogramSvg(clustering, {
      width: 290,
      height: 280,
      cutHeight,
      assignments,
    }),
  );
}

function renderHierOutlier() {
  const showOutlier = state.hierOutlierMode === "show";
  const points = showOutlier ? data.outlierBase.concat([data.outlierPoint]) : data.outlierBase;
  const clustering = showOutlier ? cache.outlierDirtyHier : cache.outlierCleanHier;
  const cutHeight = clustering.maxHeight * 0.42;
  const assignments = groupsToAssignments(cutTree(clustering.root, cutHeight), points.length);
  const singletonRisk = showOutlier ? "high" : "low";

  refs["hier-outlier-note"].textContent = showOutlier
    ? "the outlier waits until the end"
    : "no isolated branch";
  refs["hier-outlier-height"].textContent = formatNumber(clustering.maxHeight, 2);
  refs["hier-outlier-story"].textContent = showOutlier ? "late singleton" : "stable";
  refs["hier-outlier-singleton"].textContent = singletonRisk;

  setSvg(
    refs["hier-outlier-scatter"],
    scatterSvg({
      width: 290,
      height: 280,
      padding: 30,
      points,
      assignments,
      outlierIndex: showOutlier ? points.length - 1 : null,
      outlierPoint: showOutlier ? data.outlierPoint : null,
    }),
  );
  setSvg(
    refs["hier-outlier-dendrogram"],
    dendrogramSvg(clustering, {
      width: 290,
      height: 280,
      cutHeight,
      assignments,
    }),
  );
}

function renderHierFinal() {
  const cutHeight = cutHeightFromPercent(cache.hierAverage, state.hierFinalPercent);
  const groups = cutTree(cache.hierAverage.root, cutHeight);
  const assignments = groupsToAssignments(groups, data.hierMain.length);
  state.hierFocusCluster = Math.min(state.hierFocusCluster, groups.length - 1);

  refs["hier-final-value"].textContent = `${state.hierFinalPercent}%`;

  setSvg(
    refs["hier-final-dendrogram"],
    dendrogramSvg(cache.hierAverage, {
      width: 620,
      height: 300,
      cutHeight,
      assignments,
      labels: data.hierLabels,
      focusCluster: state.hierFocusCluster,
    }),
  );

  refs["hier-final-cluster-cards"].innerHTML = groups
    .slice()
    .sort((left, right) => Math.min(...left) - Math.min(...right))
    .map((group, index) => {
      const points = group.map((member) => data.hierMain[member]);
      const center = meanPoint(points);
      const color = PALETTE[index % PALETTE.length];
      return `
        <button
          type="button"
          class="cluster-card ${index === state.hierFocusCluster ? "is-active" : ""}"
          data-hier-cluster="${index}"
        >
          <div class="cluster-title">
            <span class="cluster-swatch" style="background:${color}"></span>
            Cluster ${index + 1}
          </div>
          <div class="cluster-copy">
            <span>${group.length} leaves</span>
            <span>members ${group.map((member) => member + 1).join(", ")}</span>
            <span>center (${formatNumber(center[0], 2)}, ${formatNumber(center[1], 2)})</span>
          </div>
        </button>
      `;
    })
    .join("");
}

function renderAllStatic() {
  renderKmeansUnsupervised();
  renderAlgorithm();
  renderChooseK();
  renderRandomStarts();
  renderScaleDemo();
  renderOutlierDemo();
  renderKmeansInterpretation();
  renderHierIntro();
  renderHierMerge();
  renderHierCut();
  renderHierLinkage();
  renderHierDistance();
  renderHierScale();
  renderHierOutlier();
  renderHierFinal();
}

function animateHeroPlots(time) {
  const frameDuration = 1200;
  const total = heroTrace.frames.length * frameDuration;
  const position = (time % total) / frameDuration;
  const frameIndex = Math.floor(position);
  const frame = heroTrace.frames[frameIndex];

  const jittered = data.kmeansMain.map((point, index) => [
    point[0] + Math.sin(time / 900 + index * 0.41) * 0.05,
    point[1] + Math.cos(time / 1050 + index * 0.31) * 0.05,
  ]);

  setSvg(
    refs["kmeans-hero-plot"],
    scatterSvg({
      width: 620,
      height: 520,
      padding: 56,
      points: jittered,
      assignments: frame.assignments,
      centroids: frame.centroids,
      note: frame.title,
      legend: "centroids keep pulling the partition tighter",
    }),
  );

  const heroProgress = ((time / 900) % (cache.hierAverage.merges.length + 2));
  const visibleStep = Math.max(1, Math.floor(heroProgress));
  const cutHeight = cache.hierAverage.maxHeight * 0.58;
  const assignments = groupsToAssignments(cutTree(cache.hierAverage.root, cutHeight), data.hierMain.length);

  setSvg(
    refs["hier-hero-plot"],
    dendrogramSvg(cache.hierAverage, {
      width: 620,
      height: 520,
      visibleStep,
      currentStep: visibleStep,
      cutHeight,
      assignments,
      labels: data.hierLabels,
    }),
  );

  requestAnimationFrame(animateHeroPlots);
}

function stopAlgorithmPlayback() {
  if (state.kmeansAlgorithm.timer) {
    window.clearInterval(state.kmeansAlgorithm.timer);
    state.kmeansAlgorithm.timer = null;
  }
  state.kmeansAlgorithm.playing = false;
  refs["kmeans-step-play"].textContent = "Play";
}

function startAlgorithmPlayback() {
  stopAlgorithmPlayback();
  const trace = getAlgorithmTrace();
  state.kmeansAlgorithm.playing = true;
  refs["kmeans-step-play"].textContent = "Pause";
  state.kmeansAlgorithm.timer = window.setInterval(() => {
    if (state.kmeansAlgorithm.step >= trace.frames.length - 1) {
      stopAlgorithmPlayback();
      return;
    }
    state.kmeansAlgorithm.step += 1;
    renderAlgorithm();
  }, 1100);
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.target;
      if (target === state.activeTab) return;
      state.activeTab = target;
      document.body.dataset.theme = target;
      document.querySelectorAll(".tab").forEach((tab) => {
        const active = tab.dataset.target === target;
        tab.classList.toggle("is-active", active);
        tab.setAttribute("aria-pressed", String(active));
      });
      document.querySelectorAll(".method").forEach((panel) => {
        const active = panel.id === target;
        panel.hidden = !active;
      });
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });

  refs["kmeans-unsupervised-toggle"].addEventListener("click", () => {
    state.kmeansReveal = !state.kmeansReveal;
    renderKmeansUnsupervised();
  });

  refs["kmeans-algorithm-k"].addEventListener("input", (event) => {
    state.kmeansAlgorithm.k = Number(event.target.value);
    state.kmeansAlgorithm.step = 0;
    stopAlgorithmPlayback();
    renderAlgorithm();
  });

  refs["kmeans-step-prev"].addEventListener("click", () => {
    stopAlgorithmPlayback();
    state.kmeansAlgorithm.step = Math.max(0, state.kmeansAlgorithm.step - 1);
    renderAlgorithm();
  });

  refs["kmeans-step-next"].addEventListener("click", () => {
    stopAlgorithmPlayback();
    const trace = getAlgorithmTrace();
    state.kmeansAlgorithm.step = Math.min(trace.frames.length - 1, state.kmeansAlgorithm.step + 1);
    renderAlgorithm();
  });

  refs["kmeans-step-play"].addEventListener("click", () => {
    if (state.kmeansAlgorithm.playing) {
      stopAlgorithmPlayback();
    } else {
      startAlgorithmPlayback();
    }
  });

  refs["kmeans-k-slider"].addEventListener("input", (event) => {
    state.kmeansSelectedK = Number(event.target.value);
    state.kmeansFocusCluster = 0;
    renderChooseK();
    renderKmeansInterpretation();
  });

  refs["kmeans-random-reroll"].addEventListener("click", () => {
    state.kmeansRandomIndex = (state.kmeansRandomIndex + 3) % cache.randomCases.length;
    if (state.kmeansRandomIndex === 0) state.kmeansRandomIndex = cache.randomCases.length - 1;
    renderRandomStarts();
  });

  refs["kmeans-scale-controls"].addEventListener("click", (event) => {
    const button = event.target.closest("[data-scale]");
    if (!button) return;
    state.kmeansScaleMode = button.dataset.scale;
    toggleActiveButton(refs["kmeans-scale-controls"], "scale", state.kmeansScaleMode);
    renderScaleDemo();
  });

  refs["kmeans-outlier-controls"].addEventListener("click", (event) => {
    const button = event.target.closest("[data-outlier]");
    if (!button) return;
    state.kmeansOutlierMode = button.dataset.outlier;
    toggleActiveButton(refs["kmeans-outlier-controls"], "outlier", state.kmeansOutlierMode);
    renderOutlierDemo();
  });

  refs["kmeans-cluster-cards"].addEventListener("click", (event) => {
    const button = event.target.closest("[data-kmeans-cluster]");
    if (!button) return;
    state.kmeansFocusCluster = Number(button.dataset.kmeansCluster);
    renderKmeansInterpretation();
  });

  refs["hier-merge-slider"].addEventListener("input", (event) => {
    state.hierMergeStep = Number(event.target.value);
    renderHierMerge();
  });

  refs["hier-cut-slider"].addEventListener("input", (event) => {
    state.hierCutPercent = Number(event.target.value);
    renderHierCut();
  });

  refs["hier-linkage-controls"].addEventListener("click", (event) => {
    const button = event.target.closest("[data-linkage]");
    if (!button) return;
    state.hierLinkage = button.dataset.linkage;
    toggleActiveButton(refs["hier-linkage-controls"], "linkage", state.hierLinkage);
    renderHierLinkage();
  });

  refs["hier-distance-controls"].addEventListener("click", (event) => {
    const button = event.target.closest("[data-distance]");
    if (!button) return;
    state.hierDistance = button.dataset.distance;
    toggleActiveButton(refs["hier-distance-controls"], "distance", state.hierDistance);
    renderHierDistance();
  });

  refs["hier-scale-controls"].addEventListener("click", (event) => {
    const button = event.target.closest("[data-scale]");
    if (!button) return;
    state.hierScaleMode = button.dataset.scale;
    toggleActiveButton(refs["hier-scale-controls"], "scale", state.hierScaleMode);
    renderHierScale();
  });

  refs["hier-outlier-controls"].addEventListener("click", (event) => {
    const button = event.target.closest("[data-outlier]");
    if (!button) return;
    state.hierOutlierMode = button.dataset.outlier;
    toggleActiveButton(refs["hier-outlier-controls"], "outlier", state.hierOutlierMode);
    renderHierOutlier();
  });

  refs["hier-final-slider"].addEventListener("input", (event) => {
    state.hierFinalPercent = Number(event.target.value);
    state.hierFocusCluster = 0;
    renderHierFinal();
  });

  refs["hier-final-cluster-cards"].addEventListener("click", (event) => {
    const button = event.target.closest("[data-hier-cluster]");
    if (!button) return;
    state.hierFocusCluster = Number(button.dataset.hierCluster);
    renderHierFinal();
  });
}

function bindRevealObserver() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.classList.add("is-visible");
      });
    },
    { threshold: 0.16 },
  );

  document.querySelectorAll(".reveal").forEach((section) => observer.observe(section));
}

function initialize() {
  bindElements();
  bindEvents();
  bindRevealObserver();
  refs["kmeans-algorithm-k"].value = String(state.kmeansAlgorithm.k);
  refs["kmeans-k-slider"].value = String(state.kmeansSelectedK);
  refs["hier-cut-slider"].value = String(state.hierCutPercent);
  refs["hier-final-slider"].value = String(state.hierFinalPercent);
  refs["hier-merge-slider"].value = String(state.hierMergeStep);
  renderAllStatic();
  requestAnimationFrame(animateHeroPlots);
}

initialize();
