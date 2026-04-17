// spike/corpus.js — programmatic clustered corpus shared by demo.js and
// demo-reference.js. 5 topical clusters × 200 template-generated sentences.
//
// Purpose: reproducible, network-free, verifiable-by-clustering workload.
// MS-MARCO style text would be more realistic but requires HF fetch; for
// the Day 4 end-to-end demo this is good enough and the comparison
// (GPU vs CPU) is apples-to-apples because both demos generate from the
// same seed + template set.

const CLUSTERS = {
  cooking: {
    templates: [
      "I love to {} pasta {}.",
      "Fresh {} makes any dish {}.",
      "{} bread at home is deeply satisfying.",
      "Roasting {} brings out their sweetness.",
      "A good {} is the most important kitchen tool.",
      "I always {} the pasta water generously.",
      "Homemade {} beats restaurant {} by a mile.",
      "Slow cooking transforms {} cuts of meat.",
      "Mise en place separates good cooks from {} ones.",
      "Kitchen scraps make {} stock from scratch.",
    ],
    fillers: [
      ["cook", "make", "prepare", "braise", "simmer", "bake"],
      ["basil", "thyme", "rosemary", "parsley", "cilantro", "oregano"],
      ["Sunday", "in the morning", "with care", "for friends", "on weekends"],
      ["come alive", "sing", "shine", "pop", "sparkle"],
      ["sourdough", "pizza dough", "rye", "focaccia", "baguette"],
      ["vegetables", "roots", "squash", "peppers", "onions"],
      ["knife", "pan", "pot", "cutting board", "spoon"],
      ["salt", "season", "taste"],
      ["tough", "cheap", "connective-heavy", "underappreciated"],
      ["great", "exceptional", "truly great", "world-class"],
      ["pizza", "pasta", "bread", "soup"],
    ],
  },
  programming: {
    templates: [
      "I spent the afternoon {} a race condition.",
      "Static types catch bugs that tests would {}.",
      "{} gave me a new appreciation for {}.",
      "Writing code is easy; {} code is hard.",
      "A good {} pays for itself over years.",
      "I prefer {} errors to silent failures.",
      "Refactoring is how legacy code becomes {} code.",
      "Pair programming is undervalued by most {}.",
      "Version control saved my career more than {}.",
      "The hardest part of software is {} things.",
    ],
    fillers: [
      ["debugging", "tracing", "investigating", "chasing down"],
      ["miss", "overlook", "fail to detect", "glide past"],
      ["Rust", "Go", "TypeScript", "Elixir", "OCaml", "Haskell"],
      ["ownership semantics", "borrow checking", "sum types", "immutability"],
      ["reading", "reviewing", "inheriting", "debugging"],
      ["abstraction", "interface", "API", "type signature"],
      ["explicit", "typed", "named", "checked"],
      ["new", "maintainable", "living", "working"],
      ["teams", "developers", "shops", "engineers"],
      ["once", "twice", "a dozen times"],
      ["naming", "deleting", "refactoring", "documenting"],
    ],
  },
  travel: {
    templates: [
      "{} in {} is unforgettable.",
      "I took a train across {} last summer.",
      "The best meals I had were from {} vendors.",
      "Small towns in {} charmed me more than the capital.",
      "I travel {} because airports are a test of {}.",
      "Getting lost in a new city is part of the {}.",
      "The night train to {} is a small pleasure.",
      "{} in winter feels like another planet.",
      "I try to learn a few phrases before I land {}.",
      "Traveling alone taught me how to be comfortable {}.",
    ],
    fillers: [
      ["Japan", "Italy", "Portugal", "Peru", "Morocco", "Vietnam"],
      ["autumn", "spring", "the shoulder season", "late October"],
      ["Europe", "Asia", "South America", "the Balkans", "Iberia"],
      ["street", "market", "roadside", "station", "corner"],
      ["Portugal", "Spain", "France", "Japan", "Italy"],
      ["light", "carry-on only", "minimally", "with one bag"],
      ["patience", "resolve", "sanity", "stamina"],
      ["experience", "adventure", "point", "charm"],
      ["Venice", "Prague", "Vienna", "Budapest", "Barcelona"],
      ["Iceland", "Norway", "Finland", "Patagonia"],
      ["anywhere", "somewhere new", "in a new country"],
      ["quiet", "alone", "in silence", "with my thoughts"],
    ],
  },
  music: {
    templates: [
      "Vinyl records have a warmth {} cannot match.",
      "Live jazz in a small {} is the best kind.",
      "I learned {} slowly over many years.",
      "Some albums take decades to reveal {}.",
      "A well-made playlist is a small {}.",
      "Listening to music while {} changes the city.",
      "I collect records the way others collect {}.",
      "{} music rewards patience like few things do.",
      "A great {} can anchor any song.",
      "Seeing a band live changes how you hear the {}.",
    ],
    fillers: [
      ["digital", "streaming", "CDs", "MP3s"],
      ["club", "venue", "bar", "cafe"],
      ["guitar", "piano", "bass", "drums", "violin"],
      ["themselves", "their depth", "their meaning", "their beauty"],
      ["gift", "pleasure", "joy", "treasure"],
      ["walking", "running", "commuting", "biking"],
      ["books", "stamps", "art", "posters"],
      ["Classical", "Jazz", "Folk", "Ambient"],
      ["bass line", "drum beat", "melody", "guitar riff"],
      ["record", "album", "studio version", "original"],
    ],
  },
  gardening: {
    templates: [
      "{} in a sunny spot are easy to grow.",
      "I composted all my {} scraps last year.",
      "Weeding is {} once you stop fighting it.",
      "A {} bed makes a small yard more productive.",
      "{} need patience more than water.",
      "I planted {} in October and forgot about it.",
      "{} in the garden mean you're doing something right.",
      "Native plants require less water than {}.",
      "Compost turns kitchen waste into {}.",
      "I prefer {} herbs over annual flowers.",
    ],
    fillers: [
      ["Tomatoes", "Peppers", "Herbs", "Cucumbers", "Zucchini"],
      ["kitchen", "vegetable", "coffee-ground", "food"],
      ["meditative", "peaceful", "calming", "satisfying"],
      ["raised", "deep", "lasagna", "no-dig"],
      ["Seedlings", "Young plants", "New cuttings", "Transplants"],
      ["garlic", "bulbs", "tulips", "daffodils", "shallots"],
      ["Bees", "Butterflies", "Pollinators", "Hoverflies"],
      ["lawns", "turf grass", "manicured grass"],
      ["black gold", "rich soil", "dark humus", "gardener's gold"],
      ["perennial", "woody", "hardy", "drought-tolerant"],
    ],
  },
};

function fillTemplate(tmpl, fillers) {
  let out = tmpl;
  let i = 0;
  while (out.includes('{}')) {
    const col = fillers[i++ % fillers.length];
    out = out.replace('{}', col[Math.floor(Math.random() * col.length)]);
  }
  return out;
}

function buildCorpus(perCluster = 200) {
  const docs = [];
  const clusters = [];
  let clusterId = 0;
  for (const [, { templates, fillers }] of Object.entries(CLUSTERS)) {
    for (let i = 0; i < perCluster; i++) {
      const t = templates[i % templates.length];
      docs.push(fillTemplate(t, fillers));
      clusters.push(clusterId);
    }
    clusterId++;
  }
  return { docs, clusters, clusterNames: Object.keys(CLUSTERS) };
}

function buildQueries(perCluster = 2) {
  const queries = [];
  const queryClusterTruth = [];
  const clusterNames = Object.keys(CLUSTERS);
  for (let c = 0; c < clusterNames.length; c++) {
    const { templates, fillers } = CLUSTERS[clusterNames[c]];
    for (let q = 0; q < perCluster; q++) {
      queries.push(fillTemplate(templates[(q * 3) % templates.length], fillers));
      queryClusterTruth.push(c);
    }
  }
  return { queries, queryClusterTruth };
}

module.exports = { CLUSTERS, fillTemplate, buildCorpus, buildQueries };
