// spike/reference.js — generate ground-truth embeddings for correctness gate F4.
//
// Uses @huggingface/transformers (v4+) on CPU to embed a 100-sentence sanity
// set. Runs the same pipeline (tokenize → model → mean-pool → L2-normalize)
// that MAX is expected to match in Mojo. Output goes to fixtures/ground-truth.bin;
// Day 8's correctness test loads it and compares cosine similarity.
//
// Run once on M4: node spike/reference.js
// Re-run only if the sanity set or the model revision changes.

const fs = require('fs');
const path = require('path');

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const EMBED_DIM = 384;

// 100-sentence sanity set, 5 topical clusters of 20 sentences each. Written
// flat here for diffability; expand to a .txt fixture if we grow this.
const SANITY_SET = [
  // Cluster A: cooking
  'I love to cook pasta on Sunday evenings.',
  'Fresh basil makes tomato sauce come alive.',
  'My grandmother taught me how to make bread.',
  'Roasting vegetables brings out their sweetness.',
  'A good knife is the most important kitchen tool.',
  'I always salt the pasta water generously.',
  'Homemade pizza is better than any restaurant.',
  'Slow cooking transforms tough cuts of meat.',
  'Fresh herbs lift any dish out of mediocrity.',
  'I batch cook on Sundays to save weeknight time.',
  'Reading recipes is like reading poetry sometimes.',
  'A sharp knife is a safe knife.',
  'Good olive oil makes every dish better.',
  'I grew up eating homemade sourdough.',
  'Kitchen gadgets mostly collect dust unless used daily.',
  'Stock from scratch changes how your soup tastes.',
  'Onions caramelize slowly; there is no shortcut.',
  'I prefer cast iron to nonstick for most things.',
  'A home-cooked meal is an act of love.',
  'Mise en place separates good cooks from great ones.',
  // Cluster B: programming
  'I spent the afternoon debugging a race condition.',
  'Static types catch bugs that tests would miss.',
  'Rust gave me a new appreciation for ownership semantics.',
  'Writing code is easy; reading code is hard.',
  'A good abstraction pays for itself over years.',
  'I prefer explicit errors to silent failures.',
  'Refactoring is how legacy code becomes new code.',
  'Pair programming is undervalued by most teams.',
  'Version control saved my career more than once.',
  'The hardest part of software is naming things.',
  'Tests are the living documentation of intent.',
  'I find debugging more rewarding than writing features.',
  'A small, focused commit is easier to review.',
  'Code review is how teams transmit taste.',
  'I care more about readability than cleverness.',
  'Premature optimization has cost me many weeks.',
  'Comments explain why, not what.',
  'I like statically typed languages for large projects.',
  'The compiler is a tool for thinking, not just checking.',
  'Good architecture makes the next change cheaper.',
  // Cluster C: travel
  'Japan in autumn is unforgettable.',
  'I took a train across Europe last summer.',
  'The best meals I had were from street vendors.',
  'Small towns in Portugal charmed me more than Lisbon.',
  'I travel light because airports are a test of patience.',
  'Getting lost in a new city is part of the experience.',
  'The night train to Venice is a small pleasure.',
  'Iceland in winter feels like another planet.',
  'I try to learn a few phrases before I land anywhere.',
  'Traveling alone taught me how to be comfortable quiet.',
  'Museums give you context; streets give you truth.',
  'A walkable city is a city you can love.',
  'I prefer returning to a place over seeing a new one.',
  'Mountains change the scale of your problems.',
  'Every border crossing is a reminder of luck.',
  'Food markets tell you more than any guidebook.',
  'Traveling by train has a rhythm flights never match.',
  'I always pack fewer shoes than I think I need.',
  'A paper map still beats a phone in some places.',
  'Coming home is the best part of travel.',
  // Cluster D: music
  'Vinyl records have a warmth digital cannot match.',
  'Live jazz in a small club is the best kind.',
  'I learned guitar slowly over many years.',
  'Some albums take decades to reveal themselves.',
  'A well-made playlist is a small gift.',
  'Listening to music while walking changes the city.',
  'I collect records the way others collect books.',
  'Classical music rewards patience like few things do.',
  'A great bass line can anchor any song.',
  'I prefer headphones at night, speakers in the morning.',
  'Seeing a band live changes how you hear the record.',
  'Folk music carries stories across generations.',
  'Practicing scales is boring and necessary.',
  'A good producer is invisible in the best way.',
  'I find silence as important as sound in music.',
  'Record store clerks have the best recommendations.',
  'Some songs become associated with specific years.',
  'Learning an instrument teaches humility fast.',
  'Dancing to music is different than listening to it.',
  'A cover song can outshine the original.',
  // Cluster E: gardening
  'Tomatoes in a sunny spot are easy to grow.',
  'I composted all my kitchen scraps last year.',
  'Weeding is meditative once you stop fighting it.',
  'A raised bed makes a small yard more productive.',
  'Seedlings need patience more than water.',
  'I planted garlic in October and forgot about it.',
  'Bees in the garden mean you are doing something right.',
  'Native plants require less water than lawns.',
  'Compost turns kitchen waste into black gold.',
  'I prefer perennial herbs over annual flowers.',
  'Mulch keeps the soil moist in summer heat.',
  'A garden is a conversation with the weather.',
  'Pollinators visit more when flowers bloom together.',
  'Pruning feels destructive but makes growth possible.',
  'Seed-saving connects you to growers across years.',
  'A few good tools outlast many cheap ones.',
  'Rain barrels save water and money both.',
  'Shade gardens surprise you with their quiet beauty.',
  'I watch the first spring green with real attention.',
  'Gardens teach you that failure is just a season.',
];

async function main() {
  console.log(`loading ${MODEL_ID} on CPU…`);
  const { pipeline } = await import('@huggingface/transformers');
  const extractor = await pipeline('feature-extraction', MODEL_ID, {
    quantized: false,  // match the full-precision model we expect to run in MAX
  });

  console.log(`embedding ${SANITY_SET.length} sentences…`);
  const t0 = Date.now();
  const out = await extractor(SANITY_SET, { pooling: 'mean', normalize: true });
  const ms = Date.now() - t0;
  console.log(`  done (${ms}ms, ${(ms / SANITY_SET.length).toFixed(1)}ms/sentence)`);

  // out.data is a Float32Array of length N*EMBED_DIM (already flattened).
  const expected = SANITY_SET.length * EMBED_DIM;
  if (out.data.length !== expected) {
    throw new Error(`bad shape: got ${out.data.length}, expected ${expected}`);
  }

  // Write a tiny header for sanity: magic + n + dim + float32 payload.
  const header = Buffer.alloc(16);
  header.write('GRT1', 0, 'ascii');
  header.writeUInt32LE(0, 4);  // dtype 0 = float32
  header.writeUInt32LE(SANITY_SET.length, 8);
  header.writeUInt32LE(EMBED_DIM, 12);

  const fixturesDir = path.join(__dirname, 'fixtures');
  fs.mkdirSync(fixturesDir, { recursive: true });
  const outPath = path.join(fixturesDir, 'ground-truth.bin');
  const fd = fs.openSync(outPath, 'w');
  fs.writeSync(fd, header);
  fs.writeSync(fd, Buffer.from(out.data.buffer, out.data.byteOffset, out.data.byteLength));
  fs.closeSync(fd);

  // Also write the sanity set as text for traceability.
  fs.writeFileSync(
    path.join(__dirname, 'fixtures', 'sanity-set.txt'),
    SANITY_SET.join('\n') + '\n',
  );

  console.log(`wrote ${outPath} (${SANITY_SET.length} × ${EMBED_DIM} float32)`);
}

main().catch((e) => { console.error(e); process.exit(1); });
