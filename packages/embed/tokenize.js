// spike/tokenize.js — JS-side WordPiece tokenizer for MiniLM-L6-v2.
//
// Uses @huggingface/transformers (v4+, formerly @xenova/transformers — the
// package was migrated to the official HF namespace at v3.0 in Oct 2024).
// Produces Int32Array token IDs + attention mask, shape [batch, seqLen],
// ready to ship through N-API to spike/src/embed.mojo's embedTokens().
//
// Tokenizer JSON is fetched lazily on first call and cached under HF_HOME
// (set in the RunPod bootstrap script to /workspace/persist/model-cache).
// First call on a fresh pod hits the network; subsequent calls are local.

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';  // ONNX-converted checkpoint; repo name retained after namespace migration
const MAX_SEQ_LEN = 128;  // spike cap; MiniLM allows up to 512

let tokenizerPromise = null;

async function getTokenizer() {
  if (!tokenizerPromise) {
    const { AutoTokenizer } = await import('@huggingface/transformers');
    tokenizerPromise = AutoTokenizer.from_pretrained(MODEL_ID);
  }
  return tokenizerPromise;
}

// Tokenize an array of strings. Returns { ids, mask, batch, seqLen } where
// ids and mask are Int32Array of shape [batch, seqLen], padded to the longest
// sequence in the batch (capped at MAX_SEQ_LEN).
async function tokenize(texts) {
  const tokenizer = await getTokenizer();
  const encoded = await tokenizer(texts, {
    padding: true,
    truncation: true,
    max_length: MAX_SEQ_LEN,
  });

  // @huggingface/transformers v4 returns Tensor objects with:
  //   .dims = [batch, seqLen]
  //   .data = BigInt64Array (token IDs) / BigInt64Array (mask)
  // MiniLM vocab=30522 fits in int32, so we safely down-cast BigInt -> Number.
  const [batch, seqLen] = encoded.input_ids.dims;
  const idsData = encoded.input_ids.data;
  const maskData = encoded.attention_mask.data;
  const n = batch * seqLen;

  const ids = new Int32Array(n);
  const mask = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    ids[i] = Number(idsData[i]);
    mask[i] = Number(maskData[i]);
  }
  return { ids, mask, batch, seqLen };
}

module.exports = { tokenize, MAX_SEQ_LEN, MODEL_ID };
