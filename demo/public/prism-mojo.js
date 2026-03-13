// Prism.js language definition for Mojo
// Based on Python grammar with Mojo-specific extensions

if (typeof Prism !== 'undefined') {
  Prism.languages.mojo = {
    'comment': {
      pattern: /#.*/,
      greedy: true,
    },
    'string': {
      pattern: /"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/,
      greedy: true,
    },
    'decorator': {
      pattern: /@\w+(?:\([^)]*\))?/,
      greedy: true,
    },
    'keyword': /\b(?:fn|def|struct|trait|var|let|alias|comptime|owned|borrowed|inout|raises|capturing|mut|unified|if|else|elif|for|while|in|range|return|try|except|raise|from|import|with|as|and|or|not|pass|break|continue)\b/,
    'builtin': /\b(?:vectorize|parallelize|simd_width_of|rebind|print|len|range|min|max|abs|Int|UInt8|UInt16|UInt32|UInt64|Float16|Float32|Float64|Bool|String|Byte|True|False|None|DType|SIMD|Scalar|UnsafePointer|InlineArray|Error)\b/,
    'function': {
      pattern: /\b(?!\d)\w+(?=\s*[\[(])/,
    },
    'number': /\b(?:0[xX][\da-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|\d[\d_]*(?:\.[\d_]*)?(?:[eE][+-]?\d[\d_]*)?)\b/,
    'operator': /->|[+\-*/%&|^~<>=!]=?|<<?|>>?|\*\*/,
    'punctuation': /[{}[\]();:.,]/,
  };

  // Re-highlight any existing code blocks
  if (Prism.highlightAll) {
    Prism.highlightAll();
  }
}
