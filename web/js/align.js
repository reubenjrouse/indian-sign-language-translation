// Match recognised signs to the words of the English sentence the LLM wrote, so the
// screen can draw a line from each sign to its word(s). Words with no sign are the
// ones the LLM added for grammar. No browser APIs, so Node can test it.

const clean = (w) => w.toLowerCase().replace(/[^a-z0-9']/g, "");

/** "Good Morning" -> ["good", "morning"]; "You (plural)" -> ["you"]. */
export function signTokens(label) {
  return label.replace(/\(.*?\)/g, " ").split(/[\s-]+/).map(clean).filter(Boolean);
}

function sameWord(token, word) {
  if (!token || !word) return false;
  if (token === word) return true;
  // inflections: teacher/teachers, sick/sickness, happy/happily, i/i'm
  const [a, b] = token.length <= word.length ? [token, word] : [word, token];
  if (a.length >= 3 && b.startsWith(a.slice(0, Math.max(3, a.length - 1)))) return true;
  return word.split("'")[0] === token; // "i'm" -> i, "we're" -> we
}

/**
 * sentence: the English sentence. signs: [{ label, alternatives?: [label, ...] }].
 * Returns { words: [{ text, sign: index | null }], signs: [{ word: index | null, used: label }] }.
 * Each sign takes the first unclaimed run of words matching its tokens, searched left to
 * right from the previous match (ISL order often differs from English, so it falls back
 * to the whole sentence). An alternative label is tried when the top one isn't found,
 * which shows when the LLM chose a runner-up guess.
 */
export function alignSentence(sentence, signs) {
  const words = sentence.split(/\s+/).filter(Boolean).map((text) => ({ text, sign: null }));
  const keys = words.map((w) => clean(w.text));
  const result = signs.map(() => ({ word: null, used: null }));
  let cursor = 0;

  const findRun = (tokens, from) => {
    for (let start = from; start <= keys.length - tokens.length; start++) {
      if (tokens.every((t, k) => words[start + k].sign === null && sameWord(t, keys[start + k]))) return start;
    }
    return -1;
  };

  signs.forEach((sign, i) => {
    for (const label of [sign.label, ...(sign.alternatives ?? [])]) {
      const tokens = signTokens(label);
      if (!tokens.length) continue;
      let at = findRun(tokens, cursor);
      if (at < 0) at = findRun(tokens, 0);
      if (at < 0) continue;
      tokens.forEach((_, k) => (words[at + k].sign = i));
      result[i] = { word: at, used: label };
      cursor = at + tokens.length;
      break;
    }
  });
  return { words, signs: result };
}
