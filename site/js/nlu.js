// ============================================================================
// NLU — question understanding
// ============================================================================
//
// The failure this module exists to fix:
//
//   Q: "What is the clinical significance of measuring lactate?"
//   A: "Lactate Plus — Blood Lactate Measuring Meter Kit. Associated FDA product
//       codes: KHP. Commercial distribution status: In Commercial Distribution."
//
// Retrieval was not broken. That UDI record contains "lactate" three times in
// forty words, so it is the strongest lexical match in the corpus. What was
// missing is any representation of what the *question* was asking for. A
// question about clinical significance wants a sentence that explains what an
// analyte tells a clinician — it can never be satisfied by a registration record,
// no matter how many query terms that record contains.
//
// So before retrieval is scored, we decide three things:
//   1. intent      — what kind of answer would satisfy this question
//   2. focus       — which entities/analytes the question is about
//   3. routing     — which evidence types can plausibly answer it, and which
//                    cannot, expressed as multiplicative priors
//
// No model, no training data. Cue patterns over a closed domain, which is the
// appropriate tool when the intent space is small and known.

const TOKEN_RE = /[A-Za-z0-9]+/g;

const STOP = new Set(['the','a','an','of','for','to','in','on','at','is','are','was','were',
  'be','been','do','does','did','what','which','how','why','when','where','who','whom','and',
  'or','but','if','then','with','by','from','as','that','this','these','those','it','its',
  'can','could','should','would','will','shall','may','might','must','i','we','you','they',
  'my','our','your','their','me','us','them','about','into','over','under','than','so','such']);

// ---------------------------------------------------------------------------
// Intents
// ---------------------------------------------------------------------------
// cues       : regexes on the raw question
// answerCues : phrases a *satisfying answer sentence* tends to contain. These are
//              a soft prior on sentence scoring, not a filter.
// prefer     : evidence types multiplied UP
// penalise   : evidence types multiplied DOWN
// wants      : short label shown in the UI so the routing is legible
export const INTENTS = {
  CLINICAL_SIGNIFICANCE: {
    cues: [/clinical (significance|utility|relevance|value|importance)/i,
           /why (is|are|do|does|would|should).*(measur|test|monitor|import|matter)/i,
           /what does .* (indicate|tell|mean|signify)/i,
           /significance of/i, /purpose of measuring/i],
    answerCues: [[/used in the (diagnosis|treatment|management|monitoring)/i, 1.0],
                 [/clinical utility/i, 1.0], [/diagnosis and (treatment|monitoring)/i, 1.0],
                 [/indicat(es|ive|or) of/i, 0.8], [/marker (of|for)/i, 0.8],
                 [/\bassess(es|ment of)\b/i, 0.6], [/associated with/i, 0.5],
                 [/reflects?/i, 0.5], [/elevated|decreased|abnormal/i, 0.4]],
    prefer: { document: 1.35 },
    penalise: { udi: 0.05, ifu_catalogue: 0.10, classification: 0.35, clearance: 0.7 },
    wants: 'clinical meaning',
  },
  INTENDED_USE: {
    cues: [/intended use/i, /indications? for use/i, /what is .* (used for|for\?)/i,
           /purpose of the/i, /designed (to|for)/i],
    answerCues: [[/\bis intended for\b/i, 1.0], [/\bindicated for\b/i, 0.9],
                 [/for (the )?(quantitative|qualitative) (determination|measurement)/i, 0.8],
                 [/\bin vitro diagnostic\b/i, 0.5], [/intended use/i, 0.25]],
    prefer: { document: 1.2, clearance: 1.1 },
    penalise: { udi: 0.3, ifu_catalogue: 0.3 },
    wants: 'intended-use statement',
  },
  INTERFERENCE: {
    cues: [/interfer/i, /cross[- ]?reactiv/i, /affect (the )?(results?|readings?|accuracy)/i,
           /substances? .* (affect|interfere)/i, /false (high|low|result)/i],
    answerCues: [[/(ascorbic acid|acetaminophen|uric acid|maltose|galactose|bilirubin|dopamine)/i, 1.0],
                 [/no (clinically )?significant (interference|effect)/i, 0.9],
                 [/did not (cause|show|produce)/i, 0.8],
                 [/interfer(e|ence|ing)/i, 0.6],
                 [/substances? (tested|listed|evaluated)/i, 0.5],
                 [/concentrations? (of|up to|listed)/i, 0.35]],
    prefer: { document: 1.3 },
    penalise: { udi: 0.05, ifu_catalogue: 0.1, classification: 0.4 },
    wants: 'interference findings',
  },
  MECHANISM: {
    cues: [/how does .* (work|measure|detect|determine)/i, /principle of/i,
           /methodolog/i, /based on what/i, /technolog(y|ies) (used|behind)/i,
           /how is .* measured/i],
    answerCues: [[/is based on/i, 1.0], [/methodolog/i, 0.9], [/principle/i, 0.8],
                 [/biosensor|amperometric|electrochemi/i, 0.8], [/proportional to/i, 0.7],
                 [/electrode/i, 0.5], [/enzyme|reaction|oxidase/i, 0.5]],
    prefer: { document: 1.3 },
    penalise: { udi: 0.05, ifu_catalogue: 0.1, clearance: 0.8 },
    wants: 'measurement principle',
  },
  CAUSAL: {
    cues: [/how does .* (affect|impact|influence|change)/i,
           /(effect|impact|influence) of .* on/i, /what happens (if|when)/i,
           /does .* affect/i],
    answerCues: [[/corrects? for/i, 1.0], [/compensat/i, 1.0],
                 [/affects?|influenc(e|es)|impacts?/i, 0.9],
                 [/(increase|decrease|elevat|reduc)(e|es|ed|ion)/i, 0.6],
                 [/because|due to|caused by/i, 0.6], [/results? in/i, 0.4]],
    prefer: { document: 1.3 },
    penalise: { udi: 0.05, ifu_catalogue: 0.1, classification: 0.5 },
    wants: 'cause-and-effect',
  },
  SPECIFICATION: {
    cues: [/measurement range|reportable range|linearity/i,
           /how (much|long|many|fast)/i, /what (is the )?(range|volume|time|accuracy|precision)/i,
           /sample (size|volume)/i, /test time/i, /storage (temperature|condition)/i],
    answerCues: [/range (is|of)/i, /\b\d+\s*(-|to|–)\s*\d+\b/, /mg\/dL|mmol\/L|µL|uL|seconds?|°C/i,
                 /requires? (a )?(sample|volume)/i],
    prefer: { document: 1.25 },
    penalise: { udi: 0.2, ifu_catalogue: 0.2 },
    wants: 'a specification value',
  },
  PROCEDURE: {
    cues: [/how (do|to|should) (i|you|we)/i, /steps? (to|for)/i, /procedure for/i,
           /instructions? for/i, /how .* perform/i],
    answerCues: [/^(press|insert|remove|apply|wait|select|clean|dispose|hold|touch|open|close)/i,
                 /step \d/i, /then |next |after /i],
    prefer: { document: 1.3 },
    penalise: { udi: 0.05, ifu_catalogue: 0.1, classification: 0.3, clearance: 0.6 },
    wants: 'a procedure',
  },
  REGULATORY: {
    cues: [/\bK\d{6}\b/, /510\s*\(?k\)?/i, /clearance|cleared by|fda (clear|approv)/i,
           /product code/i, /recall/i, /predicate/i, /substantially equivalent/i,
           /regulat(ion|ory)/i, /21 CFR/i],
    answerCues: [/510\(k\)/i, /\bK\d{6}\b/, /product code/i, /substantially equivalent/i,
                 /cleared|clearance/i, /recall/i, /class (I|II|III)\b/],
    prefer: { clearance: 1.5, recall: 1.4, classification: 1.3 },
    penalise: { udi: 0.6 },
    wants: 'a regulatory fact',
  },
  TEMPORAL: {
    cues: [/\bwhen (was|were|did|is)\b/i, /what (date|year)/i, /how (old|recent)/i],
    answerCues: [/\b(19|20)\d{2}\b/, /decision|received|initiated|issued|revised/i],
    prefer: { clearance: 1.4, recall: 1.3 },
    penalise: { udi: 0.5 },
    wants: 'a date',
  },
  COMPARISON: {
    cues: [/difference between/i, /\bvs\.?\b|versus/i, /compare[ds]?|comparison/i,
           /better than|instead of/i],
    answerCues: [/compared (to|with)/i, /whereas|while|however|unlike/i,
                 /difference|differs?/i, /both|either/i],
    prefer: { document: 1.2, clearance: 1.1 },
    penalise: { udi: 0.2, ifu_catalogue: 0.2 },
    wants: 'a comparison',
  },
  DEFINITION: {
    cues: [/^what (is|are) (a |an |the )?[\w\s-]+\??$/i, /define|definition of/i,
           /what does .* stand for/i],
    answerCues: [/\bis a\b|\bis an\b|\bare\b/, /refers? to/i, /consists? of/i, /means/i],
    prefer: { document: 1.15, classification: 1.1 },
    penalise: { udi: 0.3 },
    wants: 'a definition',
  },
};

export const DEFAULT_INTENT = {
  name: 'GENERAL', answerCues: [], prefer: { document: 1.1 },
  penalise: { udi: 0.4, ifu_catalogue: 0.5 }, wants: 'a direct answer',
};

// Domain analytes/entities worth recognising as the question's focus.
const FOCUS_TERMS = [
  'glucose','lactate','creatinine','hematocrit','haematocrit','ketone','hemoglobin',
  'haemoglobin','sodium','potassium','chloride','calcium','magnesium','bilirubin',
  'urea','bun','ph','pco2','po2','hba1c','albumin','uacr','egfr','osmolality',
  'statstrip','statsensor','xpress2','xpress','allegro','prime','bioprofile','flex2',
  'lactate plus','nova max','primary','meter','analyzer','analyser','strip','cartridge',
];

// Mutually-exclusive entity families. A sentence about StatSensor Creatinine
// cannot answer a question about StatStrip Glucose, however well its words match.
export const ENTITY_FAMILIES = {
  analyte: ['glucose', 'lactate', 'creatinine', 'ketone', 'hemoglobin', 'haemoglobin',
            'hba1c', 'albumin', 'uacr', 'egfr', 'bilirubin', 'osmolality'],
  product: ['statstrip', 'statsensor', 'xpress2', 'allegro', 'bioprofile', 'flex2',
            'lactate plus', 'nova max', 'prime plus', 'nova primary'],
};

/** Entities of the same family present in a text, for conflict detection. */
export function familyMembers(text, family) {
  const low = String(text).toLowerCase();
  return ENTITY_FAMILIES[family].filter(t => low.includes(t));
}

export function tokens(text) {
  return (String(text).toLowerCase().match(TOKEN_RE) || []).filter(t => t.length > 1);
}

export function contentTokens(text) {
  return tokens(text).filter(t => !STOP.has(t));
}

/** Classify intent by cue-pattern hits. Ties break toward the more specific
 *  intent, since specific intents carry stronger routing priors. */
export function classifyIntent(question) {
  const q = String(question || '');
  let best = null, bestScore = 0;
  for (const [name, spec] of Object.entries(INTENTS)) {
    let score = 0;
    for (const re of spec.cues) if (re.test(q)) score += 1;
    if (score > bestScore) { bestScore = score; best = { name, ...spec }; }
  }
  return best || { ...DEFAULT_INTENT };
}

/** Entities/analytes the question is about — used for focus bonuses and to
 *  detect when a candidate sentence is about something else entirely. */
export function extractFocus(question) {
  const low = String(question || '').toLowerCase();
  const found = [];
  for (const term of FOCUS_TERMS) {
    if (new RegExp(`\\b${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`).test(low)) {
      found.push(term);
    }
  }
  // longest-first, de-duplicated by containment ("lactate plus" beats "lactate")
  found.sort((a, b) => b.length - a.length);
  const kept = [];
  for (const f of found) if (!kept.some(k => k.includes(f))) kept.push(f);
  return kept;
}

/**
 * Evidence prior for a chunk given the question intent.
 * Multiplicative, so it reshapes ranking without discarding retrieval signal.
 */
// Sentences that talk *about* a regulatory comparison rather than stating a fact.
// "X is substantially equivalent in terms of intended use to predicate Y" contains
// the phrase "intended use" but is not an intended-use statement.
export const META_REGULATORY_RE =
  /substantially equivalent|predicate device|comparison of predicate|in terms of intended use|proposed device|characteristic predicate/i;

export function evidencePrior(chunk, intent) {
  const rt = (chunk.meta && chunk.meta.record_type) || '';
  const st = chunk.source_type || '';
  let prior = 1.0;
  if (intent.prefer) {
    if (intent.prefer[rt]) prior *= intent.prefer[rt];
    if (intent.prefer[st]) prior *= intent.prefer[st];
  }
  if (intent.penalise) {
    if (intent.penalise[rt]) prior *= intent.penalise[rt];
    if (intent.penalise[st]) prior *= intent.penalise[st];
  }
  // Chunk-level prose quality from the build (0..1). A chunk that is mostly
  // table rows should not win a prose question.
  if (typeof chunk.quality === 'number') prior *= (0.55 + 0.45 * chunk.quality);
  return prior;
}

/**
 * Re-rank retrieval results by intent-conditioned evidence priors.
 *
 * This must run BEFORE the summariser sees the pool, not after. Filtering at
 * summarisation time is too late: if the top-12 passages are all UDI
 * registration records — which is exactly what happens for
 * "clinical significance of measuring lactate", because those records are the
 * densest lexical match for "lactate" in the corpus — then the useful IFU
 * passages were never in the pool to begin with, and the summariser can only
 * choose between bad options or return nothing.
 */
export function rerankByIntent(ranked, chunks, intent, { keep = 24 } = {}) {
  const scored = ranked.map(r => {
    const chunk = chunks[r.chunkIdx];
    const prior = chunk ? evidencePrior(chunk, intent) : 0;
    return { ...r, baseScore: r.score, prior, score: r.score * prior };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.filter(r => r.prior > 0.12).slice(0, keep);
}

export function analyseQuestion(question) {
  const intent = classifyIntent(question);
  const focus = extractFocus(question);
  const terms = contentTokens(question);
  return {
    question, intent, focus, terms,
    isIdentifierLookup: /\bK\d{6}\b/i.test(question),
    expects: intent.wants || DEFAULT_INTENT.wants,
  };
}
