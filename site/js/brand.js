// ============================================================================
// Brand & narrative copy — single source of truth
// ============================================================================
//
// Every client-facing string lives here, for two reasons.
//
//   1. Attribution has to be right. This is a QualiZeal AI-CoE build, delivered
//      for Nova Biomedical. The previous footer credited it to "Nova Biomedical
//      AI-CoE", which gives away the authorship of the work.
//
//   2. This demo will be shown to other prospects. Re-pointing it at a new
//      client should be a change to ONE object, not a search across templates.
//      Change `client` below and the whole narrative re-labels itself.
//
// Copy guidance applied throughout: lead with the business outcome, keep the
// method underneath it. A reader should understand the value before they meet
// the word "retrieval".

export const BRAND = {
  // ---- who built it, and for whom ----------------------------------------
  product: 'Knowledge Fabric',
  builder: {
    name: 'QualiZeal',
    unit: 'AI Center of Excellence',
    unitShort: 'AI-CoE',
    url: 'https://www.qualizeal.com',
  },
  client: {
    name: 'Nova Biomedical',
    url: 'https://www.novabiomedical.com',
    // What the corpus actually is, in the client's language.
    corpusDescription: 'product documentation, instructions for use, and FDA regulatory records',
  },
  stage: 'MVP',

  // ---- assembled strings --------------------------------------------------
  get navContext() { return 'Built by ' + this.builder.name + ' ' + this.builder.unitShort; },
  get heroEyebrow() {
    return this.builder.name + ' ' + this.builder.unitShort + ' — ' + this.stage +
           ' for ' + this.client.name;
  },
  get footerBrand() {
    return this.product + ' — a ' + this.builder.name + ' ' + this.builder.unit +
           ' build, delivered for ' + this.client.name;
  },
  get footerNote() {
    return 'Every answer traced to its source document and page. No content generated, ' +
           'nothing paraphrased.';
  },
};

// ---------------------------------------------------------------------------
// Narrative copy — outcome first, mechanism second
// ---------------------------------------------------------------------------
export const COPY = {
  hero: {
    title: ['Every answer,', 'traced to its source.'],
    sub: 'Your documentation already contains the answers. Knowledge Fabric finds them, ' +
         'explains how it found them, and shows you the page it came from — so your teams ' +
         'can act on what they read.',
    placeholder: 'Ask anything about the documentation…',
  },

  sections: {
    answer: {
      eyebrow: 'The Answer',
      title: 'Answers your team can act on.',
      sub: 'Written as a reply, not a search result — with every statement carrying the ' +
           'document and page it came from.',
    },
    explain: {
      eyebrow: 'The Evidence',
      title: 'See exactly why it said that.',
      sub: 'Every answer opens up: which documents were consulted, which were set aside, ' +
           'and how confident the system is in what it found.',
    },
    health: {
      eyebrow: 'Knowledge Health',
      title: 'Find the gaps before they cost you.',
      sub: 'Knowledge that sits in one document, that nobody can date, or that no other ' +
           'document references, is a risk you cannot see on a shared drive. This is that ' +
           'risk, measured.',
    },
    insights: {
      eyebrow: 'The Corpus',
      title: 'What your documentation actually covers.',
      sub: 'Where knowledge concentrates, where it thins out, and which topics depend on ' +
           'a single source.',
    },
  },

  // Plain-language value statements — what a business reader takes away.
  value: {
    grounded: {
      title: 'Nothing is invented',
      body: 'Answers are assembled from sentences that exist in your documents. If the ' +
            'documentation does not say it, the system does not say it either.',
    },
    traceable: {
      title: 'Every claim is checkable',
      body: 'Each statement links to the document and page it came from, so a reviewer ' +
            'can verify it in seconds rather than trusting it.',
    },
    honest: {
      title: 'It tells you when it is unsure',
      body: 'Confidence is measured from the evidence found, not displayed as decoration. ' +
            'A weak answer is labelled as one.',
    },
    connected: {
      title: 'It connects separate documents',
      body: 'A product manual and its regulatory record are linked automatically, so ' +
            'questions that span both can finally be answered.',
    },
  },
};

/** Apply brand strings to any element carrying data-brand="key". Keeps the
 *  markup declarative and means re-branding never touches the HTML. */
export function applyBrand(root = document) {
  const map = {
    product: BRAND.product,
    navContext: BRAND.navContext,
    heroEyebrow: BRAND.heroEyebrow,
    footerBrand: BRAND.footerBrand,
    footerNote: BRAND.footerNote,
    clientName: BRAND.client.name,
    builderName: BRAND.builder.name,
  };
  for (const el of root.querySelectorAll('[data-brand]')) {
    const v = map[el.dataset.brand];
    if (v) el.textContent = v;
  }
  for (const el of root.querySelectorAll('[data-brand-href]')) {
    const target = el.dataset.brandHref === 'builder' ? BRAND.builder.url : BRAND.client.url;
    if (target) el.setAttribute('href', target);
  }
}
