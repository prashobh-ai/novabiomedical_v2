// ============================================================================
// Typeahead — suggests from the tested question bank
// ============================================================================
//
// Two jobs, one control.
//
//   For the user: they do not know what this corpus can answer. A blank box with
//   a blinking cursor is the worst possible affordance for that — they type
//   something the documentation cannot support, get a weak answer, and conclude
//   the system is weak.
//
//   For the demo: every suggestion comes from the question bank, which is the
//   set we have actually tested. Someone driving the demo — or a director poking
//   at it on their phone mid-call — is steered toward questions with known-good
//   answers instead of discovering an edge case live.
//
// Matching is deliberately forgiving, in four tiers, because a reviewer typing
// on a phone will not reproduce our exact phrasing:
//
//   1. prefix       "what is the int"     -> starts-with, ranked top
//   2. substring    "intended use"        -> appears anywhere
//   3. all-words    "use statstrip"       -> every typed word present, any order
//   4. subsequence  "sttrp gluc"          -> characters in order (typo tolerant)
//
// Built as a WAI-ARIA combobox: roles, aria-activedescendant, full keyboard
// support. It is the one control a reviewer touches first, so it has to behave
// the way every other search box they have used behaves.

const MAX_RESULTS = 6;
const MIN_CHARS = 1;

const STOP = new Set(['the', 'a', 'an', 'of', 'is', 'are', 'to', 'in', 'for', 'on', 'what', 'how', 'does', 'do']);

function norm(s) {
  return String(s || '').toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
}

/** Characters appear in order, not necessarily adjacent. Tolerates typos and
 *  the aggressive abbreviation people use on a phone keyboard. */
function subsequenceScore(needle, haystack) {
  let i = 0, gaps = 0, lastHit = -1;
  for (let j = 0; j < haystack.length && i < needle.length; j++) {
    if (haystack[j] === needle[i]) {
      if (lastHit >= 0) gaps += j - lastHit - 1;
      lastHit = j;
      i++;
    }
  }
  if (i < needle.length) return 0;
  return 1 / (1 + gaps / Math.max(needle.length, 1));
}

/**
 * @param {string} query
 * @param {string[]} bank
 * @returns {Array<{question:string, score:number, tier:string}>}
 */
export function suggest(query, bank) {
  const q = norm(query);
  if (q.length < MIN_CHARS) return [];

  const words = q.split(' ').filter(w => w && !STOP.has(w));
  const results = [];

  for (const question of bank) {
    const h = norm(question);
    let score = 0, tier = '';

    if (h.startsWith(q)) {
      score = 100 - h.length * 0.01;             // shorter completions first
      tier = 'prefix';
    } else if (h.includes(q)) {
      score = 80 - h.indexOf(q) * 0.05;          // earlier match is stronger
      tier = 'contains';
    } else if (words.length && words.every(w => h.includes(w))) {
      score = 60 + words.length;
      tier = 'words';
    } else {
      const sub = subsequenceScore(q.replace(/\s/g, ''), h.replace(/\s/g, ''));
      if (sub > 0.35) {
        score = 30 * sub;
        tier = 'fuzzy';
      }
    }

    if (score > 0) results.push({ question, score, tier });
  }

  results.sort((a, b) => b.score - a.score);
  return results.slice(0, MAX_RESULTS);
}

/** Wrap the matched span so the user can see WHY a row is being offered. */
function highlight(question, query) {
  const q = norm(query);
  const h = norm(question);
  const at = h.indexOf(q);
  if (at < 0 || !q) return escapeHtml(question);

  // norm() collapses punctuation, so map the normalised offset back onto the
  // original string by walking both in step.
  let orig = 0, normed = 0, start = -1, end = -1;
  while (orig < question.length && normed <= at + q.length) {
    const c = question[orig].toLowerCase();
    const isSep = !/[a-z0-9]/.test(c);
    if (normed === at && start < 0) start = orig;
    if (normed === at + q.length && end < 0) end = orig;
    if (!isSep) normed++;
    else if (normed > 0 && h[normed - 1] !== ' ') normed++;
    orig++;
  }
  if (start < 0) return escapeHtml(question);
  if (end < 0) end = question.length;

  return escapeHtml(question.slice(0, start)) +
         '<mark>' + escapeHtml(question.slice(start, end)) + '</mark>' +
         escapeHtml(question.slice(end));
}

export class Typeahead {
  /**
   * @param {HTMLInputElement} input
   * @param {string[]} bank
   * @param {(q:string)=>void} onSelect  called when a suggestion is chosen
   */
  constructor(input, bank, onSelect) {
    if (!input) return;
    this.input = input;
    this.bank = bank || [];
    this.onSelect = onSelect;
    this.items = [];
    this.active = -1;
    this.open = false;

    this.list = document.createElement('ul');
    this.list.className = 'ta-list';
    this.list.id = `${input.id}-suggestions`;
    this.list.setAttribute('role', 'listbox');
    this.list.hidden = true;

    // Anchor to the input's positioned parent so the dropdown tracks it on
    // scroll and resize without any per-frame measurement.
    const host = input.closest('form') || input.parentElement;
    if (host) {
      if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
      host.appendChild(this.list);
    }

    input.setAttribute('role', 'combobox');
    input.setAttribute('aria-expanded', 'false');
    input.setAttribute('aria-controls', this.list.id);
    input.setAttribute('aria-autocomplete', 'list');
    input.setAttribute('autocomplete', 'off');

    this._bind();
  }

  _bind() {
    const { input } = this;

    input.addEventListener('input', () => this.refresh());
    input.addEventListener('focus', () => { if (input.value.trim()) this.refresh(); });

    input.addEventListener('keydown', e => {
      if (!this.open) {
        // Down-arrow on an empty box offers the bank — a discoverable way in
        // for someone who does not know what to ask.
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          this.show(this.bank.slice(0, MAX_RESULTS).map(q => ({ question: q, tier: 'browse' })), '');
        }
        return;
      }
      switch (e.key) {
        case 'ArrowDown': e.preventDefault(); this.move(1); break;
        case 'ArrowUp':   e.preventDefault(); this.move(-1); break;
        case 'Enter':
          if (this.active >= 0) { e.preventDefault(); this.choose(this.active); }
          break;
        case 'Tab':
          if (this.active >= 0) { e.preventDefault(); this.commit(this.items[this.active].question); }
          break;
        case 'Escape': e.preventDefault(); this.hide(); break;
        default: break;
      }
    });

    // pointerdown, not click: blur fires first on some mobile browsers and would
    // close the list before the tap registers.
    this.list.addEventListener('pointerdown', e => {
      const li = e.target.closest('.ta-item');
      if (!li) return;
      e.preventDefault();
      this.choose(Number(li.dataset.index));
    });

    document.addEventListener('pointerdown', e => {
      if (!this.list.contains(e.target) && e.target !== input) this.hide();
    });
  }

  refresh() {
    const q = this.input.value.trim();
    if (!q) { this.hide(); return; }
    const matches = suggest(q, this.bank);
    if (!matches.length) { this.hide(); return; }
    this.show(matches, q);
  }

  show(items, query) {
    this.items = items;
    this.active = -1;
    this.list.innerHTML = items.map((it, i) => `
      <li class="ta-item" role="option" id="${this.list.id}-${i}" data-index="${i}"
          aria-selected="false">
        <span class="ta-icon" aria-hidden="true"></span>
        <span class="ta-text">${query ? highlight(it.question, query) : escapeHtml(it.question)}</span>
        ${it.tier === 'fuzzy' ? '<span class="ta-tier">similar</span>' : ''}
      </li>`).join('') +
      `<li class="ta-foot" role="presentation">
         ${items.length} tested question${items.length === 1 ? '' : 's'} · ↑↓ to browse · ↵ to ask
       </li>`;
    this.list.hidden = false;
    this.open = true;
    this.input.setAttribute('aria-expanded', 'true');
  }

  hide() {
    this.list.hidden = true;
    this.open = false;
    this.active = -1;
    this.input.setAttribute('aria-expanded', 'false');
    this.input.removeAttribute('aria-activedescendant');
  }

  move(delta) {
    const n = this.items.length;
    if (!n) return;
    this.active = (this.active + delta + n) % n;
    [...this.list.querySelectorAll('.ta-item')].forEach((el, i) => {
      const on = i === this.active;
      el.classList.toggle('is-active', on);
      el.setAttribute('aria-selected', String(on));
      if (on) {
        this.input.setAttribute('aria-activedescendant', el.id);
        // Not universal in embedded webviews. An exception here fires inside the
        // keydown handler and would kill keyboard navigation outright, so scrolling
        // the active row into view must never be load-bearing.
        if (typeof el.scrollIntoView === 'function') {
          el.scrollIntoView({ block: 'nearest' });
        }
      }
    });
  }

  /** Fill the box without asking — lets the user edit before submitting. */
  commit(question) {
    this.input.value = question;
    this.hide();
    this.input.focus();
  }

  choose(index) {
    const item = this.items[index];
    if (!item) return;
    this.input.value = item.question;
    this.hide();
    if (this.onSelect) this.onSelect(item.question);
  }

  destroy() {
    this.list.remove();
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
