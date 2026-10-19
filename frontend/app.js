(() => {
  const CFG = window.DEMO_CONFIG, D = window.DEMO, CAT = window.CATALOG;
  const $ = s => document.querySelector(s);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  // Dataset names are ALL CAPS, so convert to Title Case for display
  const title = n => n.trim().toLowerCase().replace(/(^|[\s\-'(])([a-z])/g, (m, a, b) => a + b.toUpperCase());
  const ICONS = [[/HEART|LOVE/, '💝'], [/CANDLE|LIGHT|LANTERN/, '🕯️'], [/CHRISTMAS|XMAS|SNOW/, '🎄'], [/BAG/, '👜'], [/TEA|MUG|CUP/, '☕'],
    [/CAKE|BAKING|CASES/, '🧁'], [/BIRD/, '🐦'], [/FLOWER|ROSE|GARDEN/, '🌷'], [/CARD|WRAP|RIBBON|PAPER/, '🎀'], [/BOX|TIN|JAR|BASKET/, '🧺'], [/STAR|ANGEL|FAIRY/, '⭐']];
  const icon = n => { const m = ICONS.find(([r]) => r.test(n.toUpperCase())); return m ? m[1] : '🎁'; };
  const card = (n, note) => `<div class="card"><div class="tile">${icon(n)}</div><div class="info"><h3>${esc(title(n))}</h3>` +
    `${note ? `<p>${esc(note)}</p>` : ''}${CAT[n.trim()] ? `<b>£${CAT[n.trim()].toFixed(2)}</b>` : ''}</div></div>`;
  const grid = (items, fn) => `<div class="grid">${items.map(fn).join('')}</div>`;
  const times = n => 'Ordered ' + n.toLocaleString() + ' times';
  let cur;

  function show(i) {
    const p = cur = D.personas[i];
    $('#shoppers').innerHTML = D.personas.map((s, j) => `<button class="${j === i ? 'on' : ''}" data-i="${j}">${esc(s.name)}<small>${esc(s.tagline)}</small></button>`).join('');
    $('#page').innerHTML = `<h2>${esc(p.name)}'s recent purchases</h2>${grid(p.recent, n => card(n))}` +
      `<div class="ask"><p>The engine compares ${esc(p.name)} with thousands of other shoppers, finds the most similar ones, and suggests what they bought that ${esc(p.name)} hasn't.</p>` +
      `<button id="go">Get recommendations for ${esc(p.name)}</button></div><div id="out"></div>` +
      `<h2>Bestsellers <small>the same list for every shopper</small></h2>${grid(D.bestsellers, b => card(b.name, times(b.orders)))}`;
  }

  const fail = (head, body) => `<div class="err"><b>${head}</b><br>${body}</div>`;

  async function ask() {
    const p = cur, out = $('#out');
    if (!CFG || !CFG.API_KEY || CFG.API_KEY.startsWith('paste')) {
      out.innerHTML = fail('API key not set', 'Copy <code>config.example.js</code> to <code>config.js</code> and paste the API_KEY from your .env file.'); return;
    }
    out.innerHTML = '<p class="msg">Asking the recommendation engine…</p>';
    const t0 = performance.now();
    try {
      const [rec, nw] = await Promise.all(['recommend', 'new-arrivals'].map(async e => {
        const r = await fetch(`${CFG.API_BASE}/${e}/${p.customerId}`, { headers: { 'X-API-Key': CFG.API_KEY } });
        if (!r.ok) throw { status: r.status };
        return r.json();
      }));
      if (cur !== p) return;
      const ms = Math.round(performance.now() - t0);
      out.innerHTML = `<h2>Recommended for ${esc(p.name)}</h2>${grid(rec.recommendations, r => card(r.product, times(r.orders)))}` +
        (nw.new_arrivals.length ? `<h2>New in, picked for ${esc(p.name)}</h2>${grid(nw.new_arrivals, r => card(r.product, 'Matches your taste'))}` +
          '<p class="foot">New-arrival products are sample items created for this demo.</p>' : '') +
        `<p class="live">✔ Live answer from the server: GET /recommend/${p.customerId} and /new-arrivals/${p.customerId} took ${ms} ms</p>` +
        `<details><summary>See the raw response from the API</summary><pre>${esc(JSON.stringify({ recommend: rec, new_arrivals: nw }, null, 2))}</pre></details>`;
    } catch (e) {
      if (cur !== p) return;
      out.innerHTML = e.status === 401 || e.status === 403 ? fail('The server rejected the API key', 'Check that API_KEY in config.js matches the one in your .env file.')
        : e.status === 404 ? fail('Customer not found', 'This shopper is not in the data the server loaded. Check DATA_PATH in your .env.')
        : e.status === 429 ? fail('Too many requests', 'Wait a minute and try again.')
        : fail('Cannot reach the recommendation server', `Nothing answered at ${esc(CFG.API_BASE)}. Start it with <code>uvicorn main:app --reload</code> and try again.`);
    }
  }

  document.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.dataset.i !== undefined) show(+b.dataset.i); else if (b.id === 'go') ask();
  });
  show(0);
})();
