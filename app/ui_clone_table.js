export default function ({ parentElement, data }) {
  const columns = (data?.columns || []).map((column, index) => ({ ...column, index }));
  const visibleColumns = columns.filter(column => column.kind !== 'selected');
  const sourceRows = (data?.rows || []).map((values, index) => ({ values, index }));
  const pageSize = Math.max(1, Math.floor(Number(data?.pageSize) || 8));
  const selectedColumn = columns.find(column => column.kind === 'selected');
  const coverageColumn = columns.find(column => normalized(column.name) === 'item_coverage_ratio');
  const state = parentElement.__cdhCloneTableState || { page: 0, sort: null, ascending: true };
  parentElement.__cdhCloneTableState = state;
  if (state.sort !== null && !visibleColumns.some(column => column.name === state.sort)) state.sort = null;
  const root = element('section', 'clone-table');
  root.setAttribute('aria-label', data?.label || 'Medicine records');
  parentElement.appendChild(root);
  const numeric = new Intl.NumberFormat('en-PH', { maximumFractionDigits: 2 });
  const currency = new Intl.NumberFormat('en-PH', {
    style: 'currency', currency: 'PHP', minimumFractionDigits: 2, maximumFractionDigits: 2,
  });
  const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

  function normalized(value) { return String(value).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, ''); }
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }
  function missing(value) { return value === null || value === undefined || (typeof value === 'number' && !Number.isFinite(value)); }
  function pretty(value) {
    const text = String(value).replaceAll('_', ' ').trim();
    return text === text.toUpperCase() ? text.charAt(0) + text.slice(1).toLowerCase() : text;
  }
  function formatNumber(value, column) {
    const name = normalized(column.name);
    if (/(?:^|_)(?:cost|value|budget|price|amount)(?:_|$)/.test(name) && !/(percent|pct|ratio|share|level)/.test(name)) {
      return currency.format(value);
    }
    if (/(percent|pct)$/.test(name) || column.label.includes('%')) return numeric.format(value) + '%';
    return numeric.format(value);
  }
  function chip(text, tone, variant = '', icon = '') {
    const node = element('span', ['clone-chip', tone, variant].filter(Boolean).join(' '));
    if (icon) {
      const marker = element('span', '', icon);
      marker.setAttribute('aria-hidden', 'true');
      node.appendChild(marker);
    }
    node.appendChild(element('span', '', text));
    return node;
  }
  function statusChip(value) {
    const name = normalized(value);
    if (['pass', 'passed', 'good', 'ready', 'ok', 'complete', 'valid', 'validated'].includes(name)) return chip(pretty(value) === 'Pass' ? 'Good' : pretty(value), 'good', 'status', '✓');
    if (['review', 'needs_review', 'needs_attention', 'review_required', 'requires_review'].includes(name)) return chip('Needs review', 'amber', 'status', '⚠');
    if (['fail', 'failed', 'invalid', 'error'].includes(name)) return chip(pretty(value), 'red', 'status', '!');
    if (['evidence_available', 'historical_evidence_available', 'linked_history', 'linked'].includes(name)) return chip(pretty(value), 'blue', 'status', 'ⓘ');
    if (/(missing|unresolved|conflict|needs_|requires_)/.test(name)) return chip(pretty(value), 'amber', 'status', '⚠');
    return chip(pretty(value), 'neutral', 'status');
  }
  function cellValue(value, column) {
    if (missing(value)) return element('span', '', '—');
    const name = normalized(column.name);
    if (name === 'item_coverage_ratio' && typeof value === 'number') {
      const percent = value * 100;
      const progress = element('div', 'clone-progress' + (value < 1 ? ' partial' : ''));
      const line = element('div', 'clone-progress-label');
      line.append(element('span', '', value >= 1 ? 'Fully funded' : value > 0 ? 'Partially funded' : 'Unfunded'), element('strong', '', numeric.format(percent) + '%'));
      const track = element('div', 'clone-progress-track');
      track.setAttribute('role', 'progressbar');
      track.setAttribute('aria-label', 'Requirement funded');
      track.setAttribute('aria-valuemin', '0');
      track.setAttribute('aria-valuemax', '100');
      track.setAttribute('aria-valuenow', String(Math.min(100, Math.max(0, percent))));
      track.setAttribute('aria-valuetext', numeric.format(percent) + '% funded');
      const fill = element('div', 'clone-progress-fill');
      fill.style.width = Math.min(100, Math.max(0, percent)) + '%';
      track.appendChild(fill);
      progress.append(line, track);
      return progress;
    }
    if (['abc_class', 'abc_asof', 'abc'].includes(name)) {
      const abc = String(value).toUpperCase();
      if (['A', 'B', 'C'].includes(abc)) return chip('Class ' + abc, { A: 'teal', B: 'amber', C: 'neutral' }[abc], 'class');
      return chip(abc === 'UNCLASSIFIED_COST' ? 'Unclassified' : pretty(value), 'neutral', 'class');
    }
    if (name.includes('behavior_segment') || name === 'demand_pattern' || name === 'demand_segment') {
      const demand = normalized(value);
      return chip({ persistent_stable: 'Persistent stable', persistent_variable: 'Persistent variable', sparse: 'Sparse', non_moving: 'Non-moving' }[demand] || pretty(value), { persistent_stable: 'good', persistent_variable: 'blue', sparse: 'amber', non_moving: 'neutral' }[demand] || 'neutral');
    }
    if (name.includes('status') || name === 'data_quality' || name === 'source_quality_code') return statusChip(value);
    if (column.kind === 'boolean') return element('span', '', value ? 'Yes' : 'No');
    if (typeof value === 'number') return element('span', '', formatNumber(value, column));
    if (column.kind === 'date' && typeof value === 'string') return element('time', '', value.replace('T00:00:00.000', ''));
    return element('span', '', typeof value === 'object' ? JSON.stringify(value) : value);
  }
  function compareRows(left, right, column) {
    const a = left.values[column.index], b = right.values[column.index];
    if (missing(a) && missing(b)) return left.index - right.index;
    if (missing(a)) return 1;
    if (missing(b)) return -1;
    let result;
    if (typeof a === 'number' && typeof b === 'number') result = a - b;
    else if (typeof a === 'boolean' && typeof b === 'boolean') result = Number(a) - Number(b);
    else result = collator.compare(String(a), String(b));
    return (state.ascending ? result : -result) || left.index - right.index;
  }
  function render(focusTarget = null) {
    root.replaceChildren();
    const rows = sourceRows.slice();
    const sortedColumn = visibleColumns.find(column => column.name === state.sort);
    if (sortedColumn) rows.sort((a, b) => compareRows(a, b, sortedColumn));
    const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
    state.page = Math.max(0, Math.min(state.page, totalPages - 1));
    const first = state.page * pageSize;
    const frame = element('div', 'clone-table-frame');
    const scroll = element('div', 'clone-table-scroll');
    scroll.tabIndex = 0;
    scroll.setAttribute('role', 'region');
    scroll.setAttribute('aria-label', 'Scrollable records table');
    const table = element('table');
    table.style.minWidth = Math.min(visibleColumns.length * 158, 1600) + 'px';
    table.appendChild(element('caption', 'clone-sr-only', data?.label || 'Medicine records'));
    const head = element('thead');
    const headRow = element('tr');
    const sortButtons = new Map();
    visibleColumns.forEach(column => {
      const numberColumn = column.kind === 'number' && normalized(column.name) !== 'item_coverage_ratio';
      const th = element('th', numberColumn ? 'clone-number' : '');
      th.scope = 'col';
      th.setAttribute('aria-sort', state.sort === column.name ? state.ascending ? 'ascending' : 'descending' : 'none');
      const button = element('button', 'clone-sort');
      button.type = 'button';
      button.setAttribute('aria-label', `Sort by ${column.label}, ${state.sort === column.name && state.ascending ? 'descending' : 'ascending'}`);
      const arrow = element('span', 'clone-sort-arrow', state.sort === column.name ? state.ascending ? '↑' : '↓' : '↕');
      arrow.setAttribute('aria-hidden', 'true');
      button.append(element('span', '', column.label), arrow);
      button.addEventListener('click', () => {
        state.ascending = state.sort === column.name ? !state.ascending : true;
        state.sort = column.name;
        state.page = 0;
        render(column.name);
      });
      sortButtons.set(column.name, button);
      th.appendChild(button);
      headRow.appendChild(th);
    });
    head.appendChild(headRow);
    table.appendChild(head);
    const body = element('tbody');
    rows.slice(first, first + pageSize).forEach(row => {
      const tr = element('tr');
      if (selectedColumn && row.values[selectedColumn.index] === true) tr.classList.add('clone-selected');
      else if (coverageColumn && typeof row.values[coverageColumn.index] === 'number' && row.values[coverageColumn.index] < 1) tr.classList.add('clone-funding-gap');
      visibleColumns.forEach(column => {
        const name = normalized(column.name);
        const td = element('td', column.kind === 'number' && name !== 'item_coverage_ratio' ? 'clone-number' : 'clone-plain');
        if (name === 'brand_name') td.classList.add('clone-primary');
        td.appendChild(cellValue(row.values[column.index], column));
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    if (!rows.length || !visibleColumns.length) {
      const row = element('tr');
      const cell = element('td', 'clone-empty', 'No records match the current filters.');
      cell.colSpan = Math.max(1, visibleColumns.length);
      row.appendChild(cell);
      body.appendChild(row);
    }
    table.appendChild(body);
    scroll.appendChild(table);
    frame.appendChild(scroll);
    const footer = element('div', 'clone-footer');
    const recordCount = element('span', '', rows.length ? `${numeric.format(first + 1)}–${numeric.format(Math.min(first + pageSize, rows.length))} of ${numeric.format(rows.length)} records` : '0 records');
    recordCount.setAttribute('role', 'status');
    recordCount.setAttribute('aria-live', 'polite');
    const pager = element('nav', 'clone-pagination');
    pager.setAttribute('aria-label', 'Table pagination');
    const previous = element('button', 'clone-page-button', '‹');
    previous.type = 'button';
    previous.setAttribute('aria-label', 'Previous page');
    previous.disabled = state.page === 0;
    previous.addEventListener('click', () => { state.page -= 1; render('previous'); });
    const next = element('button', 'clone-page-button', '›');
    next.type = 'button';
    next.setAttribute('aria-label', 'Next page');
    next.disabled = state.page >= totalPages - 1;
    next.addEventListener('click', () => { state.page += 1; render('next'); });
    pager.append(previous, element('span', '', `Page ${state.page + 1} of ${totalPages}`), next);
    footer.append(recordCount, pager);
    root.append(frame, footer);
    if (focusTarget === 'previous') (previous.disabled ? next : previous).focus({ preventScroll: true });
    else if (focusTarget === 'next') (next.disabled ? previous : next).focus({ preventScroll: true });
    else if (sortButtons.has(focusTarget)) sortButtons.get(focusTarget).focus({ preventScroll: true });
  }
  render();
  return () => root.remove();
}
