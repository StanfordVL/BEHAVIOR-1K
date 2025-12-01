document$.subscribe(function () {
  // Utility: build header occupancy grid to compute visual column indices
  function buildHeaderGrid(tHead) {
    const rows = Array.from(tHead.rows);
    const grid = [];
    for (let r = 0; r < rows.length; r++) {
      grid[r] = grid[r] || [];
      let c = 0;
      for (const cell of Array.from(rows[r].cells)) {
        // find first free column
        while (grid[r][c] !== undefined) c++;
        const colspan = parseInt(cell.getAttribute("colspan") || "1", 10);
        const rowspan = parseInt(cell.getAttribute("rowspan") || "1", 10);
        // mark occupied cells in grid by this cell
        for (let rr = 0; rr < rowspan; rr++) {
          grid[r + rr] = grid[r + rr] || [];
          for (let cc = 0; cc < colspan; cc++) {
            grid[r + rr][c + cc] = cell;
          }
        }
        // store the starting visual column index on the cell for convenience
        cell.__visColIndex = c;
        c += colspan;
      }
    }
    // compute total columns as max row length
    const totalCols = grid.reduce((m, row) => Math.max(m, row.length || 0), 0);
    return { grid, totalCols };
  }

  // Utility: parse cell text to number if possible; else return string
  function parseCellValue(text) {
    if (text === null || text === undefined) return { isNum: true, val: -Infinity };
    const s = String(text).trim();
    if (s === "" || s === "-" || s.toLowerCase() === "n/a") return { isNum: true, val: -Infinity };
    // Try comma removal for thousands
    const num = parseFloat(s.replace(/,/g, ""));
    if (!isNaN(num)) return { isNum: true, val: num };
    return { isNum: false, val: s.toLowerCase() };
  }

  // Sort tbody rows by visual column index col (0-based). Toggles direction.
  function sortTableByVisualColumn(table, col, forceDir = null) {
    const tbody = table.tBodies[0];
    if (!tbody) return;
    const rows = Array.from(tbody.rows);
    // determine current direction for toggling
    const currentCol = table.__sortedCol;
    let asc = true;
    if (forceDir !== null) asc = forceDir;
    else if (currentCol === col) asc = !table.__sortedAsc;
    // extract compare values
    const mapped = rows.map((row, idx) => {
      // get cell by visual column: row.cells is DOM order; use cellIndex trick
      // find the cell where cell.cellIndex === col OR fallback to row.cells[col]
      let target = null;
      for (const c of Array.from(row.cells)) {
        if (c.cellIndex === col) {
          target = c;
          break;
        }
      }
      if (!target) target = row.cells[col] || null;
      const text = target ? target.textContent.trim() : "";
      const parsed = parseCellValue(text);
      return { row, val: parsed.val, isNum: parsed.isNum, idx };
    });

    // comparator that keeps stable order for equal keys
    mapped.sort((a, b) => {
      // numeric vs numeric
      if (a.isNum && b.isNum) {
        if (a.val < b.val) return asc ? -1 : 1;
        if (a.val > b.val) return asc ? 1 : -1;
        return a.idx - b.idx;
      }
      // string compare
      if (!a.isNum && !b.isNum) {
        if (a.val < b.val) return asc ? -1 : 1;
        if (a.val > b.val) return asc ? 1 : -1;
        return a.idx - b.idx;
      }
      // mixed: prefer numeric > non-numeric (so numbers sort to top), you can invert if desired
      return a.isNum ? (asc ? -1 : 1) : (asc ? 1 : -1);
    });

    // re-append rows in sorted order
    const frag = document.createDocumentFragment();
    for (const m of mapped) frag.appendChild(m.row);
    tbody.appendChild(frag);

    // store sort state and update header classes
    table.__sortedCol = col;
    table.__sortedAsc = asc;
    updateHeaderSortIndicators(table, col, asc);
  }

  function clearHeaderSortIndicators(table) {
    if (!table.tHead) return;
    for (const th of table.tHead.querySelectorAll("th")) {
      th.classList.remove("sorted-asc", "sorted-desc");
      th.removeAttribute("aria-sort");
      // reset icon to default if present
      const icon = th.querySelector(".sort-icon");
      if (icon) {
        icon.textContent = " ⇅";
        icon.style.opacity = "0.6";
      }
    }
  }

  function updateHeaderSortIndicators(table, col, asc) {
    clearHeaderSortIndicators(table);
    if (!table.tHead) return;
    const ths = Array.from(table.tHead.querySelectorAll("th"));
    for (const th of ths) {
      if (typeof th.__visColIndex === "number" && th.__visColIndex === col) {
        th.classList.add(asc ? "sorted-asc" : "sorted-desc");
        th.setAttribute("aria-sort", asc ? "ascending" : "descending");
        // update or create icon
        let icon = th.querySelector(".sort-icon");
        if (!icon) {
          icon = document.createElement("span");
          icon.className = "sort-icon";
          icon.style.marginLeft = "6px";
          icon.style.fontSize = "0.9em";
          icon.style.opacity = "1";
          th.appendChild(icon);
        }
        icon.textContent = asc ? " ▲" : " ▼";
        icon.style.opacity = "1";
      }
    }
  }

  // Helper: ensure a sort icon element exists on a header cell
  function ensureSortIcon(th) {
    if (!th.querySelector(".sort-icon")) {
      const span = document.createElement("span");
      span.className = "sort-icon";
      span.textContent = " ⇅";
      span.style.marginLeft = "6px";
      span.style.fontSize = "0.9em";
      span.style.opacity = "0.6";
      span.setAttribute("aria-hidden", "true");
      th.appendChild(span);
    }
  }

  // Initialize sorting for a table: attach handlers to appropriate header cells.
  function initSortableTable(table) {
    // If there's no thead/tbody, skip
    if (!table.tHead || !table.tBodies || table.tBodies.length === 0) return;

    // Build header grid and total columns
    const { grid, totalCols } = buildHeaderGrid(table.tHead);

    // If multi-row header, allow sorting by the bottom header row (subcolumns).
    const headerRows = Array.from(table.tHead.rows);
    const clickableRow = headerRows.length > 1 ? headerRows[headerRows.length - 1] : headerRows[0];

    // For top-row cells with colspan>1, disable pointer events to force clicking subcolumns
    if (headerRows.length > 1) {
      for (const th of headerRows[0].cells) {
        const colspan = parseInt(th.getAttribute("colspan") || "1", 10);
        if (colspan > 1) {
          th.style.pointerEvents = "none";
          th.style.cursor = "default";
          th.title = "Click a subcolumn to sort";
        }
      }
    }

    // Attach listener to each th in clickableRow and add icon
    for (const th of Array.from(clickableRow.cells)) {
      // ensure each th has computed visual column index (from buildHeaderGrid)
      const visIdx = typeof th.__visColIndex === "number" ? th.__visColIndex : th.cellIndex;
      th.style.cursor = "pointer";
      th.tabIndex = 0;
      ensureSortIcon(th);
      th.addEventListener("click", function (evt) {
        sortTableByVisualColumn(table, visIdx);
      });
      // keyboard sorting (Enter / Space)
      th.addEventListener("keydown", function (evt) {
        if (evt.key === "Enter" || evt.key === " ") {
          evt.preventDefault();
          sortTableByVisualColumn(table, visIdx);
        }
      });
    }

    // Optionally, allow clicking top headers that do not span subcolumns (colspan==1) and add icon
    if (headerRows.length > 1) {
      for (const th of Array.from(headerRows[0].cells)) {
        const colspan = parseInt(th.getAttribute("colspan") || "1", 10);
        if (colspan === 1) {
          const visIdx = typeof th.__visColIndex === "number" ? th.__visColIndex : th.cellIndex;
          th.style.pointerEvents = "auto";
          th.style.cursor = "pointer";
          ensureSortIcon(th);
          th.addEventListener("click", function () {
            sortTableByVisualColumn(table, visIdx);
          });
        }
      }
    }

    // Initialize sort metadata
    table.__sortedCol = null;
    table.__sortedAsc = true;
  }

  // Find target tables in the doc and initialize them
  const tables = document.querySelectorAll("article table:not([class])");
  tables.forEach(function (table) {
    try {
      initSortableTable(table);
    } catch (e) {
      // silent fail for unexpected table shapes
      console.error("Failed to init table sorting", e);
    }
  });
});
