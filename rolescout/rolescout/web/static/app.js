// Inline Save / Applied / Hide actions on job cards, no page reload.
(function () {
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";

  async function setStatus(jobId, status, reason) {
    const res = await fetch(`/app/jobs/${jobId}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
      body: JSON.stringify({ status, reason }),
      credentials: "same-origin",
    });
    if (res.status === 401) { window.location = "/login"; return false; }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      alert(body.error || "Something went wrong. Reload and try again.");
      return false;
    }
    return true;
  }

  function closeMenus(except) {
    document.querySelectorAll(".menu.open").forEach((m) => {
      if (m !== except) { m.classList.remove("open"); m.querySelector("[data-menu]")?.setAttribute("aria-expanded", "false"); }
    });
  }

  function toggleButton(btn, on, onLabel, offLabel) {
    btn.classList.toggle("is-on", on);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    const span = btn.querySelector("span");
    if (span) span.textContent = on ? onLabel : offLabel;
  }

  function removeCard(card) {
    card.classList.add("removing");
    setTimeout(() => card.remove(), 250);
  }

  document.addEventListener("click", async (ev) => {
    const menuBtn = ev.target.closest("[data-menu]");
    if (menuBtn) {
      const menu = menuBtn.closest(".menu");
      const open = !menu.classList.contains("open");
      closeMenus(menu);
      menu.classList.toggle("open", open);
      menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
      return;
    }
    if (!ev.target.closest(".menu")) closeMenus(null);

    const btn = ev.target.closest("[data-action]");
    if (!btn) return;
    const card = btn.closest("[data-job]");
    if (!card) return;
    const jobId = card.dataset.job;
    const action = btn.dataset.action;
    const tab = new URLSearchParams(window.location.search).get("tab") || "matches";
    btn.disabled = true;
    try {
      if (action === "saved" || action === "applied") {
        const on = !btn.classList.contains("is-on");
        if (await setStatus(jobId, on ? action : "clear")) {
          card.querySelectorAll('[data-action="saved"],[data-action="applied"]').forEach((b) => {
            const isThis = b === btn;
            const state = isThis ? on : false;
            if (b.dataset.action === "saved") toggleButton(b, state, "Saved", "Save");
            else toggleButton(b, state, "Applied", "Mark applied");
          });
          if (!on && (tab === "saved" || tab === "applied")) removeCard(card);
        }
      } else if (action === "hidden") {
        if (await setStatus(jobId, "hidden", btn.dataset.reason)) removeCard(card);
      } else if (action === "clear") {
        if (await setStatus(jobId, "clear")) removeCard(card);
      }
    } finally {
      btn.disabled = false;
    }
  });

  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeMenus(null); });
})();
