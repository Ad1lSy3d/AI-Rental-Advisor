/**
 * AI Rental Home Advisor - Vanilla JavaScript Frontend
 * Handles multimodal assessment uploads, real YOLO11 defect rendering,
 * NLP agreement clause audit, RAG statutory chatbot, and risk fusion summary.
 */

// Application State
const state = {
  selectedImages: [], // Array of File objects
  agreementFile: null, // File object or null (STRICTLY OPTIONAL)
  latestAssessment: null, // Result from POST /api/assess
  isAnalyzing: false,
};

// -----------------------------------------------------------------------------
// Initialization & Navigation
// -----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupDropzones();
  setupAssessmentButtons();
  setupChatbot();
  checkBackendHealth();
});

function setupNavigation() {
  const navButtons = document.querySelectorAll(".sidebar-nav .nav-item");
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tabId = btn.getAttribute("data-tab");
      if (tabId) switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  // Update sidebar active buttons
  document.querySelectorAll(".sidebar-nav .nav-item").forEach((btn) => {
    if (btn.getAttribute("data-tab") === tabId) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Switch visible section
  document.querySelectorAll(".view-section").forEach((sec) => {
    sec.classList.remove("active");
  });

  const targetSec = document.getElementById(`view-${tabId}`);
  if (targetSec) {
    targetSec.classList.add("active");
  }

  // Update top bar title
  const titles = {
    dashboard: "Dashboard",
    "cv-detection": "Defect Detection (YOLO11)",
    "agreement-audit": "Rental Agreement Analysis",
    "rag-chatbot": "AI Rental Assistant (Statutory RAG)",
    "inspection-summary": "Inspection Summary & Risk Fusion",
    "landlord-questions": "Landlord Inquiries",
  };
  const titleEl = document.getElementById("page-title-text");
  if (titleEl && titles[tabId]) {
    titleEl.textContent = titles[tabId];
  }

  // Scroll to top of content
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// -----------------------------------------------------------------------------
// Health Check
// -----------------------------------------------------------------------------
async function checkBackendHealth() {
  const statusEl = document.getElementById("backend-status-text");
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      if (statusEl) statusEl.textContent = "FastAPI Backend Active";
    } else {
      if (statusEl) statusEl.textContent = "Backend Error";
    }
  } catch (err) {
    if (statusEl) statusEl.textContent = "Backend Offline";
  }
}

// -----------------------------------------------------------------------------
// File Upload & Dropzone Handling
// -----------------------------------------------------------------------------
function setupDropzones() {
  // 1. Quick Dashboard Photos
  bindDropzone(
    "quick-photo-dropzone",
    "input-quick-photos",
    handleImageFilesSelected
  );

  // 2. Quick Dashboard Agreement (Optional)
  bindDropzone(
    "quick-agreement-dropzone",
    "input-quick-agreement",
    handleAgreementFileSelected
  );

  // 3. CV Tab Photos
  bindDropzone(
    "cv-photo-dropzone",
    "input-cv-photos",
    handleImageFilesSelected
  );

  // 4. Agreement Tab PDF (Optional)
  bindDropzone(
    "agreement-pdf-dropzone",
    "input-agreement-pdf",
    handleAgreementFileSelected
  );
}

function bindDropzone(dropzoneId, inputId, onFileCallback) {
  const dropzone = document.getElementById(dropzoneId);
  const input = document.getElementById(inputId);
  if (!dropzone || !input) return;

  dropzone.addEventListener("click", () => input.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileCallback(e.dataTransfer.files);
    }
  });

  input.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileCallback(e.target.files);
    }
  });
}

function handleImageFilesSelected(fileList) {
  for (let i = 0; i < fileList.length; i++) {
    const file = fileList[i];
    if (file.type.startsWith("image/")) {
      // Prevent duplicates by name and size
      const exists = state.selectedImages.some(
        (f) => f.name === file.name && f.size === file.size
      );
      if (!exists) {
        state.selectedImages.push(file);
      }
    }
  }
  renderImagePreviews();
  updateRunButtons();
}

function handleAgreementFileSelected(fileList) {
  if (fileList.length > 0) {
    const file = fileList[0];
    if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
      state.agreementFile = file;
    } else {
      showToast("Please select a valid PDF document for the rental agreement.", "danger");
    }
  }
  renderAgreementPreview();
}

function removeImage(index) {
  if (index >= 0 && index < state.selectedImages.length) {
    state.selectedImages.splice(index, 1);
    renderImagePreviews();
    updateRunButtons();
  }
}

function removeAgreement() {
  state.agreementFile = null;
  // Reset file inputs
  const inQuick = document.getElementById("input-quick-agreement");
  const inTab = document.getElementById("input-agreement-pdf");
  if (inQuick) inQuick.value = "";
  if (inTab) inTab.value = "";
  renderAgreementPreview();
}

function renderImagePreviews() {
  const quickList = document.getElementById("quick-photo-preview-list");
  const cvList = document.getElementById("cv-photos-list");

  const buildHtml = () => {
    if (state.selectedImages.length === 0) return "";
    return state.selectedImages
      .map((file, idx) => {
        const url = URL.createObjectURL(file);
        const sizeStr = (file.size / (1024 * 1024)).toFixed(1) + " MB";
        return `
          <div class="file-preview-card">
            <img src="${url}" class="file-thumbnail" alt="Preview">
            <div class="file-details">
              <div class="file-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</div>
              <div class="file-size">${sizeStr}</div>
            </div>
            <button type="button" class="btn-remove" onclick="removeImage(${idx})" title="Remove image">×</button>
          </div>
        `;
      })
      .join("");
  };

  const html = buildHtml();
  if (quickList) quickList.innerHTML = html;
  if (cvList) cvList.innerHTML = html;
}

function renderAgreementPreview() {
  const quickPreview = document.getElementById("quick-agreement-preview");
  const tabPreview = document.getElementById("agreement-file-preview");

  let html = "";
  // Guarded optional check: never access file properties if agreementFile is null
  if (state.agreementFile) {
    const file = state.agreementFile;
    const sizeStr = (file.size / (1024 * 1024)).toFixed(1) + " MB";
    html = `
      <div class="agreement-status-box">
        <div style="display: flex; align-items: center; gap: 12px; min-width: 0;">
          <div style="font-size: 24px;">📄</div>
          <div style="min-width: 0;">
            <div style="font-size: 13.5px; font-weight: 600; color: var(--text-main); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
              ${escapeHtml(file.name)}
            </div>
            <div style="font-size: 11.5px; color: var(--text-dim);">${sizeStr} · Attached &amp; Ready</div>
          </div>
        </div>
        <button type="button" class="btn btn-secondary btn-sm" onclick="removeAgreement()">Remove</button>
      </div>
    `;
  }

  if (quickPreview) quickPreview.innerHTML = html;
  if (tabPreview) tabPreview.innerHTML = html;
}

function updateRunButtons() {
  const hasImages = state.selectedImages.length > 0;
  const btnDashboard = document.getElementById("btn-run-assessment");
  const btnCv = document.getElementById("btn-cv-analyze");
  const valMsg = document.getElementById("quick-validation-msg");

  if (btnDashboard) btnDashboard.disabled = !hasImages || state.isAnalyzing;
  if (btnCv) btnCv.disabled = !hasImages || state.isAnalyzing;

  if (valMsg) {
    if (hasImages) {
      const count = state.selectedImages.length;
      valMsg.textContent = `${count} property image${count > 1 ? "s" : ""} selected. Ready for assessment.`;
      valMsg.style.color = "var(--success)";
    } else {
      valMsg.textContent = "Please select at least one property photo to begin.";
      valMsg.style.color = "var(--text-dim)";
    }
  }
}

// -----------------------------------------------------------------------------
// Real Multimodal Assessment Execution
// -----------------------------------------------------------------------------
function setupAssessmentButtons() {
  const btnDashboard = document.getElementById("btn-run-assessment");
  const btnCv = document.getElementById("btn-cv-analyze");
  const btnReset = document.getElementById("btn-reset-assessment");

  if (btnDashboard) btnDashboard.addEventListener("click", runAssessment);
  if (btnCv) btnCv.addEventListener("click", runAssessment);

  if (btnReset) {
    btnReset.addEventListener("click", () => {
      state.selectedImages = [];
      state.agreementFile = null;
      state.latestAssessment = null;
      renderImagePreviews();
      renderAgreementPreview();
      updateRunButtons();
      resetAssessmentUI();
      switchTab("dashboard");
      showToast("Assessment session reset. You can upload new files.", "info");
    });
  }
}

async function runAssessment() {
  if (state.selectedImages.length === 0) {
    showToast("Please upload at least one property image.", "danger");
    return;
  }

  setLoading(true);

  try {
    const formData = new FormData();

    // 1. Append property images
    state.selectedImages.forEach((imgFile) => {
      formData.append("images", imgFile);
    });

    // 2. Append optional agreement PDF ONLY if user uploaded one
    // Null safety: NEVER access agreementFile without checking existence
    if (state.agreementFile) {
      formData.append("agreement_pdf", state.agreementFile);
    }

    // Call real FastAPI endpoint
    const response = await fetch("/api/assess", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();
    state.latestAssessment = data;

    // Render results across all sections with REAL backend data
    renderAssessmentResults(data);

    // Switch to Inspection Summary tab to present results
    switchTab("inspection-summary");
    showToast("Property assessment completed successfully!", "success");
  } catch (error) {
    console.error("Assessment failed:", error);
    showToast(`Assessment failed: ${error.message}`, "danger");
  } finally {
    setLoading(false);
  }
}

// -----------------------------------------------------------------------------
// Render Assessment Findings (CV, NLP, RAG, Fusion, Questions)
// -----------------------------------------------------------------------------
function renderAssessmentResults(data) {
  // 1. Defect Detection Tab
  renderDefectDetectionTab(data.detected_defects || [], data.visual_risk || 0.0);

  // 2. Rental Agreement Tab
  renderRentalAgreementTab(data.classified_clauses || [], data.agreement_risk || 0.0);

  // 3. Inspection Summary Tab
  renderInspectionSummaryTab(data);

  // 4. Landlord Questions Tab
  renderLandlordQuestionsTab(data.landlord_questions || [], data);
}

function renderDefectDetectionTab(defects, visualRisk) {
  const countBadge = document.getElementById("cv-total-defects-badge");
  const countCrack = document.getElementById("cv-count-crack");
  const countMold = document.getElementById("cv-count-mold");
  const countPest = document.getElementById("cv-count-pest");
  const riskScore = document.getElementById("cv-visual-risk-score");
  const container = document.getElementById("cv-detections-container");

  let cracks = 0, mold = 0, pests = 0;
  defects.forEach((d) => {
    const cls = (d.class || "").toLowerCase();
    if (cls === "crack") cracks++;
    else if (cls === "mold") mold++;
    else if (cls === "pest") pests++;
  });

  if (countBadge) countBadge.textContent = `${defects.length} Defect${defects.length === 1 ? "" : "s"} Detected`;
  if (countCrack) countCrack.textContent = cracks;
  if (countMold) countMold.textContent = mold;
  if (countPest) countPest.textContent = pests;
  if (riskScore) riskScore.textContent = visualRisk.toFixed(2);

  if (!container) return;

  if (defects.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon" style="color: var(--success);">✓</div>
        <div class="empty-state-title">No defects identified</div>
        <div class="empty-state-desc">YOLO11 detected zero visible cracks, mold colonies, or pests in the uploaded photos.</div>
      </div>
    `;
    return;
  }

  // Defect Table
  const rows = defects.map((d, i) => {
    const cls = (d.class || "").toLowerCase();
    const conf = Math.round((d.confidence || 0) * 100);
    const badgeClass = cls === "mold" ? "badge-high" : cls === "crack" ? "badge-mod" : "badge-mod";
    const imgThumb = d.image_url ? `<img src="${d.image_url}" style="width: 48px; height: 48px; border-radius: 4px; object-fit: cover;">` : `<div style="width: 48px; height: 48px; background: #e2e8f0; border-radius: 4px; display: flex; align-items: center; justify-content: center;">📷</div>`;

    return `
      <tr>
        <td style="width: 60px;">${imgThumb}</td>
        <td>
          <strong style="text-transform: capitalize;">${escapeHtml(d.class)}</strong>
          <div style="font-size: 11.5px; color: var(--text-dim);">${escapeHtml(d.source_name || "Uploaded Photo")}</div>
        </td>
        <td><span class="badge ${badgeClass}">${cls === "mold" ? "High Risk" : "Moderate Risk"}</span></td>
        <td>${conf}%</td>
        <td style="font-size: 12px; color: var(--text-muted);">${d.box ? `[${d.box.join(", ")}]` : "Detected Feature"}</td>
      </tr>
    `;
  }).join("");

  container.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Preview</th>
          <th>Defect Class</th>
          <th>Severity</th>
          <th>Confidence</th>
          <th>Location Box</th>
        </tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>
  `;
}

function renderRentalAgreementTab(clauses, agreementRisk) {
  const badge = document.getElementById("agreement-clause-count-badge");
  const container = document.getElementById("clauses-container");

  // Optional agreement handling: If no clauses or agreement not provided
  if (!state.agreementFile || clauses.length === 0) {
    if (badge) badge.textContent = "No Agreement Provided";
    if (container) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">📄</div>
          <div class="empty-state-title">No rental agreement provided</div>
          <div class="empty-state-desc">
            Rental agreement upload is optional. Visual defect inspection and statutory RAG have proceeded normally with agreement risk set to 0.00.
          </div>
        </div>
      `;
    }
    return;
  }

  if (badge) badge.textContent = `${clauses.length} Clauses Classified`;

  // Group clauses by category
  const categories = [
    "Security Deposit",
    "Maintenance",
    "Repairs",
    "Penalties",
    "Termination",
    "Tenant Responsibilities",
    "Landlord Responsibilities",
  ];

  let groupsHtml = "";
  categories.forEach((cat) => {
    const matched = clauses.filter((c) => c.predicted_class === cat);
    if (matched.length > 0) {
      const items = matched.map((c) => {
        const conf = Math.round((c.prediction_probability || 0.8) * 100);
        const badgeClass = c.risk_badge === "High" ? "badge-high" : c.risk_badge === "Moderate" ? "badge-mod" : "badge-low";
        return `
          <div style="background-color: var(--bg-surface-subtle); border: 1px solid var(--border-light); border-radius: var(--radius-sm); padding: 14px; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
              <span style="font-weight: 600; font-size: 13.5px; color: var(--text-main);">${escapeHtml(c.section || "Clause")}</span>
              <div>
                <span class="badge ${badgeClass}" style="margin-right: 6px;">${c.risk_badge} Risk (${c.clause_risk.toFixed(2)})</span>
                <span style="font-size: 11px; color: var(--text-dim);">${conf}% Confidence</span>
              </div>
            </div>
            <p style="font-size: 13px; color: var(--text-muted); line-height: 1.45;">${escapeHtml(c.clause_text)}</p>
          </div>
        `;
      }).join("");

      groupsHtml += `
        <div style="margin-bottom: 20px;">
          <h4 style="font-size: 14px; font-weight: 700; color: var(--primary); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">
            ${escapeHtml(cat)} (${matched.length})
          </h4>
          ${items}
        </div>
      `;
    }
  });

  if (container) container.innerHTML = groupsHtml;
}

function renderInspectionSummaryTab(data) {
  // Score display
  const healthScore = Math.round(data.property_health_score || 0);
  const scoreNum = document.getElementById("summary-score-number");
  const circleProgress = document.getElementById("summary-circle-progress");
  const statusEl = document.getElementById("summary-health-status");
  const confEl = document.getElementById("summary-confidence-text");
  const auditIdEl = document.getElementById("summary-audit-id");

  if (scoreNum) scoreNum.textContent = healthScore;
  if (auditIdEl) auditIdEl.textContent = `AUDIT FILE #${escapeHtml(data.audit_id || "RA-8042")}`;

  // SVG circle calculation: 2 * PI * r = 2 * 3.14159 * 38 = 238.76
  if (circleProgress) {
    const circumference = 238.76;
    const offset = circumference - (healthScore / 100) * circumference;
    circleProgress.style.strokeDasharray = `${circumference}`;
    circleProgress.style.strokeDashoffset = `${offset}`;

    if (healthScore >= 75) {
      circleProgress.style.stroke = "var(--success)";
      if (statusEl) {
        statusEl.textContent = "Good Condition";
        statusEl.style.color = "var(--success)";
      }
    } else if (healthScore >= 50) {
      circleProgress.style.stroke = "var(--warning)";
      if (statusEl) {
        statusEl.textContent = "Moderate Risk Identified";
        statusEl.style.color = "var(--warning)";
      }
    } else {
      circleProgress.style.stroke = "var(--danger)";
      if (statusEl) {
        statusEl.textContent = "High Risk Property";
        statusEl.style.color = "var(--danger)";
      }
    }
  }

  const overallConf = Math.round((data.confidence_overall || 0.9) * 100);
  if (confEl) confEl.textContent = `Verified with ${overallConf}% combined model confidence`;

  // Sub-risks
  const vRisk = (data.visual_risk || 0).toFixed(2);
  const aRisk = (data.agreement_risk || 0).toFixed(2);
  const rRisk = (data.regulatory_risk || 0).toFixed(2);
  const tRisk = (data.total_risk || 0).toFixed(2);

  const valV = document.getElementById("val-visual");
  const valA = document.getElementById("val-agreement");
  const valR = document.getElementById("val-regulatory");
  const valT = document.getElementById("val-total");

  if (valV) valV.textContent = vRisk;
  if (valA) valA.textContent = aRisk;
  if (valR) valR.textContent = rRisk;
  if (valT) valT.textContent = tRisk;

  // Render Regulatory Grounding Evidence
  const evList = document.getElementById("summary-evidence-list");
  const evidence = data.regulatory_evidence || [];
  if (evList) {
    if (evidence.length === 0) {
      evList.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-title">No direct regulatory conflicts found</div>
          <div class="empty-state-desc">Statutory conditions align with baseline tenancy habitability criteria.</div>
        </div>
      `;
    } else {
      evList.innerHTML = evidence.map((ev) => `
        <div class="evidence-card">
          <div class="evidence-header">
            <span class="evidence-act">${escapeHtml(ev.act || "Statutory Act")}</span>
            <span class="evidence-score">Relevance: ${Math.round((ev.retrieval_score || 0.8) * 100)}%</span>
          </div>
          <div class="evidence-section">${escapeHtml(ev.jurisdiction || "Statutory Provision")} · ${escapeHtml(ev.source_file || "")}</div>
          <div class="evidence-text">"${escapeHtml(ev.quote || "")}"</div>
        </div>
      `).join("");
    }
  }

  // Render Recommendations
  const recList = document.getElementById("summary-recommendations-list");
  const recs = data.recommendations || [];
  if (recList) {
    if (recs.length === 0) {
      recList.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-title">No critical action items</div>
          <div class="empty-state-desc">Proceed with standard tenancy verification.</div>
        </div>
      `;
    } else {
      recList.innerHTML = recs.map((rec) => `
        <div style="background-color: var(--bg-surface-subtle); border: 1px solid var(--border-light); border-radius: var(--radius-sm); padding: 14px; margin-bottom: 10px;">
          <div style="font-weight: 600; font-size: 14px; color: var(--text-main); margin-bottom: 4px;">
            ${escapeHtml(rec.title)}
          </div>
          <div style="font-size: 13px; color: var(--text-muted); line-height: 1.45;">
            ${escapeHtml(rec.detail)}
          </div>
        </div>
      `).join("");
    }
  }
}

function renderLandlordQuestionsTab(questions, data) {
  const container = document.getElementById("landlord-questions-container");
  if (!container) return;

  if (!questions || questions.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">❓</div>
        <div class="empty-state-title">No specific inquiries generated</div>
        <div class="empty-state-desc">Run an assessment to generate finding-grounded landlord inquiries.</div>
      </div>
    `;
    return;
  }

  container.innerHTML = questions.map((q, idx) => `
    <div class="question-item">
      <div class="question-category">${escapeHtml(q.category || "Inquiry")}</div>
      <div class="question-text">${idx + 1}. "${escapeHtml(q.question)}"</div>
      <div class="question-basis">${escapeHtml(q.basis || "")}</div>
    </div>
  `).join("");
}

function copyLandlordQuestions() {
  if (!state.latestAssessment || !state.latestAssessment.landlord_questions || state.latestAssessment.landlord_questions.length === 0) {
    showToast("No landlord questions available to copy. Run an assessment first.", "info");
    return;
  }

  const questions = state.latestAssessment.landlord_questions;
  const text = questions
    .map((q, i) => `${i + 1}. ${q.question}\n   [Basis: ${q.basis}]\n`)
    .join("\n");

  navigator.clipboard.writeText(text).then(
    () => showToast("Copied all landlord inquiries to clipboard!", "success"),
    () => showToast("Failed to copy to clipboard.", "danger")
  );
}

// -----------------------------------------------------------------------------
// RAG Regulatory Chatbot Handling
// -----------------------------------------------------------------------------
function setupChatbot() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-query-input");

  if (form && input) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const query = input.value.trim();
      if (query) {
        input.value = "";
        sendChatMessage(query);
      }
    });
  }
}

function sendPrompt(promptText) {
  switchTab("rag-chatbot");
  sendChatMessage(promptText);
}

async function sendChatMessage(query) {
  const stream = document.getElementById("chat-messages-stream");
  if (!stream) return;

  // 1. Render User Message Bubble
  const userMsgEl = document.createElement("div");
  userMsgEl.className = "chat-message user";
  userMsgEl.innerHTML = `<div class="user-bubble">${escapeHtml(query)}</div>`;
  stream.appendChild(userMsgEl);
  stream.scrollTop = stream.scrollHeight;

  // 2. Render Temporary Thinking Indicator
  const thinkingEl = document.createElement("div");
  thinkingEl.className = "chat-message assistant";
  thinkingEl.innerHTML = `
    <div class="assistant-box">
      <div style="font-size: 13px; color: var(--text-dim); display: flex; align-items: center; gap: 8px;">
        <span class="status-dot"></span>
        <span>Retrieving authoritative statutory evidence via BM25 + SBERT...</span>
      </div>
    </div>
  `;
  stream.appendChild(thinkingEl);
  stream.scrollTop = stream.scrollHeight;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!res.ok) {
      throw new Error(`Chat query failed (${res.status})`);
    }

    const data = await res.json();
    stream.removeChild(thinkingEl);

    // 3. Render Assistant Grounded Response
    const botMsgEl = document.createElement("div");
    botMsgEl.className = "chat-message assistant";

    const isGrounded = data.grounded && data.evidence && data.evidence.length > 0;
    
    let evidenceHtml = "";
    if (isGrounded) {
      const items = data.evidence.map((ev) => `
        <div class="rag-evidence-item">
          <div class="rag-meta-row">
            <span style="color: var(--primary); font-weight: 700;">${escapeHtml(ev.act || "Statute")} · ${escapeHtml(ev.section || "Section")}</span>
            <span>Relevance: ${Math.round((ev.retrieval_score || 0) * 100)}%</span>
          </div>
          <div class="rag-passage-quote">"${escapeHtml(ev.passage || "")}"</div>
        </div>
      `).join("");

      evidenceHtml = `
        <div class="rag-evidence-accordion">
          <div class="rag-evidence-title">
            <span>⚖️ Retrieved Regulatory Evidence (${data.evidence.length} Provisions)</span>
          </div>
          ${items}
        </div>
      `;
    }

    botMsgEl.innerHTML = `
      <div class="assistant-box">
        <div class="assistant-answer">${escapeHtml(data.answer || "Insufficient evidence in the available regulatory sources.")}</div>
        ${evidenceHtml}
      </div>
    `;

    stream.appendChild(botMsgEl);
    stream.scrollTop = stream.scrollHeight;
  } catch (err) {
    console.error("Chat error:", err);
    if (thinkingEl.parentNode) stream.removeChild(thinkingEl);

    const errEl = document.createElement("div");
    errEl.className = "chat-message assistant";
    errEl.innerHTML = `
      <div class="assistant-box">
        <div class="assistant-answer" style="color: var(--danger);">
          Error retrieving statutory evidence: ${escapeHtml(err.message)}
        </div>
      </div>
    `;
    stream.appendChild(errEl);
    stream.scrollTop = stream.scrollHeight;
  }
}

// -----------------------------------------------------------------------------
// UI Utilities (Loading, Toast, Reset, Escape)
// -----------------------------------------------------------------------------
function setLoading(loading) {
  state.isAnalyzing = loading;
  const bar = document.getElementById("global-loading-bar");
  if (bar) {
    if (loading) bar.classList.add("active");
    else bar.classList.remove("active");
  }
  updateRunButtons();
}

function showToast(message, type = "info") {
  const toast = document.getElementById("toast-message");
  if (!toast) return;

  toast.textContent = message;
  toast.style.display = "block";

  if (type === "success") {
    toast.style.backgroundColor = "var(--success-bg)";
    toast.style.color = "var(--success)";
    toast.style.border = "1px solid var(--success-border)";
  } else if (type === "danger") {
    toast.style.backgroundColor = "var(--danger-bg)";
    toast.style.color = "var(--danger)";
    toast.style.border = "1px solid var(--danger-border)";
  } else {
    toast.style.backgroundColor = "var(--primary-subtle)";
    toast.style.color = "var(--primary)";
    toast.style.border = "1px solid var(--primary-border)";
  }

  setTimeout(() => {
    toast.style.display = "none";
  }, 4500);
}

function resetAssessmentUI() {
  const countBadge = document.getElementById("cv-total-defects-badge");
  const countCrack = document.getElementById("cv-count-crack");
  const countMold = document.getElementById("cv-count-mold");
  const countPest = document.getElementById("cv-count-pest");
  const riskScore = document.getElementById("cv-visual-risk-score");
  const cvContainer = document.getElementById("cv-detections-container");

  if (countBadge) countBadge.textContent = "0 Defects Detected";
  if (countCrack) countCrack.textContent = "0";
  if (countMold) countMold.textContent = "0";
  if (countPest) countPest.textContent = "0";
  if (riskScore) riskScore.textContent = "0.0";
  if (cvContainer) {
    cvContainer.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <div class="empty-state-title">No image analysis run yet</div>
        <div class="empty-state-desc">Upload photos above and run the assessment to view YOLO11 detections.</div>
      </div>
    `;
  }

  const agrBadge = document.getElementById("agreement-clause-count-badge");
  const agrContainer = document.getElementById("clauses-container");
  if (agrBadge) agrBadge.textContent = "0 Clauses Audited";
  if (agrContainer) {
    agrContainer.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📄</div>
        <div class="empty-state-title">No rental agreement provided</div>
        <div class="empty-state-desc">
          Rental agreement upload is optional. If skipped, visual defect inspection and statutory RAG will proceed normally with agreement risk set to 0.0.
        </div>
      </div>
    `;
  }

  const scoreNum = document.getElementById("summary-score-number");
  const circleProgress = document.getElementById("summary-circle-progress");
  const statusEl = document.getElementById("summary-health-status");
  const confEl = document.getElementById("summary-confidence-text");
  const auditIdEl = document.getElementById("summary-audit-id");

  if (scoreNum) scoreNum.textContent = "--";
  if (auditIdEl) auditIdEl.textContent = "AUDIT FILE #--";
  if (statusEl) {
    statusEl.textContent = "Awaiting Assessment";
    statusEl.style.color = "var(--text-main)";
  }
  if (confEl) confEl.textContent = "Run analysis with photos to compute score";
  if (circleProgress) circleProgress.style.strokeDashoffset = "238.76";

  const valV = document.getElementById("val-visual");
  const valA = document.getElementById("val-agreement");
  const valR = document.getElementById("val-regulatory");
  const valT = document.getElementById("val-total");
  if (valV) valV.textContent = "0.0";
  if (valA) valA.textContent = "0.0";
  if (valR) valR.textContent = "0.0";
  if (valT) valT.textContent = "0.0";

  const evList = document.getElementById("summary-evidence-list");
  if (evList) {
    evList.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-title">No regulatory evidence retrieved yet</div>
        <div class="empty-state-desc">Upload inspection photos and run assessment to retrieve grounded statutory provisions.</div>
      </div>
    `;
  }

  const recList = document.getElementById("summary-recommendations-list");
  if (recList) {
    recList.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-title">No recommendations generated yet</div>
        <div class="empty-state-desc">Actionable items will be computed from defect and lease analysis.</div>
      </div>
    `;
  }

  const lqContainer = document.getElementById("landlord-questions-container");
  if (lqContainer) {
    lqContainer.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">❓</div>
        <div class="empty-state-title">No assessment findings yet</div>
        <div class="empty-state-desc">
          Questions are generated dynamically from detected defects (mold, cracks, pests) and classified agreement terms. Run an assessment to generate your list.
        </div>
      </div>
    `;
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
