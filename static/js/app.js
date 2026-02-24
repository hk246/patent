/**
 * Global application JavaScript
 */
document.addEventListener('DOMContentLoaded', function () {

    // ── Sidebar toggle (mobile) ──
    const sidebar = document.getElementById('sidebar');
    const toggle = document.getElementById('sidebar-toggle');
    const overlay = document.getElementById('sidebar-overlay');

    function openSidebar() {
        sidebar.classList.remove('-translate-x-full');
        overlay.classList.remove('hidden');
    }
    function closeSidebar() {
        sidebar.classList.add('-translate-x-full');
        overlay.classList.add('hidden');
    }

    if (toggle) toggle.addEventListener('click', openSidebar);
    if (overlay) overlay.addEventListener('click', closeSidebar);

    // ── Auto-save settings on change ──
    const settingsForm = document.getElementById('settings-form');
    if (settingsForm) {
        let saveTimeout;
        settingsForm.addEventListener('change', function () {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(saveSettings, 300);
        });
    }

    // ── Vectorize button ──
    const btnVectorize = document.getElementById('btn-vectorize');
    if (btnVectorize) {
        btnVectorize.addEventListener('click', async function () {
            await saveSettings();
            const result = await ajaxPost('/settings/vectorize', getSettingsFormData(), {
                loadingMessage: 'ベクトル化中...'
            });
            const statusEl = document.getElementById('vec-status');
            if (result.status === 'ok') {
                showFlash(`ベクトル化完了: 自社${result.company_count}件 候補${result.db_count}件`, 'success');
                if (statusEl) {
                    statusEl.textContent = `ベクトル化済み（自社:${result.company_count}件 候補:${result.db_count}件）`;
                    statusEl.className = 'mt-2 text-xs text-center text-emerald-400';
                }
            } else if (result.error) {
                if (statusEl) {
                    statusEl.textContent = 'ベクトル化エラー';
                    statusEl.className = 'mt-2 text-xs text-center text-red-400';
                }
            }
        });
    }

    // ── Flash message auto-dismiss ──
    document.querySelectorAll('.flash-msg').forEach(function (el) {
        setTimeout(function () { if (el.parentElement) el.remove(); }, 8000);
    });

    // ── File upload drag & drop ──
    document.querySelectorAll('.upload-zone').forEach(function (zone) {
        const input = zone.querySelector('input[type="file"]');

        zone.addEventListener('dragover', function (e) {
            e.preventDefault();
            zone.classList.add('dragover');
        });
        zone.addEventListener('dragleave', function () {
            zone.classList.remove('dragover');
        });
        zone.addEventListener('drop', function (e) {
            e.preventDefault();
            zone.classList.remove('dragover');
            if (input && e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                input.dispatchEvent(new Event('change'));
            }
        });
    });
});
